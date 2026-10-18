"""LeRobot Diffusion Policy as a ggmlc program with the whole denoising loop on device:
`ggmlc compile examples/diffusion/export.py:program --out out/diffusion`.

encode: state and camera history -> the U-Net's global conditioning vector.
step:   one scheduler step: noise prediction from the 1-D U-Net, then the DDPM/DDIM update.
finish: the n_action_steps actions starting at the current observation.

The scheduler is unrolled at compile time. Its per-step coefficients become loop scalars, and the
per-step noise DDPM adds is a program input (`step_noise`, one slice per step), so the artifact is
deterministic and the caller owns the random numbers. Normalization stays with the caller, as in
LeRobot's processors.
"""

import os

import torch

import ggmlc

CHECKPOINT = os.environ.get("DIFFUSION_CHECKPOINT", "lerobot/diffusion_pusht")


def load(checkpoint: str = CHECKPOINT, scheduler: str | None = None, num_inference_steps: int | None = None):
    """Load a checkpoint, optionally switching the noise scheduler (DDIM with fewer steps is the
    usual way to run a DDPM-trained Diffusion Policy faster)."""
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    config = DiffusionConfig.from_pretrained(checkpoint)
    if scheduler is not None:
        config.noise_scheduler_type = scheduler
    if num_inference_steps is not None:
        config.num_inference_steps = num_inference_steps
    return DiffusionPolicy.from_pretrained(checkpoint, config=config).eval()


def schedule(model) -> dict[str, list[float]]:
    """The scheduler's update, step by step, as the coefficients of

        x0 = clamp((x - sb * eps) / sa, -r, r)                (clamp only if clip_sample)
        x' = c0 * x0 + c1 * x + c2 * eps + sigma * z          (z: this step's noise)

    computed in float32 exactly as DDPMScheduler.step (variance "fixed_small") and
    DDIMScheduler.step (eta = 0) compute them, so the compiled loop does the same arithmetic.
    A zero coefficient adds an exact zero."""
    s = model.noise_scheduler
    s.set_timesteps(model.num_inference_steps)
    if model.config.prediction_type != "epsilon":
        raise NotImplementedError(f"prediction_type={model.config.prediction_type!r}")
    ddim = type(s).__name__ == "DDIMScheduler"
    if not ddim and s.config.variance_type != "fixed_small":
        raise NotImplementedError(f"DDPM variance_type={s.config.variance_type!r}")
    out: dict[str, list[float]] = {k: [] for k in ("t", "sa", "sb", "c0", "c1", "c2", "sigma")}
    zero = torch.tensor(0.0)
    for t in s.timesteps.tolist():
        a_t = s.alphas_cumprod[t]
        beta_t = 1 - a_t
        if ddim:
            prev = t - s.config.num_train_timesteps // s.num_inference_steps
            a_prev = s.alphas_cumprod[prev] if prev >= 0 else s.final_alpha_cumprod
            c0, c1, c2, sigma = a_prev ** 0.5, zero, (1 - a_prev) ** 0.5, zero
        else:
            prev = int(s.previous_timestep(t))
            a_prev = s.alphas_cumprod[prev] if prev >= 0 else s.one
            cur_alpha = a_t / a_prev
            cur_beta = 1 - cur_alpha
            c0 = (a_prev ** 0.5 * cur_beta) / beta_t
            c1 = cur_alpha ** 0.5 * (1 - a_prev) / beta_t
            c2 = zero
            sigma = torch.clamp((1 - a_prev) / (1 - a_t) * cur_beta, min=1e-20) ** 0.5 if t > 0 else zero
        for k, v in zip(out, (float(t), a_t ** 0.5, beta_t ** 0.5, c0, c1, c2, sigma)):
            out[k].append(float(v))
    return out


class Stages:
    def __init__(self, model):
        self.m = model
        cfg = model.config
        self.clip = cfg.clip_sample_range if cfg.clip_sample else None

    def encode(self, state, images):
        """state (B, n_obs, S); images (B, n_obs, n_cam, C, H, W) -> (B, global_cond_dim)."""
        return self.m._prepare_global_conditioning({"observation.state": state, "observation.images": images})

    def step(self, x, cond, z, t, sa, sb, c0, c1, c2, sigma):
        eps = self.m.unet(x, t, global_cond=cond)
        x0 = (x - sb * eps) / sa
        if self.clip is not None:
            x0 = x0.clamp(-self.clip, self.clip)
        return c0 * x0 + c1 * x + c2 * eps + sigma * z

    def finish(self, x):
        start = self.m.config.n_obs_steps - 1
        return x[:, start: start + self.m.config.n_action_steps]


def example_inputs(config, batch: int = 1, seed: int = 0) -> dict[str, torch.Tensor]:
    """Synthetic normalized inputs: images as the ACT parity rig's pattern, state 0.01 * (i + 1),
    noise from a seeded generator."""
    cams = list(config.image_features)
    c, h, w = config.image_features[cams[0]].shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    frames = []
    for s in range(config.n_obs_steps):
        views = [torch.stack([(x + 2 * y + 40 * ch + 17 * v + 5 * s) & 255 for ch in range(c)]).float() / 255
                 for v in range(len(cams))]
        frames.append(torch.stack(views))
    images = (torch.stack(frames) * 2 - 1).unsqueeze(0).expand(batch, -1, -1, -1, -1, -1).contiguous()
    n_state = config.robot_state_feature.shape[0]
    state = 0.01 * torch.arange(1, n_state + 1, dtype=torch.float32)
    state = state.expand(batch, config.n_obs_steps, n_state).contiguous()
    g = torch.Generator().manual_seed(seed)
    shape = (batch, config.horizon, config.action_feature.shape[0])
    return {"state": state, "images": images, "noise": torch.randn(shape, generator=g)}


def step_noise(model, batch: int, seed: int = 1) -> torch.Tensor:
    """Per-step DDPM noise (n_steps, B, horizon, action_dim), the way LeRobot draws it: one
    torch.randn per step with t > 0 after torch.manual_seed(seed)."""
    cfg = model.config
    n = model.num_inference_steps
    shape = (batch, cfg.horizon, cfg.action_feature.shape[0])
    torch.manual_seed(seed)
    ts = schedule(model)["t"]
    return torch.stack([torch.randn(shape) if t > 0 else torch.zeros(shape) for t in ts])


def program_from_policy(policy, inputs: dict[str, torch.Tensor] | None = None, seed: int = 1) -> ggmlc.Program:
    model = policy.diffusion
    st = Stages(model)
    inputs = inputs or example_inputs(policy.config)
    sched = schedule(model)
    n = len(sched["t"])
    B = inputs["state"].shape[0]
    zs = step_noise(model, B, seed)
    cond = st.encode(inputs["state"], inputs["images"])
    # Examples for the per-step scalars: the first step's values. The coefficients are (1,) so they
    # broadcast against x; t is (B,), one timestep per batch row as the U-Net expects.
    first = [torch.full((B,) if k == "t" else (1,), v[0]) for k, v in sched.items()]
    stages = {
        "encode": ggmlc.Stage(st.encode, (inputs["state"], inputs["images"])),
        "step": ggmlc.Stage(st.step, (inputs["noise"], cond, zs[0], *first)),
        "finish": ggmlc.Stage(st.finish, (inputs["noise"],)),
    }
    driver = ggmlc.Driver(
        inputs=["state", "images", "noise", "step_noise"],
        body=[
            ggmlc.call("encode", ["state", "images"], ["cond"]),
            ggmlc.loop(n, [ggmlc.call("step", ["x", "cond", "z", *sched], ["x"])],
                       carry={"x": "noise"}, scalars=sched, slices={"z": "step_noise"}),
            ggmlc.call("finish", ["x"], ["actions"]),
        ],
        outputs=["actions"],
        examples={**inputs, "step_noise": zs},
    )
    return ggmlc.Program(root=model, stages=stages, driver=driver)


def program(checkpoint: str = CHECKPOINT, scheduler: str | None = None,
            num_inference_steps: int | None = None) -> ggmlc.Program:
    return program_from_policy(load(checkpoint, scheduler, num_inference_steps))
