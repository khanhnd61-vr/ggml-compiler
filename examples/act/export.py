"""LeRobot ACT as a ggmlc program: `ggmlc compile examples/act/export.py:program --out out/act`.

The compiled program maps normalized state and camera tensors to the normalized action chunk,
exactly like `ACTPolicy.model`. Normalization stays with the caller (LeRobot's processors).
"""

import os

import torch

import ggmlc

CHECKPOINT = os.environ.get("ACT_CHECKPOINT", "khanhnd61/act_so101-multi-task-clean")


class ACTInfer(torch.nn.Module):
    """Plain tensors in and out: state, then one image per camera in config order."""

    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.act = act

    def forward(self, state, *images):
        actions, _ = self.act({"observation.state": state, "observation.images": list(images)})
        return actions


def load(checkpoint: str = CHECKPOINT):
    from lerobot.policies.act.modeling_act import ACTPolicy

    return ACTPolicy.from_pretrained(checkpoint).eval()


def example_inputs(config, image_size: tuple[int, int] | None = None) -> tuple[torch.Tensor, ...]:
    """The vla.cpp parity rig's synthetic inputs: pixel (x + 2y + 40c + 17v) & 255 for view v,
    scaled to [0, 1] and ImageNet-normalized; state 0.01 * (i + 1). Images default to the
    training resolution; ACT itself runs at any size."""
    cams = list(config.image_features)
    _, h, w = config.image_features[cams[0]].shape
    if image_size is not None:
        h, w = image_size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = []
    for v in range(len(cams)):
        px = torch.stack([(x + 2 * y + 40 * c + 17 * v) & 255 for c in range(3)]).float() / 255
        images.append(((px - mean) / std).unsqueeze(0))
    n_state = config.robot_state_feature.shape[0]
    state = 0.01 * torch.arange(1, n_state + 1, dtype=torch.float32).unsqueeze(0)
    return (state, *images)


def program(checkpoint: str = CHECKPOINT, image_size: tuple[int, int] | None = None) -> ggmlc.Program:
    policy = load(checkpoint)
    cams = list(policy.config.image_features)
    return ggmlc.Program.single(ACTInfer(policy.model).eval(), example_inputs(policy.config, image_size),
                                input_names=["state", *cams], output_names=["actions"])
