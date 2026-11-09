"""LeRobot ACT as a ggml-compiler program: `ggml-compiler compile examples/act/export.py:program --out out/act`.

The compiled program maps normalized state and camera tensors to the normalized action chunk,
exactly like `ACTPolicy.model`. Normalization stays with the caller (LeRobot's processors); the
checkpoint's statistics travel in the GGUF as a `vla` serving manifest (docs/SERVING.md), so a host
such as vla.cpp's vla-server applies them without model-specific code.
"""

import json
import os
from pathlib import Path

import torch

import ggml_compiler
from ggml_compiler import vla

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


def checkpoint_dir(checkpoint: str = CHECKPOINT) -> Path:
    """The local directory of a checkpoint: as given, or the Hub snapshot."""
    if Path(checkpoint).is_dir():
        return Path(checkpoint)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint))


def normalizer_stats(checkpoint: str = CHECKPOINT) -> dict[str, dict[str, list[float]]]:
    """Feature name -> {"mean": [...], "std": [...]} from the checkpoint's LeRobot processor
    pipelines (`policy_preprocessor.json` and `policy_postprocessor.json` name the step whose
    safetensors file holds the statistics). Only MEAN_STD features are returned."""
    from safetensors import safe_open

    d = checkpoint_dir(checkpoint)
    stats: dict[str, dict[str, list[float]]] = {}
    for pipeline in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in json.loads((d / pipeline).read_text())["steps"]:
            if step["registry_name"] not in ("normalizer_processor", "unnormalizer_processor"):
                continue
            cfg = step["config"]
            with safe_open(str(d / step["state_file"]), "pt") as f:
                for feat, spec in cfg["features"].items():
                    if cfg["norm_map"].get(spec["type"]) != "MEAN_STD":
                        continue
                    stats[feat] = {k: f.get_tensor(f"{feat}.{k}").flatten().tolist() for k in ("mean", "std")}
    return stats


def serving_manifest(policy, checkpoint: str = CHECKPOINT, image_size: tuple[int, int] | None = None) -> dict:
    """The `vla` manifest for `program()`: ImageNet-normalized cameras at the compiled size, the
    checkpoint's state and action statistics. LeRobot normalizes images with per-camera dataset
    statistics only when its norm_map says so; ACT's defaults keep the ImageNet constants."""
    config = policy.config
    cams = list(config.image_features)
    _, h, w = config.image_features[cams[0]].shape
    if image_size is not None:
        h, w = image_size
    stats = normalizer_stats(checkpoint)
    images = []
    for cam in cams:
        s = stats.get(cam)
        mean, std = (s["mean"], s["std"]) if s else (vla.IMAGENET_MEAN, vla.IMAGENET_STD)
        images.append(vla.image(cam, h, w, mean=mean, std=std))
    st, ac = stats.get("observation.state", {}), stats.get("action", {})
    return vla.manifest(
        images=images,
        state=vla.state("state", dim=config.robot_state_feature.shape[0], mean=st.get("mean"), std=st.get("std")),
        actions=vla.actions("actions", chunk=config.chunk_size, dim=config.action_feature.shape[0],
                            mean=ac.get("mean"), std=ac.get("std")),
    )


def program(checkpoint: str = CHECKPOINT, image_size: tuple[int, int] | None = None) -> ggml_compiler.Program:
    policy = load(checkpoint)
    cams = list(policy.config.image_features)
    return ggml_compiler.Program.single(ACTInfer(policy.model).eval(), example_inputs(policy.config, image_size),
                                        input_names=["state", *cams], output_names=["actions"],
                                        metadata={"vla": serving_manifest(policy, checkpoint, image_size)})
