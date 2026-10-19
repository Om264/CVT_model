"""CvT (Convolutional vision Transformer) via Hugging Face Transformers, plus EMA and checkpoint helpers.

CvT is not part of timm, so the backbone comes from `transformers.CvtForImageClassification`.
`CvTClassifier` wraps it so that `forward(tensor) -> logits`, which keeps DDP, EMA, AMP and ONNX export simple.
"""
from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, CvtConfig, CvtForImageClassification

from .utils import log

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Architecture presets from the CvT paper, used for --no-pretrained so that no download is needed.
# All other config fields keep the CvT-13 defaults of transformers' CvtConfig.
_PRESETS = {
    "cvt-13": dict(embed_dim=[64, 192, 384], num_heads=[1, 3, 6], depth=[1, 2, 10], drop_path_rate=[0.0, 0.0, 0.1]),
    "cvt-21": dict(embed_dim=[64, 192, 384], num_heads=[1, 3, 6], depth=[1, 4, 16], drop_path_rate=[0.0, 0.0, 0.1]),
    "cvt-w24": dict(embed_dim=[192, 768, 1024], num_heads=[3, 12, 16], depth=[2, 2, 20], drop_path_rate=[0.0, 0.0, 0.3]),
}


def resolve_hub_id(name: str) -> str:
    """Map a short name to a Hugging Face Hub id.

    'cvt-13' -> 'microsoft/cvt-13', 'cvt-21-384-22k' -> 'microsoft/cvt-21-384-22k',
    'cvt-w24' -> 'microsoft/cvt-w24-384-22k'. Hub ids ('org/name') and local folders are used as given.
    """
    if "/" in name or os.path.isdir(name):
        return name
    if name == "cvt-w24":
        return "microsoft/cvt-w24-384-22k"
    return f"microsoft/{name}"


def _arch_key(name: str):
    key = name.lower().split("/")[-1]
    for k in ("cvt-w24", "cvt-21", "cvt-13"):
        if k in key:
            return k
    return None


def default_img_size(name: str) -> int:
    key = name.lower().split("/")[-1]
    return 384 if ("384" in key or "w24" in key) else 224


class CvTClassifier(nn.Module):
    """Wraps CvtForImageClassification: forward(pixel_values tensor) -> logits tensor."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    @property
    def config(self):
        return self.model.config

    def forward(self, x):
        return self.model(pixel_values=x, return_dict=True).logits


def build_model(name, class_names, pretrained=True, drop_rate=None, drop_path=None):
    """Create a CvT classifier with len(class_names) outputs.

    name:       'cvt-13' | 'cvt-21' | 'cvt-w24' | 'cvt-13-384' | ... | any Hub id / local folder
    pretrained: load ImageNet weights from the Hub (the classification head is re-initialised)
    drop_rate / drop_path: optional overrides (drop_path applies to the last stage); None keeps the defaults
    """
    id2label = {i: c for i, c in enumerate(class_names)}
    label2id = {c: i for i, c in id2label.items()}
    cfg_kwargs = dict(num_labels=len(class_names), id2label=id2label, label2id=label2id)
    if drop_rate is not None:
        cfg_kwargs["drop_rate"] = [float(drop_rate)] * 3
    if drop_path is not None:
        cfg_kwargs["drop_path_rate"] = [0.0, 0.0, float(drop_path)]

    if pretrained:
        hf = CvtForImageClassification.from_pretrained(
            resolve_hub_id(name), ignore_mismatched_sizes=True, **cfg_kwargs
        )
    else:
        key = _arch_key(name)
        if key is not None:
            cfg = CvtConfig(**{**_PRESETS[key], **cfg_kwargs})
        else:  # custom Hub id / local folder: read its config
            cfg = CvtConfig.from_pretrained(resolve_hub_id(name), **cfg_kwargs)
        hf = CvtForImageClassification(cfg)
    return CvTClassifier(hf)


def preprocess_cfg(name, pretrained=True):
    """Returns (mean, std, default_img_size). Read from the checkpoint's image processor when possible."""
    mean, std, size = IMAGENET_MEAN, IMAGENET_STD, default_img_size(name)
    if pretrained:
        try:
            proc = AutoImageProcessor.from_pretrained(resolve_hub_id(name))
            if proc.image_mean is not None and proc.image_std is not None:
                mean, std = tuple(proc.image_mean), tuple(proc.image_std)
            s = proc.size
            if isinstance(s, dict):
                s = s.get("shortest_edge") or s.get("height")
            if isinstance(s, int):
                size = s
        except Exception as e:  # offline / local checkpoint without a processor config
            log(f"[warn] could not read the image processor ({type(e).__name__}); using ImageNet mean/std")
    return tuple(mean), tuple(std), int(size)


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay):
        self.decay, self.step = decay, 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def load_ckpt(path):
    """Load a checkpoint written by training (weights + HF config + class names + preprocessing)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from_ckpt(ck, device):
    """Rebuild the model from a checkpoint alone (no Hub access needed)."""
    cfg = CvtConfig.from_dict(copy.deepcopy(ck["hf_config"]))
    m = CvTClassifier(CvtForImageClassification(cfg))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
