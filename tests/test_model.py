"""Model tests. They use the built-in architecture presets, so no download is needed. Run with: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from plant_cls.model import build_model, default_img_size, model_from_ckpt, resolve_hub_id  # noqa: E402


def test_name_resolution():
    assert resolve_hub_id("cvt-13") == "microsoft/cvt-13"
    assert resolve_hub_id("cvt-21-384-22k") == "microsoft/cvt-21-384-22k"
    assert resolve_hub_id("cvt-w24") == "microsoft/cvt-w24-384-22k"
    assert resolve_hub_id("some-org/my-cvt") == "some-org/my-cvt"
    assert default_img_size("cvt-13") == 224
    assert default_img_size("cvt-21-384-22k") == 384


def test_forward_shape_and_labels():
    m = build_model("cvt-13", ["a", "b", "c"], pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)
    assert m.config.id2label[1] == "b"


def test_other_resolution():
    # CvT uses convolutional token embeddings and no positional embeddings
    m = build_model("cvt-13", ["a", "b"], pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 192, 192))
    assert out.shape == (1, 2)


def test_checkpoint_roundtrip():
    m = build_model("cvt-13", ["a", "b"], pretrained=False).eval()
    ck = {"state_dict": m.state_dict(), "hf_config": m.config.to_dict()}
    m2 = model_from_ckpt(ck, torch.device("cpu"))
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-5)
