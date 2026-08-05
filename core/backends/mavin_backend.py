import sys
from pathlib import Path

import torch
import yaml

from core.backends.checkpoints import find_checkpoint
from core.backends.errors import ModelUnavailableError

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "mavin-hpc"


def _load_config():
    model_cfg = yaml.safe_load((MODEL_DIR / "configs" / "model.yaml").read_text())["model"]
    train_cfg = yaml.safe_load((MODEL_DIR / "configs" / "train.yaml").read_text())["data"]
    return model_cfg, train_cfg


def get_roi_and_overlap():
    _, train_cfg = _load_config()
    roi = list(train_cfg.get("roi", [128, 128, 128]))
    overlap = float(train_cfg.get("infer_overlap", 0.5))
    sw_batch_size = int(train_cfg.get("sw_batch_size", 1))
    return roi, overlap, sw_batch_size


def _import_mambavision_unet():
    model_root = str(MODEL_DIR)
    inserted = model_root not in sys.path
    if inserted:
        sys.path.insert(0, model_root)
    try:
        from src.mambaVisionUNet import MambaVisionUNet
    except ImportError as exc:
        raise ModelUnavailableError(
            f"MambaVision failed to import ({exc}). It needs the mamba_ssm CUDA "
            "kernels (GPU-only, no CPU fallback) — check that mamba_ssm was built "
            "against this environment's torch/CUDA version."
        ) from exc
    finally:
        if inserted:
            sys.path.remove(model_root)
    return MambaVisionUNet


def build_model(device, roi):
    MambaVisionUNet = _import_mambavision_unet()
    model_cfg, _ = _load_config()
    depths = model_cfg.get("encoder", {}).get("depths", 2)

    model = MambaVisionUNet(
        img_size=tuple(roi),
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        feature_size=int(model_cfg.get("feature_size", 48)),
        depths=int(depths[0] if isinstance(depths, list) else depths),
        num_heads=int(model_cfg.get("num_heads", 16)),
        d_state=int(model_cfg.get("d_state", 16)),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
    ).to(device)

    checkpoint = find_checkpoint(MODEL_DIR)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict)

    model.eval()
    return model, checkpoint
