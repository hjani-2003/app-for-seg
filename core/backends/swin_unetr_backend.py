from pathlib import Path

import torch
import yaml
from monai.networks.nets import SwinUNETR

from core.backends.checkpoints import find_checkpoint

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "swin_unetr"


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


def build_model(device):
    model_cfg, _ = _load_config()

    model = SwinUNETR(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        feature_size=int(model_cfg.get("feature_size", 48)),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        dropout_path_rate=float(model_cfg.get("dropout_path_rate", 0.0)),
        use_v2=bool(model_cfg.get("use_v2", False)),
    ).to(device)

    checkpoint = find_checkpoint(MODEL_DIR)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict)

    model.eval()
    return model, checkpoint
