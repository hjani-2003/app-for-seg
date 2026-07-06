def find_checkpoint(model_dir):
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    candidates = sorted(ckpt_dir.glob("*.pth"))
    return candidates[0] if candidates else None
