import numpy as np


def normalize_for_model(volume_4ch):
    """Per-channel z-score over nonzero voxels, matching MONAI's
    NormalizeIntensityd(nonzero=True, channel_wise=True) used to train
    the SwinUNETR/MambaVision checkpoints. Zero voxels are left at 0."""
    normalized = volume_4ch.astype(np.float32, copy=True)

    for channel in normalized:
        mask = channel != 0
        if not mask.any():
            continue

        values = channel[mask]
        mean = values.mean()
        std = values.std()
        if std == 0:
            std = 1.0

        channel[mask] = (values - mean) / std

    return normalized
