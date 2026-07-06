import numpy as np

from ui.style import SEGMENTATION_COLORS, hex_to_rgb01

_CLASS_RGB = {cls: hex_to_rgb01(color) for cls, color in SEGMENTATION_COLORS.items()}


def colorize_mask(mask_slice):
    rgb = np.zeros((*mask_slice.shape, 3), dtype=np.float32)
    for class_id, color in _CLASS_RGB.items():
        rgb[mask_slice == class_id] = color
    return rgb


def overlay_image_with_mask(base, mask_slice, alpha=0.45):
    gray = np.repeat(base[:, :, None], 3, axis=2).astype(np.float32)
    colored = colorize_mask(mask_slice)

    present = mask_slice > 0
    blended = gray.copy()
    blended[present] = (1 - alpha) * gray[present] + alpha * colored[present]
    return blended
