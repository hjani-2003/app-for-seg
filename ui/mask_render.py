import numpy as np

from ui.style import SEGMENTATION_COLORS, hex_to_rgb01
from ui.synthseg_lut import LUT_ARRAY, MAX_LABEL

_CLASS_RGB = {cls: hex_to_rgb01(color) for cls, color in SEGMENTATION_COLORS.items()}


def colorize_mask(mask_slice):
    rgb = np.zeros((*mask_slice.shape, 3), dtype=np.float32)
    for class_id, color in _CLASS_RGB.items():
        rgb[mask_slice == class_id] = color
    return rgb


def colorize_label_map(mask_slice):
    """Colour a SynthSeg slice by indexing the FreeSurfer lookup table.

    A parcellated slice can carry ~100 distinct labels, so this indexes a
    precomputed table rather than looping per class the way colorize_mask does.
    """
    clipped = np.clip(mask_slice.astype(np.int32), 0, MAX_LABEL)
    return LUT_ARRAY[clipped]


def _blend(base_rgb, colored, present, alpha):
    base_rgb[present] = (1 - alpha) * base_rgb[present] + alpha * colored[present]
    return base_rgb


def overlay_image_with_masks(
    base, tumor_slice=None, synthseg_slice=None, tumor_alpha=0.45, synthseg_alpha=0.30
):
    """Composite either or both overlays onto a grayscale slice.

    SynthSeg goes down first and at a lower alpha, so the tumour mask stays
    readable on top of it when both are shown.
    """
    blended = np.repeat(base[:, :, None], 3, axis=2).astype(np.float32)

    if synthseg_slice is not None:
        blended = _blend(
            blended,
            colorize_label_map(synthseg_slice),
            synthseg_slice > 0,
            synthseg_alpha,
        )

    if tumor_slice is not None:
        blended = _blend(
            blended, colorize_mask(tumor_slice), tumor_slice > 0, tumor_alpha
        )

    return blended
