"""3D connected-component lesion detection + per-lesion 2D measurement.

Bridges rano_measure.regions (raw labels -> boolean region mask) and
rano_measure.geometry (2D slice -> bidimensional diameters): finds 3D
connected components within a region mask, then measures each component
on whichever of a shortlist of candidate slices yields the largest
diameter product.

Geometry runs in raw voxel/data space -- the same (row, col) convention
as `np.take(volume, idx, axis=axis)` used elsewhere in this app (see
ui/main_window.py:update_view / NOTES.md) -- NOT the flipped/rotated
space `MRIViewer._render()` uses only for on-screen display. Mapping a
measured line back into display space is a Phase 3+ (UI wiring) concern.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from rano_measure.geometry import find_bidimensional_diameters

DEFAULT_TOP_N_CANDIDATE_SLICES = 5


@dataclass
class Lesion:
    id: int
    region_type: str  # "CE" or "nonCE"
    slice_index: int
    major_line: tuple  # ((r1, c1), (r2, c2)) in voxel coords
    minor_line: tuple
    major_mm: float
    minor_mm: float
    product_mm2: float
    measurable: bool


def find_lesions(
    region_mask,
    spacing,
    axis,
    region_type,
    *,
    top_n_candidate_slices=DEFAULT_TOP_N_CANDIDATE_SLICES,
    **geometry_kwargs,
):
    """Find 3D-connected-component lesions in `region_mask` and measure
    each on its best candidate slice.

    region_mask: 3D boolean array (from rano_measure.regions.build_region_mask).
    spacing: (sag_mm, cor_mm, ax_mm) -- full volume spacing, RAS+ axis
        order (see NOTES.md / core.constants.PLANE_AXES).
    axis: volume axis being sliced (0=sagittal, 1=coronal, 2=axial).
    region_type: "CE" or "nonCE", stamped onto the resulting Lesion
        objects -- not used for geometry.
    top_n_candidate_slices: per component, only the N slices with the
        largest in-plane cross-sectional area are searched, rather than
        every slice -- keeps this fast and avoids picking a degenerate
        single-voxel outlier slice.

    Returns one Lesion per connected component (its best-by-product
    candidate slice), regardless of whether that best result clears the
    measurable threshold -- callers (rano_measure.burden) are responsible
    for filtering to `measurable` lesions when selecting targets, so lesion
    counts here reflect true detected-component counts.
    """
    labeled, num_components = ndimage.label(region_mask)
    in_plane_spacing = tuple(s for a, s in enumerate(spacing) if a != axis)

    lesions = []
    next_id = 1
    for component_id in range(1, num_components + 1):
        component_mask = labeled == component_id
        best = _measure_best_slice(
            component_mask, in_plane_spacing, axis, top_n_candidate_slices, geometry_kwargs
        )
        if best is None:
            continue
        slice_index, measurement = best
        lesions.append(
            Lesion(
                id=next_id,
                region_type=region_type,
                slice_index=slice_index,
                major_line=measurement.major_line,
                minor_line=measurement.minor_line,
                major_mm=measurement.major_mm,
                minor_mm=measurement.minor_mm,
                product_mm2=measurement.product_mm2,
                measurable=measurement.measurable,
            )
        )
        next_id += 1

    return lesions


def _measure_best_slice(component_mask, in_plane_spacing, axis, top_n, geometry_kwargs):
    other_axes = tuple(a for a in range(3) if a != axis)
    areas = component_mask.sum(axis=other_axes)
    nonzero_idx = np.nonzero(areas)[0]
    if len(nonzero_idx) == 0:
        return None

    order = np.argsort(areas[nonzero_idx])[::-1]
    candidate_idx = nonzero_idx[order][:top_n]

    best_measurement = None
    best_slice_index = None
    for idx in candidate_idx:
        slice_mask = np.take(component_mask, idx, axis=axis)
        measurement = find_bidimensional_diameters(slice_mask, in_plane_spacing, **geometry_kwargs)
        if best_measurement is None or measurement.product_mm2 > best_measurement.product_mm2:
            best_measurement = measurement
            best_slice_index = int(idx)

    return best_slice_index, best_measurement
