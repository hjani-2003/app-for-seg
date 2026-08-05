import numpy as np
from skimage.draw import disk

from rano_measure.lesion import find_lesions

ISO_SPACING_3D = (1.0, 1.0, 1.0)  # (sag_mm, cor_mm, ax_mm)
AXIAL_AXIS = 2


def _sphere_like_blob(shape, center_2d, z_range, radius):
    # Same disk radius across a small z-range: enough to exercise 3D
    # connectivity + candidate-slice selection without needing a true
    # sphere rasterizer.
    mask = np.zeros(shape, dtype=bool)
    rr, cc = disk(center_2d, radius, shape=shape[:2])
    for z in range(*z_range):
        mask[rr, cc, z] = True
    return mask


def test_two_separate_blobs_yield_two_lesions_with_region_type():
    shape = (80, 80, 20)
    region_mask = np.zeros(shape, dtype=bool)
    region_mask |= _sphere_like_blob(shape, (20, 20), (2, 8), radius=15)  # large, measurable
    region_mask |= _sphere_like_blob(shape, (60, 60), (10, 16), radius=2)  # tiny, not measurable

    lesions = find_lesions(region_mask, ISO_SPACING_3D, AXIAL_AXIS, "CE")

    assert len(lesions) == 2
    assert all(lesion.region_type == "CE" for lesion in lesions)

    by_size = sorted(lesions, key=lambda lesion: lesion.product_mm2, reverse=True)
    large, small = by_size
    assert large.measurable
    assert large.major_mm >= 10.0 and large.minor_mm >= 10.0
    assert 2 <= large.slice_index <= 7

    assert not small.measurable


def test_empty_region_mask_yields_no_lesions():
    region_mask = np.zeros((40, 40, 10), dtype=bool)
    lesions = find_lesions(region_mask, ISO_SPACING_3D, AXIAL_AXIS, "nonCE")
    assert lesions == []


def test_lesion_ids_are_unique_and_sequential():
    shape = (80, 80, 20)
    region_mask = np.zeros(shape, dtype=bool)
    region_mask |= _sphere_like_blob(shape, (20, 20), (2, 6), radius=12)
    region_mask |= _sphere_like_blob(shape, (60, 60), (2, 6), radius=12)
    region_mask |= _sphere_like_blob(shape, (20, 60), (14, 18), radius=12)

    lesions = find_lesions(region_mask, ISO_SPACING_3D, AXIAL_AXIS, "CE")

    assert sorted(lesion.id for lesion in lesions) == [1, 2, 3]
