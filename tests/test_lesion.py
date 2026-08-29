import numpy as np
from skimage.draw import disk

from rano_measure.lesion import find_lesions
from rano_measure.regions import LABEL_IDS, REGION_DEFS, build_region_mask

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


def test_lesion_touching_volume_boundary_is_detected_and_measurable():
    # A component flush against the edge of the in-plane axes (rows 0-19)
    # -- exercises the same edge-touching case as
    # test_geometry.test_lesion_touching_image_boundary_is_measured_not_dropped,
    # but through the full 3D connected-components path.
    shape = (60, 60, 10)
    region_mask = np.zeros(shape, dtype=bool)
    region_mask[0:20, 0:20, 3:7] = True

    lesions = find_lesions(region_mask, ISO_SPACING_3D, AXIAL_AXIS, "CE")

    assert len(lesions) == 1
    assert lesions[0].measurable
    assert lesions[0].major_line is not None


def test_multiple_disjoint_components_of_same_raw_label_stay_separate_lesions():
    # Phase 4 edge case: multiple disjoint ET components that a clinician
    # might think of as "one lesion" -- v1 must keep them as separate
    # Lesion objects, not silently merge or drop any of them. Build this
    # through the real regions.py mapping (raw ET label -> CE region) to
    # match how the app actually gets here, not just via a pre-built mask.
    shape = (80, 80, 20)
    label_volume = np.zeros(shape, dtype=np.uint8)
    et_id = LABEL_IDS["ET"]
    rr1, cc1 = disk((20, 20), 12, shape=shape[:2])
    rr2, cc2 = disk((60, 60), 12, shape=shape[:2])
    rr3, cc3 = disk((20, 60), 12, shape=shape[:2])
    for z in range(2, 6):
        label_volume[rr1, cc1, z] = et_id
        label_volume[rr2, cc2, z] = et_id
        label_volume[rr3, cc3, z] = et_id

    region_mask = build_region_mask(label_volume, LABEL_IDS, REGION_DEFS, "CE")
    lesions = find_lesions(region_mask, ISO_SPACING_3D, AXIAL_AXIS, "CE")

    assert len(lesions) == 3
    assert all(lesion.measurable for lesion in lesions)
    assert len({lesion.id for lesion in lesions}) == 3  # no id collisions/merging


def test_sub_threshold_component_is_returned_not_omitted():
    # Phase 4 edge case: a component below the 10mm measurable threshold
    # must still come back as a Lesion (flagged not-measurable) so the
    # caller can show it in the table, rather than silently vanishing.
    shape = (40, 40, 10)
    region_mask = np.zeros(shape, dtype=bool)
    region_mask[10:13, 10:13, 4:6] = True  # ~3x3 voxels, well under 10mm

    lesions = find_lesions(region_mask, ISO_SPACING_3D, AXIAL_AXIS, "nonCE")

    assert len(lesions) == 1
    assert lesions[0].measurable is False
