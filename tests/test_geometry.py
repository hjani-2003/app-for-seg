import numpy as np
import pytest
from skimage.draw import disk, ellipse

from rano_measure.geometry import find_bidimensional_diameters


ISO_SPACING = (1.0, 1.0)


def _circle_mask(radius, shape=(80, 80)):
    mask = np.zeros(shape, dtype=bool)
    center = (shape[0] // 2, shape[1] // 2)
    rr, cc = disk(center, radius, shape=shape)
    mask[rr, cc] = True
    return mask


def _ellipse_mask(r_radius, c_radius, shape=(100, 140)):
    mask = np.zeros(shape, dtype=bool)
    center = (shape[0] // 2, shape[1] // 2)
    rr, cc = ellipse(*center, r_radius, c_radius, shape=shape)
    mask[rr, cc] = True
    return mask


@pytest.mark.parametrize("method", ["product", "greedy"])
def test_circle_major_minor_approx_equal_and_matches_known_diameter(method):
    radius = 20
    mask = _circle_mask(radius)

    result = find_bidimensional_diameters(mask, ISO_SPACING, method=method)

    expected_diameter = 2 * radius
    assert result.measurable
    assert result.major_mm == pytest.approx(expected_diameter, rel=0.15)
    assert result.minor_mm == pytest.approx(expected_diameter, rel=0.15)
    # For a circle every diameter is equal, so major/minor should be close.
    assert result.major_mm == pytest.approx(result.minor_mm, rel=0.1)


def test_ellipse_major_minor_match_known_axes():
    r_radius, c_radius = 15, 30  # short axis along rows, long axis along cols
    mask = _ellipse_mask(r_radius, c_radius)

    result = find_bidimensional_diameters(mask, ISO_SPACING, method="product")

    assert result.measurable
    assert result.major_mm == pytest.approx(2 * c_radius, rel=0.15)
    assert result.minor_mm == pytest.approx(2 * r_radius, rel=0.15)


def test_dumbbell_rejects_line_crossing_outside_mask():
    # Two 12x12 squares in opposite corners, joined only by a thin bridge
    # offset to one side. The straight line between the squares' far
    # corners cuts through the empty notch above the bridge, so a valid
    # measurement must never approach that (invalid) length.
    shape = (40, 60)
    mask = np.zeros(shape, dtype=bool)
    mask[2:14, 2:14] = True       # square A, top-left
    mask[26:38, 46:58] = True     # square B, bottom-right
    mask[17:20, 14:46] = True     # thin bridge connecting them

    far_corner_a = (2, 2)
    far_corner_b = (37, 57)
    invalid_straight_length = float(np.hypot(*(np.subtract(far_corner_b, far_corner_a))))

    result = find_bidimensional_diameters(mask, ISO_SPACING, method="product")

    assert result.measurable
    # The invalid corner-to-corner line crosses outside the mask (through
    # the notch above the bridge) — a correct implementation must not
    # report a major diameter anywhere near that invalid length.
    assert result.major_mm < invalid_straight_length - 5


def test_small_mask_is_not_measurable():
    # 5x5 px at 1mm/px spacing -> well under the 10mm measurable threshold.
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    result = find_bidimensional_diameters(mask, ISO_SPACING, method="product")

    assert not result.measurable


def test_empty_mask_returns_not_measurable():
    mask = np.zeros((20, 20), dtype=bool)

    result = find_bidimensional_diameters(mask, ISO_SPACING, method="product")

    assert not result.measurable
    assert result.major_mm == 0.0
    assert result.minor_mm == 0.0


def test_lesion_touching_image_boundary_is_measured_not_dropped():
    # A 20x20 square flush against the top-left edge of the array. Before
    # padding the mask for contour extraction, skimage.measure.find_contours
    # can't detect a 0.5-level crossing at the array edge, so this used to
    # come back as an unmeasurable degenerate result even though it's a
    # clean 20x20mm lesion.
    mask = np.zeros((60, 60), dtype=bool)
    mask[0:20, 0:20] = True

    result = find_bidimensional_diameters(mask, ISO_SPACING, method="product")

    assert result.measurable
    assert result.major_line is not None
    assert result.major_mm >= 10.0
    assert result.minor_mm >= 10.0
