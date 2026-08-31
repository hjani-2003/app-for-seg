"""Pure-geometry RANO-style bidimensional measurement on a single 2D slice.

No UI, no napari, no I/O — everything here takes a plain boolean 2D mask
and physical pixel spacing in, and returns lengths in mm. Callers (Phase 3,
``rano_measure/lesion.py``) own picking which slice and which spacing axes
to pass in.
"""
from dataclasses import dataclass

import numpy as np
from skimage import measure

DEFAULT_ANGLE_TOL_DEG = 5.0
DEFAULT_MIN_DIAMETER_MM = 10.0
DEFAULT_MAX_CONTOUR_POINTS = 80


@dataclass
class SliceMeasurement:
    major_line: tuple  # ((r1, c1), (r2, c2)) in voxel coords, or None
    minor_line: tuple
    major_mm: float
    minor_mm: float
    product_mm2: float
    measurable: bool


def _empty_result():
    return SliceMeasurement(None, None, 0.0, 0.0, 0.0, False)


def _polygon_area(contour):
    r = contour[:, 0]
    c = contour[:, 1]
    return 0.5 * abs(np.dot(r, np.roll(c, 1)) - np.dot(c, np.roll(r, 1)))


def _contour_points(mask, max_points):
    # skimage.measure.find_contours can't detect a 0.5-level crossing at
    # the array edge (there's no "outside" pixel to compare against), so a
    # lesion touching the image boundary would otherwise come back with an
    # incomplete/open contour instead of its true closed boundary. Pad with
    # a 1-pixel border of False on every side, then shift the returned
    # coordinates back by that same padding.
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    contours = measure.find_contours(padded.astype(np.float32), level=0.5)
    if not contours:
        return None
    contours = [contour - 1.0 for contour in contours]

    # A slice mask can contain more than one closed loop (e.g. a pinched
    # component, or stray noise) — take the one enclosing the largest area
    # as the lesion boundary.
    best = max(contours, key=_polygon_area)
    if len(best) > max_points:
        idx = np.linspace(0, len(best) - 1, max_points).astype(int)
        best = best[idx]
    return best


def _segment_valid(mask, p1, p2, samples_per_px=2):
    """True iff every sampled point on the p1->p2 line is foreground.

    This is the "full line segment must lie inside the mask" check —
    rasterizes the candidate line at >=2 samples per pixel of length and
    checks each sample, not just the two endpoints, so a straight line
    that cuts across a concave notch (e.g. a dumbbell's waist) is rejected.
    """
    r0, c0 = p1
    r1, c1 = p2
    length_px = float(np.hypot(r1 - r0, c1 - c0))
    n = max(int(np.ceil(length_px * samples_per_px)), 2)
    rs = np.linspace(r0, r1, n)
    cs = np.linspace(c0, c1, n)
    ri = np.clip(np.round(rs).astype(int), 0, mask.shape[0] - 1)
    ci = np.clip(np.round(cs).astype(int), 0, mask.shape[1] - 1)
    return bool(np.all(mask[ri, ci]))


def _segment_length_mm(p1, p2, row_mm, col_mm):
    dr = (p2[0] - p1[0]) * row_mm
    dc = (p2[1] - p1[1]) * col_mm
    return float(np.hypot(dr, dc))


def _segment_angle_rad(p1, p2, row_mm, col_mm):
    # Angle measured in physical (mm) space, not raw pixel space, so
    # perpendicularity is correct even under anisotropic voxel spacing.
    # Folded into [0, pi) since a line has no direction.
    dr = (p2[0] - p1[0]) * row_mm
    dc = (p2[1] - p1[1]) * col_mm
    angle = np.arctan2(dr, dc) % np.pi
    return angle


@dataclass
class _Segment:
    p1: tuple
    p2: tuple
    length_mm: float
    angle_rad: float


def _all_valid_segments(mask, points, row_mm, col_mm):
    segments = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if not _segment_valid(mask, p1, p2):
                continue
            length_mm = _segment_length_mm(p1, p2, row_mm, col_mm)
            angle = _segment_angle_rad(p1, p2, row_mm, col_mm)
            segments.append(_Segment(tuple(p1), tuple(p2), length_mm, angle))
    return segments


def _acute_angle_between(a, b):
    diff = abs(a - b) % np.pi
    return min(diff, np.pi - diff)


def _is_perpendicular(a, b, angle_tol_rad):
    return _acute_angle_between(a, b) >= (np.pi / 2 - angle_tol_rad)


def _best_perpendicular(major, segments, angle_tol_rad):
    candidates = [s for s in segments if _is_perpendicular(major.angle_rad, s.angle_rad, angle_tol_rad)]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.length_mm)


def _best_product_pair(segments, angle_tol_rad):
    """Jointly optimize major_mm * minor_mm over all valid perpendicular
    pairs, rather than greedily fixing the longest segment as major first.

    Meaningfully more robust to small contour noise from the segmentation
    model than the greedy approach: the single globally-longest inscribed
    segment can be a noisy outlier with a poor (short) perpendicular
    partner, whereas a slightly shorter major with a much longer
    perpendicular partner yields a larger, more representative product.
    """
    if not segments:
        return None, None

    lengths = np.array([s.length_mm for s in segments])
    angles = np.array([s.angle_rad for s in segments])

    best_product = -1.0
    best_i = best_j = -1
    for i in range(len(segments)):
        diff = np.abs(angles - angles[i]) % np.pi
        acute = np.minimum(diff, np.pi - diff)
        perp_mask = acute >= (np.pi / 2 - angle_tol_rad)
        if not np.any(perp_mask):
            continue
        j = int(np.argmax(np.where(perp_mask, lengths, -1.0)))
        product = lengths[i] * lengths[j]
        if product > best_product:
            best_product = product
            best_i, best_j = i, j

    if best_i < 0:
        return None, None
    return segments[best_i], segments[best_j]


def find_bidimensional_diameters(
    mask,
    spacing,
    *,
    angle_tol_deg=DEFAULT_ANGLE_TOL_DEG,
    min_diameter_mm=DEFAULT_MIN_DIAMETER_MM,
    max_contour_points=DEFAULT_MAX_CONTOUR_POINTS,
    method="product",
):
    """Measure RANO-style major/minor bidimensional diameters on one 2D
    boolean slice mask.

    mask: 2D boolean array.
    spacing: (row_mm, col_mm) — physical spacing of THIS slice's two
        in-plane axes. Caller picks which two of the volume's three
        spacing values match the mask's row/col axes for the plane in use.
    method: "product" (default, recommended) jointly optimizes the
        major*minor product across all valid perpendicular pairs;
        "greedy" fixes the single longest valid segment as major first,
        then searches for the best perpendicular minor.

    All lengths are computed in mm from `spacing`, never raw pixel
    distance. `measurable` is True only if both diameters are
    >= min_diameter_mm.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return _empty_result()

    points = _contour_points(mask, max_points=max_contour_points)
    if points is None or len(points) < 2:
        return _empty_result()

    row_mm, col_mm = spacing
    segments = _all_valid_segments(mask, points, row_mm, col_mm)
    if not segments:
        return _empty_result()

    angle_tol_rad = np.radians(angle_tol_deg)

    if method == "product":
        a, b = _best_product_pair(segments, angle_tol_rad)
        # The product search is symmetric in the pair, so it can hand back
        # the shorter segment first — relabel by length so "major" is
        # always the longer of the two, matching the "greedy" branch.
        if a is not None and b is not None and b.length_mm > a.length_mm:
            a, b = b, a
        major, minor = a, b
    elif method == "greedy":
        major = max(segments, key=lambda s: s.length_mm)
        minor = _best_perpendicular(major, segments, angle_tol_rad)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if major is None or minor is None:
        return _empty_result()

    measurable = major.length_mm >= min_diameter_mm and minor.length_mm >= min_diameter_mm

    return SliceMeasurement(
        major_line=(major.p1, major.p2),
        minor_line=(minor.p1, minor.p2),
        major_mm=major.length_mm,
        minor_mm=minor.length_mm,
        product_mm2=major.length_mm * minor.length_mm,
        measurable=measurable,
    )
