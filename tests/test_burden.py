from rano_measure.burden import select_largest_per_region, select_target_lesions
from rano_measure.lesion import Lesion


def _lesion(id, region_type, product_mm2, measurable=True):
    return Lesion(
        id=id,
        region_type=region_type,
        slice_index=0,
        major_line=((0, 0), (1, 1)),
        minor_line=((0, 1), (1, 0)),
        major_mm=product_mm2 ** 0.5,
        minor_mm=product_mm2 ** 0.5,
        product_mm2=product_mm2,
        measurable=measurable,
    )


def test_ce_capped_at_three_even_with_more_measurable_ce_lesions():
    lesions = [_lesion(i, "CE", product_mm2=100 - i) for i in range(5)]

    summary = select_target_lesions(lesions)

    assert len(summary.ce_target_lesions) == 3
    assert [l.id for l in summary.ce_target_lesions] == [0, 1, 2]  # largest first


def test_total_capped_at_four_ce_then_nonce_fills_remaining_slots():
    ce_lesions = [_lesion(i, "CE", product_mm2=200 - i) for i in range(3)]
    nonce_lesions = [_lesion(10 + i, "nonCE", product_mm2=100 - i) for i in range(3)]

    summary = select_target_lesions(ce_lesions + nonce_lesions)

    assert len(summary.ce_target_lesions) == 3
    assert len(summary.nonce_target_lesions) == 1  # only 1 slot left of 4 total
    assert summary.nonce_target_lesions[0].id == 10  # the largest nonCE lesion
    assert len(summary.target_lesions) == 4


def test_non_measurable_lesions_are_excluded():
    lesions = [
        _lesion(1, "CE", product_mm2=500, measurable=False),
        _lesion(2, "CE", product_mm2=50, measurable=True),
    ]

    summary = select_target_lesions(lesions)

    assert [l.id for l in summary.ce_target_lesions] == [2]


def test_ce_and_nonce_sums_kept_separate_not_combined():
    lesions = [
        _lesion(1, "CE", product_mm2=30),
        _lesion(2, "CE", product_mm2=20),
        _lesion(3, "nonCE", product_mm2=40),
    ]

    summary = select_target_lesions(lesions)

    assert summary.ce_product_sum_mm2 == 50
    assert summary.nonce_product_sum_mm2 == 40
    # No combined-total attribute should exist on the summary at all --
    # combining CE/nonCE into one number is out of scope for this phase.
    assert not hasattr(summary, "total_product_sum_mm2")


def test_fewer_than_four_lesions_total_selects_all_of_them():
    lesions = [_lesion(1, "CE", product_mm2=90), _lesion(2, "nonCE", product_mm2=60)]

    summary = select_target_lesions(lesions)

    assert len(summary.target_lesions) == 2


def test_no_lesions_for_a_region_yields_zero_sums_not_an_error():
    # Phase 4 edge case: zero lesions found should show as an empty state
    # with 0 sums, not raise or produce an error result.
    summary = select_target_lesions([])

    assert summary.target_lesions == []
    assert summary.ce_target_lesions == []
    assert summary.nonce_target_lesions == []
    assert summary.ce_product_sum_mm2 == 0
    assert summary.nonce_product_sum_mm2 == 0


def test_largest_per_region_picks_one_of_each():
    lesions = [
        _lesion(1, "CE", product_mm2=3339.2),
        _lesion(2, "CE", product_mm2=4.0),
        _lesion(3, "nonCE", product_mm2=1970.6),
        _lesion(4, "nonCE", product_mm2=986.0),
    ]

    largest = select_largest_per_region(lesions)

    assert [l.id for l in largest] == [1, 3]


def test_largest_per_region_ignores_a_bigger_non_measurable_lesion():
    lesions = [
        _lesion(1, "CE", product_mm2=5000, measurable=False),
        _lesion(2, "CE", product_mm2=50, measurable=True),
    ]

    largest = select_largest_per_region(lesions)

    assert [l.id for l in largest] == [2]


def test_largest_per_region_keeps_regions_independent():
    # The point of not reusing select_target_lesions: its cap is on the
    # combined total, so an absent CE lesion would widen the nonCE
    # selection to two. Here nonCE still yields exactly its largest.
    lesions = [
        _lesion(1, "nonCE", product_mm2=90),
        _lesion(2, "nonCE", product_mm2=60),
    ]

    largest = select_largest_per_region(lesions)

    assert [l.id for l in largest] == [1]


def test_largest_per_region_with_nothing_measurable_is_empty_not_an_error():
    lesions = [
        _lesion(1, "CE", product_mm2=4.0, measurable=False),
        _lesion(2, "nonCE", product_mm2=1.0, measurable=False),
    ]

    assert select_largest_per_region(lesions) == []
    assert select_largest_per_region([]) == []
