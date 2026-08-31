"""RANO 2.0 "mixed tumor" target lesion selection.

Takes a flat list of rano_measure.lesion.Lesion objects (already measured,
already tagged CE/nonCE) and selects target lesions per RANO 2.0's mixed-
tumor rule: up to MAX_CE_TARGET_LESIONS CE lesions, up to
MAX_TOTAL_TARGET_LESIONS combined. Sums bidimensional products separately
for CE and nonCE -- does NOT combine them into a single burden number, since
that decision belongs to response classification (out of scope for this
phase).

Also offers select_largest_per_region, the narrower one-per-region selection
the measurement table reports.
"""
from dataclasses import dataclass

# RANO 2.0 mixed-tumor target lesion caps -- config constants, not magic
# numbers buried in the selection logic below.
MAX_CE_TARGET_LESIONS = 3
MAX_TOTAL_TARGET_LESIONS = 4


@dataclass
class BurdenSummary:
    target_lesions: list  # selected Lesion objects, CE targets then nonCE targets
    ce_target_lesions: list
    nonce_target_lesions: list
    ce_product_sum_mm2: float
    nonce_product_sum_mm2: float


def select_target_lesions(lesions, *, max_ce=MAX_CE_TARGET_LESIONS, max_total=MAX_TOTAL_TARGET_LESIONS):
    """Rank measurable lesions by product_mm2 descending and select RANO
    2.0 mixed-tumor target lesions.

    Only lesions with `measurable=True` are eligible. CE lesions are
    ranked and capped at `max_ce` first; the remaining target-lesion
    budget (up to `max_total` total) is then filled with the best-ranked
    nonCE lesions.
    """
    measurable = [lesion for lesion in lesions if lesion.measurable]

    ce_sorted = sorted(
        (lesion for lesion in measurable if lesion.region_type == "CE"),
        key=lambda lesion: lesion.product_mm2,
        reverse=True,
    )
    nonce_sorted = sorted(
        (lesion for lesion in measurable if lesion.region_type == "nonCE"),
        key=lambda lesion: lesion.product_mm2,
        reverse=True,
    )

    ce_targets = ce_sorted[:max_ce]
    remaining_slots = max(max_total - len(ce_targets), 0)
    nonce_targets = nonce_sorted[:remaining_slots]

    return BurdenSummary(
        target_lesions=ce_targets + nonce_targets,
        ce_target_lesions=ce_targets,
        nonce_target_lesions=nonce_targets,
        ce_product_sum_mm2=sum(lesion.product_mm2 for lesion in ce_targets),
        nonce_product_sum_mm2=sum(lesion.product_mm2 for lesion in nonce_targets),
    )


def select_largest_per_region(lesions, *, region_types=("CE", "nonCE")):
    """The single largest measurable lesion of each region type.

    Each region is ranked on its own, so one with nothing measurable simply
    contributes nothing and never hands its place to the other -- which is
    where select_target_lesions differs: its cap is on the combined total, so
    an absent CE lesion there widens the nonCE selection instead.

    Returned in `region_types` order, so CE precedes nonCE.
    """
    largest = []
    for region_type in region_types:
        candidates = [
            lesion for lesion in lesions
            if lesion.measurable and lesion.region_type == region_type
        ]
        if candidates:
            largest.append(max(candidates, key=lambda lesion: lesion.product_mm2))
    return largest
