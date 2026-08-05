import numpy as np

from ui.style import CLASS_LABELS

# Raw label name -> integer id in the segmentation volume. Inverted from the
# app's existing id -> name mapping (ui/style.py:CLASS_LABELS, confirmed in
# Phase 0 / NOTES.md), plus BG which has no legend entry there but is part
# of the raw label scheme every backend produces (0=BG, 1=NCR, 2=ED, 3=ET).
LABEL_IDS = {"BG": 0, **{name: label_id for label_id, name in CLASS_LABELS.items()}}

# Default region composition for RANO-style measurement.
# CE region should be the enhancing tumor only (matches RANO's CE-lesion
# definition, which already excludes necrotic/cystic core).
# non-CE region is an open modeling question: RANO's true non-CE tumor
# definition excludes vasogenic edema, which is not separable from tumor
# by segmentation label alone. Default includes both NCR and ED; expose
# an alternate ED-only mode as a config toggle rather than deciding now.
REGION_DEFS = {
    "CE": ["ET"],
    "nonCE": ["NCR", "ED"],
}

REGION_DEFS_ED_ONLY = {
    "CE": ["ET"],
    "nonCE": ["ED"],
}


def build_region_mask(label_volume, label_ids, region_defs, region_name):
    """Build a boolean region mask from a raw integer label volume.

    label_volume: integer array of any shape holding raw label ids.
    label_ids: dict mapping label name (e.g. "ET") -> integer id.
    region_defs: dict mapping region name (e.g. "CE") -> list of label
        names composing that region.
    region_name: which key of region_defs to build.
    """
    mask = np.zeros(label_volume.shape, dtype=bool)
    for label_name in region_defs[region_name]:
        mask |= label_volume == label_ids[label_name]
    return mask
