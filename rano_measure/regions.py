import numpy as np

from core.constants import LABEL_NAMES

# Raw label name -> integer id in the segmentation volume. Inverted from
# core.constants.LABEL_NAMES, the app's canonical id -> name mapping
# (confirmed in Phase 0 / NOTES.md). Deliberately sourced from core/, not
# ui/ — rano_measure/ must stay free of Qt/GUI imports (Phase 4).
LABEL_IDS = {name: label_id for label_id, name in LABEL_NAMES.items()}

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
