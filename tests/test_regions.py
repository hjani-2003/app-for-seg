import numpy as np

from rano_measure.regions import (
    LABEL_IDS,
    REGION_DEFS,
    REGION_DEFS_ED_ONLY,
    build_region_mask,
)


def _synthetic_label_volume():
    # 4x4x1 volume with one voxel of each raw label placed at a known
    # location, background everywhere else.
    vol = np.zeros((4, 4, 1), dtype=np.uint8)
    vol[0, 0, 0] = LABEL_IDS["NCR"]
    vol[1, 1, 0] = LABEL_IDS["ED"]
    vol[2, 2, 0] = LABEL_IDS["ET"]
    return vol


def test_build_region_mask_ce_is_et_only():
    vol = _synthetic_label_volume()
    mask = build_region_mask(vol, LABEL_IDS, REGION_DEFS, "CE")

    assert mask.dtype == bool
    assert mask.shape == vol.shape
    assert mask[2, 2, 0]
    assert not mask[0, 0, 0]
    assert not mask[1, 1, 0]
    assert mask.sum() == 1


def test_build_region_mask_nonce_is_ncr_plus_ed():
    vol = _synthetic_label_volume()
    mask = build_region_mask(vol, LABEL_IDS, REGION_DEFS, "nonCE")

    assert mask[0, 0, 0]
    assert mask[1, 1, 0]
    assert not mask[2, 2, 0]
    assert mask.sum() == 2


def test_build_region_mask_ed_only_variant_excludes_ncr():
    vol = _synthetic_label_volume()
    mask = build_region_mask(vol, LABEL_IDS, REGION_DEFS_ED_ONLY, "nonCE")

    assert not mask[0, 0, 0]
    assert mask[1, 1, 0]
    assert mask.sum() == 1


def test_build_region_mask_background_never_matches():
    vol = _synthetic_label_volume()
    ce_mask = build_region_mask(vol, LABEL_IDS, REGION_DEFS, "CE")
    nonce_mask = build_region_mask(vol, LABEL_IDS, REGION_DEFS, "nonCE")

    background = ~(ce_mask | nonce_mask)
    # Every voxel except the 3 explicitly-placed labels should be background.
    assert background.sum() == vol.size - 3
