"""Tests for the pure-logic half of the radiomics backend.

Nothing here needs PyRadiomics installed: the job planning, the region algebra
and the CSV serialisation all run in the app's own environment, and they are
the parts that can be wrong in a way the extraction itself would never notice.
"""
import numpy as np

from core.backends.radiomics_backend import (
    RadiomicsResult, _feature_order, build_jobs, build_region_mask,
)
from core.constants import MODALITIES, RADIOMICS_REGIONS


def label_volume():
    """A 6x6x6 volume with one voxel of each tumour class, plus a second NCR."""
    volume = np.zeros((6, 6, 6), dtype=np.uint8)
    volume[1, 1, 1] = 1  # NCR
    volume[1, 1, 2] = 1  # NCR
    volume[2, 2, 2] = 2  # ED
    volume[3, 3, 3] = 3  # ET
    return volume


def test_single_label_region_selects_only_that_label():
    mask = build_region_mask(label_volume(), RADIOMICS_REGIONS["ED"])
    assert mask.sum() == 1
    assert mask[2, 2, 2] == 1
    assert mask.dtype == np.uint8


def test_tumour_core_is_necrosis_and_enhancement_but_not_edema():
    mask = build_region_mask(label_volume(), RADIOMICS_REGIONS["TC"])
    assert mask.sum() == 3
    assert mask[2, 2, 2] == 0


def test_whole_tumour_is_every_labelled_voxel():
    volume = label_volume()
    mask = build_region_mask(volume, RADIOMICS_REGIONS["WT"])
    assert mask.sum() == int((volume > 0).sum()) == 4


def test_region_mask_is_binary_not_a_copy_of_the_label_values():
    mask = build_region_mask(label_volume(), RADIOMICS_REGIONS["ET"])
    assert set(np.unique(mask)) == {0, 1}


def test_every_region_is_paired_with_every_requested_modality():
    jobs, skipped = build_jobs(label_volume(), ["T1", "FLAIR"])
    assert len(jobs) == len(RADIOMICS_REGIONS) * 2
    assert skipped == []
    assert ("T1", "WT") in jobs and ("FLAIR", "WT") in jobs


def test_region_absent_from_the_mask_is_skipped_rather_than_extracted():
    volume = label_volume()
    volume[volume == 3] = 0  # a case with no enhancing component

    jobs, skipped = build_jobs(volume, ["T1"])

    assert ("T1", "ET") not in jobs
    assert ("T1", "ET", "region is empty in this mask") in skipped
    # The composites that do not need ET are unaffected.
    assert ("T1", "WT") in jobs and ("T1", "TC") in jobs


def test_a_skipped_region_is_reported_once_per_modality():
    volume = label_volume()
    volume[volume == 3] = 0

    _, skipped = build_jobs(volume, MODALITIES)

    assert len(skipped) == len(MODALITIES)
    assert {entry[0] for entry in skipped} == set(MODALITIES)


def test_an_entirely_empty_mask_plans_no_jobs_at_all():
    jobs, skipped = build_jobs(np.zeros((4, 4, 4), dtype=np.uint8), ["T1"])
    assert jobs == []
    assert len(skipped) == len(RADIOMICS_REGIONS)


def test_feature_order_follows_the_first_row_not_the_alphabet():
    rows = [
        {"features": {"original_shape_Volume": 1, "original_firstorder_Mean": 2}},
        {"features": {"original_firstorder_Mean": 3}},
    ]
    assert _feature_order(rows) == [
        "original_shape_Volume", "original_firstorder_Mean",
    ]


def test_feature_order_picks_up_names_a_later_row_added():
    rows = [
        {"features": {"a": 1}},
        {"features": {"a": 2, "b": 3}},
    ]
    assert _feature_order(rows) == ["a", "b"]


def result_with(rows, feature_names):
    return RadiomicsResult(
        rows=rows,
        feature_names=feature_names,
        preset="Standard",
        modalities=["T1"],
    )


def test_csv_has_one_row_per_extraction_behind_a_header():
    result = result_with(
        [
            {"Modality": "T1", "Region": "WT", "a": 1.5, "b": 2},
            {"Modality": "T1", "Region": "ET", "a": 3.5, "b": 4},
        ],
        ["a", "b"],
    )

    lines = result.to_csv().strip().split("\n")

    assert lines[0] == "Modality,Region,a,b"
    assert lines[1] == "T1,WT,1.5,2"
    assert len(lines) == 3


def test_csv_columns_follow_feature_names_not_dict_order():
    result = result_with([{"Modality": "T1", "Region": "WT", "b": 2, "a": 1}], ["a", "b"])
    assert result.to_csv().strip().split("\n")[1] == "T1,WT,1,2"


def test_csv_leaves_a_feature_missing_from_one_row_blank():
    result = result_with(
        [
            {"Modality": "T1", "Region": "WT", "a": 1, "b": 2},
            {"Modality": "T1", "Region": "ET", "a": 3},
        ],
        ["a", "b"],
    )
    assert result.to_csv().strip().split("\n")[2] == "T1,ET,3,"


def test_csv_quotes_a_value_containing_a_comma():
    result = result_with(
        [{"Modality": "T1", "Region": "WT", "a": "1, 2, 3"}], ["a"]
    )
    assert '"1, 2, 3"' in result.to_csv()
