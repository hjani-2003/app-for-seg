import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from core.backends.errors import ModelUnavailableError
from core.constants import MODEL_INPUT_CHANNELS

MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "nnunet" / "results"
    / "Dataset001_BraTS" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
USE_FOLDS = (0,)
CHECKPOINT_NAME = "checkpoint_best.pth"
STEP_SIZE = 0.3  # matches the BraTS-GoAT submission container's entrypoint.sh


def predict(image_paths, device):
    """Run nnUNetv2 inference on the raw modality files for one case.

    Unlike the other backends, nnUNet reads the original (un-reoriented)
    NIfTI files itself and does its own resampling/normalization from
    plans.json, so it's given file paths rather than preloaded arrays.
    """
    if not MODEL_DIR.is_dir():
        raise ModelUnavailableError(
            f"nnUNet results folder not found at {MODEL_DIR}. Extract "
            "dataset.json, plans.json and fold_0/checkpoint_best.pth from "
            "the trained results into that path."
        )

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=STEP_SIZE,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR), use_folds=USE_FOLDS, checkpoint_name=CHECKPOINT_NAME,
    )

    case_files = [[image_paths[m] for m in MODEL_INPUT_CHANNELS]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_truncated = str(Path(tmp_dir) / "case")
        predictor.predict_from_files_sequential(
            case_files, [output_truncated], save_probabilities=False, overwrite=True,
        )

        # Reorient to RAS+ canonical to match the app's other volumes (see core/data_loader.py).
        seg_img = nib.as_closest_canonical(nib.load(output_truncated + ".nii.gz"))
        mask = seg_img.get_fdata().astype(np.uint8)

    checkpoint = MODEL_DIR / f"fold_{USE_FOLDS[0]}" / CHECKPOINT_NAME
    return mask, checkpoint
