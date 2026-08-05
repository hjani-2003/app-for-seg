MODALITY_FILE_KEYS = {
    "t1c": "T1Gd",
    "t1n": "T1",
    "t2f": "FLAIR",
    "t2w": "T2",
}


MODALITIES = ["T1", "T1Gd", "T2", "FLAIR"]

MODEL_ARCHITECTURES = ["SwinUNETR", "MaViN", "nnUnet"]

# array axis 0/1/2 correspond to sagittal/coronal/axial.
PLANES = ["Axial", "Coronal", "Sagittal"]

PLANE_AXES = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0
}

# Raw segmentation label id -> name (BraTS scheme: BG/NCR/ED/ET). Canonical
# source for both the UI legend (ui/style.py:CLASS_LABELS) and the RANO
# region mapping (rano_measure/regions.py:LABEL_IDS) — kept here, not in
# ui/, so rano_measure/ has no reason to import anything Qt-related.
LABEL_NAMES = {0: "BG", 1: "NCR", 2: "ED", 3: "ET"}


# Channel order the trained models expect (matches the t1c, t1n, t2f, t2w order used in the BraTS-GoAT dataset JSONs the models were trained from).
MODEL_INPUT_CHANNELS = ["T1Gd", "T1", "FLAIR", "T2"]
