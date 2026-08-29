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

# Which overlays the right-hand panel draws. Tumour and SynthSeg masks are
# independent layers, so they can be shown alone or composited together.
OVERLAY_TUMOR = "Tumor"
OVERLAY_SYNTHSEG = "SynthSeg"
OVERLAY_BOTH = "Both"
OVERLAY_MODES = [OVERLAY_TUMOR, OVERLAY_SYNTHSEG, OVERLAY_BOTH]

# SynthSeg is contrast-agnostic but is trained and validated on T1, so that is
# the default even though any loaded modality can be selected.
SYNTHSEG_DEFAULT_MODALITY = "T1"

# Modality -> the BraTS filename key, used to name saved SynthSeg outputs so a
# T1 run and a T1Gd run of the same case do not overwrite each other.
MODALITY_TO_FILE_KEY = {v: k for k, v in MODALITY_FILE_KEYS.items()}

# ROIs radiomic features are extracted over. The three model classes on their
# own, plus the two composite regions the BraTS literature reports against:
# tumour core (necrosis + enhancing) and whole tumour (everything). Composites
# matter because a feature computed over the whole lesion is not recoverable
# from features of its parts.
RADIOMICS_REGIONS = {
    "NCR": (1,),
    "ED": (2,),
    "ET": (3,),
    "TC": (1, 3),
    "WT": (1, 2, 3),
}

RADIOMICS_REGION_DESCRIPTIONS = {
    "NCR": "necrotic core",
    "ED": "peritumoral edema",
    "ET": "enhancing tumour",
    "TC": "tumour core (NCR + ET)",
    "WT": "whole tumour (NCR + ED + ET)",
}

# Extraction presets, in increasing cost. The names index the YAML files in
# core/radiomics_params/.
RADIOMICS_PRESETS = ["Fast", "Standard", "Extended"]
RADIOMICS_CUSTOM_PRESET = "Custom params.yaml…"
