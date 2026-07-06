MODALITY_FILE_KEYS = {
    "t1c": "T1Gd",
    "t1n": "T1",
    "t2w": "T2",
    "t2f": "FLAIR",
}

MODALITIES = ["T1", "T1Gd", "T2", "FLAIR"]
MODEL_ARCHITECTURES = ["SwinUNETR", "MambaVision", "nnU-Net"]
VIEW_MODES = ["Image", "Mask", "Image + Mask"]

# Volumes are reoriented to RAS+ canonical on load (see core/data_loader.py),
# so array axis 0/1/2 always correspond to sagittal/coronal/axial.
PLANES = ["Axial", "Coronal", "Sagittal"]
PLANE_AXES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}

# Channel order the trained models expect (matches the t1c, t1n, t2f, t2w
# order used in the BraTS-GoAT dataset JSONs the models were trained from).
MODEL_INPUT_CHANNELS = ["T1Gd", "T1", "FLAIR", "T2"]
