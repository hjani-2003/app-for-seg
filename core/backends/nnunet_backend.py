from core.backends.errors import ModelUnavailableError

_UNAVAILABLE_MESSAGE = (
    "nnU-Net requires a full trained results folder (plans.json + dataset.json + "
    "fold checkpoints) produced by nnUNetv2 training, not just a checkpoint file. "
    "That folder isn't configured yet, so nnU-Net can't run from this viewer."
)


def get_roi_and_overlap():
    raise ModelUnavailableError(_UNAVAILABLE_MESSAGE)


def build_model(device):
    raise ModelUnavailableError(_UNAVAILABLE_MESSAGE)
