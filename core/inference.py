import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PySide6.QtCore import QThread, Signal

from core.backends import mambavision_backend, nnunet_backend, swin_unetr_backend
from core.backends.errors import ModelUnavailableError
from core.constants import MODEL_INPUT_CHANNELS
from core.preprocessing import normalize_for_model

_BACKENDS = {
    "SwinUNETR": swin_unetr_backend,
    "MambaVision": mambavision_backend,
    "nnU-Net": nnunet_backend,
}


class InferenceWorker(QThread):
    finished = Signal(np.ndarray, str)
    failed = Signal(str)

    def __init__(self, volumes, model_name):
        super().__init__()
        self.volumes = volumes
        self.model_name = model_name

    def run(self):
        backend = _BACKENDS[self.model_name]

        try:
            roi, overlap, sw_batch_size = backend.get_roi_and_overlap()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self.model_name == "MambaVision":
                model, checkpoint = backend.build_model(device, roi)
            else:
                model, checkpoint = backend.build_model(device)

            volume_4ch = np.stack(
                [self.volumes[m] for m in MODEL_INPUT_CHANNELS], axis=0
            )
            input_tensor = torch.from_numpy(
                normalize_for_model(volume_4ch)
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=roi,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                )
                mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        except ModelUnavailableError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"Inference failed: {exc}")
            return

        info = (
            f"checkpoint: {checkpoint.name}"
            if checkpoint is not None
            else "no checkpoint found — randomly initialized weights (dummy inference)"
        )
        self.finished.emit(mask, info)
