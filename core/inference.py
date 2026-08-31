import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PySide6.QtCore import QThread, Signal

from core.backends import mavin_backend, nnunet_backend, swin_unetr_backend
from core.backends.errors import ModelUnavailableError
from core.constants import MODEL_INPUT_CHANNELS
from core.preprocessing import normalize_for_model

_BACKENDS = {
    "SwinUNETR": swin_unetr_backend,
    "MaViN": mavin_backend,
}


class InferenceWorker(QThread):
    finished = Signal(np.ndarray, str)
    failed = Signal(str)

    def __init__(self, volumes, paths, model_name):
        super().__init__()
        self.volumes = volumes
        self.paths = paths
        self.model_name = model_name

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self.model_name == "nnUnet":
                # nnUNet does its own file-based preprocessing from plans.json,
                # so it reads the original files directly instead of going
                # through MONAI's sliding_window_inference like the other backends.
                mask, checkpoint = nnunet_backend.predict(self.paths, device)
            else:
                backend = _BACKENDS[self.model_name]
                roi, overlap, sw_batch_size = backend.get_roi_and_overlap()

                if self.model_name == "MaViN":
                    model, checkpoint = backend.build_model(device, roi)
                else:
                    model, checkpoint = backend.build_model(device)

                volume_4ch = np.stack(
                    [self.volumes[m] for m in MODEL_INPUT_CHANNELS], axis=0
                )
                input_tensor = torch.from_numpy(
                    normalize_for_model(volume_4ch)
                ).unsqueeze(0).to(device)

                # inference_mode rather than no_grad: it also skips view and
                # version-counter tracking, and nothing downstream of here
                # needs autograd metadata — the mask leaves as a numpy array.
                with torch.inference_mode():
                    logits = sliding_window_inference(
                        inputs=input_tensor,
                        roi_size=roi,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=overlap,
                        # The windows run on the GPU, but the stitched output —
                        # out_channels x the whole volume in float32, the single
                        # largest allocation of a run — is assembled on the CPU.
                        # Nothing downstream needs it on the device.
                        sw_device=device,
                        device="cpu",
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
