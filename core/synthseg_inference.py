import numpy as np
from PySide6.QtCore import QThread, Signal

from core.backends import synthseg_backend
from core.backends.errors import ModelUnavailableError


class SynthSegWorker(QThread):
    """Runs SynthSeg off the GUI thread.

    Kept separate from InferenceWorker: SynthSeg takes a single file rather
    than a 4-channel stack, needs no torch device, and runs long enough on CPU
    that its progress output is worth surfacing.
    """

    finished = Signal(np.ndarray, object)  # mask, SynthSegResult
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, image_path, reference_shape, modality, options):
        super().__init__()
        self.image_path = image_path
        self.reference_shape = reference_shape
        self.modality = modality
        self.options = options

    def run(self):
        try:
            result = synthseg_backend.predict(
                self.image_path,
                self.reference_shape,
                modality=self.modality,
                on_progress=self._report,
                **self.options,
            )
        except ModelUnavailableError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"SynthSeg failed: {exc}")
            return

        self.finished.emit(result.mask, result)

    def _report(self, line):
        # SynthSeg's own progress lines are the only useful ones; the rest is
        # TensorFlow start-up noise that would just flicker in the status bar.
        if "predicting" in line.lower():
            self.progress.emit(f"SynthSeg: {line.strip()}")
