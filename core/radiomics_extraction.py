import re

from PySide6.QtCore import QThread, Signal

from core.backends import radiomics_backend
from core.backends.errors import ModelUnavailableError

_PROGRESS = re.compile(r"^PROGRESS (\d+)/(\d+) (\S+) (\S+)$")


class RadiomicsWorker(QThread):
    """Runs PyRadiomics off the GUI thread.

    Separate from the segmentation workers because it consumes a mask rather
    than producing one, and because a run over every region of every modality
    takes long enough — minutes, on the Extended preset — that its per-ROI
    progress is worth surfacing.
    """

    finished = Signal(object)  # RadiomicsResult
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        raw_volumes,
        segmentation,
        reference_path,
        modalities,
        preset,
        params_path,
        case_folder,
    ):
        super().__init__()
        self.raw_volumes = raw_volumes
        self.segmentation = segmentation
        self.reference_path = reference_path
        self.modalities = modalities
        self.preset = preset
        self.params_path = params_path
        # Which case this run belongs to. Features are computed against one
        # case's images and mask; if the user loads another meanwhile the
        # result must not be shown against it.
        self.case_folder = case_folder

    def run(self):
        try:
            result = radiomics_backend.extract(
                self.raw_volumes,
                self.segmentation,
                self.reference_path,
                self.modalities,
                self.preset,
                params_path=self.params_path,
                on_progress=self._report,
            )
        except ModelUnavailableError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"Radiomics failed: {exc}")
            return

        self.finished.emit(result)

    def _report(self, line):
        # Only the runner's own progress lines are useful; anything else the
        # child prints is library noise that would flicker in the status bar.
        match = _PROGRESS.match(line)
        if match:
            done, total, modality, region = match.groups()
            self.progress.emit(
                f"Radiomics: {modality} {region} ({done}/{total})"
            )
