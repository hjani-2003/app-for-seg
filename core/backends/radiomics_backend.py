"""PyRadiomics feature extraction, run as a subprocess.

PyRadiomics' last release ships compiled wheels for CPython 3.7-3.9 only, and
its C extensions are built against the numpy 1.x ABI. This app runs Python 3.11
with numpy 2.x, so installing it here means a source build that does not
survive the numpy major version. In a Python 3.9 env it is a prebuilt wheel and
nothing has to be compiled at all — so, like SynthSeg, it is not imported but
invoked through its own interpreter, communicating over files.

Unlike SynthSeg there is no CLI worth driving (see radiomics_runner.py), so the
child runs a runner script that ships with this repo.
"""
import json
import os
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.backends.errors import ModelUnavailableError
from core.constants import RADIOMICS_REGIONS
from core.data_loader import save_segmentation, save_volume

# `or` rather than a get() default throughout: an env var set but left empty
# would otherwise resolve to the current directory.
RADIOMICS_ENV_NAME = os.environ.get("RADIOMICS_ENV_NAME") or "radiomics_39"
RUNNER = Path(__file__).resolve().parent / "radiomics_runner.py"
PARAMS_DIR = Path(__file__).resolve().parents[1] / "radiomics_params"

_LOG_TAIL = 25


def _candidate_interpreters():
    """Where the radiomics conda env might live, best guess first.

    Same ladder as core/backends/synthseg_backend.py, and for the same reason:
    hardcoding one absolute path breaks the moment the repo moves to another
    machine. The sibling-of-the-active-env guess is the reliable one.
    """
    relative = Path("envs") / RADIOMICS_ENV_NAME / "bin" / "python"

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        yield Path(prefix).parent / RADIOMICS_ENV_NAME / "bin" / "python"

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        yield Path(conda_exe).parent.parent / relative

    for root in (
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "miniforge3",
        Path.home() / "mambaforge",
        Path("/opt/conda"),
    ):
        yield root / relative


def candidate_interpreters():
    """_candidate_interpreters with duplicates removed, order preserved."""
    seen = set()
    for candidate in _candidate_interpreters():
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _discover_interpreter():
    explicit = os.environ.get("RADIOMICS_PYTHON")
    if explicit:
        return Path(explicit)
    for candidate in candidate_interpreters():
        if candidate.is_file():
            return candidate
    # Nothing found: return the most likely location so the error message names
    # a plausible path rather than an empty one.
    return next(iter(candidate_interpreters()), Path(RADIOMICS_ENV_NAME))


RADIOMICS_PYTHON = _discover_interpreter()


def _package_dir(interpreter):
    """The installed `radiomics` package inside the given interpreter's env.

    Found by looking rather than by importing: check_available is called on
    every case load, and spawning the child just to answer it would make
    loading a folder feel slow.
    """
    env_root = Path(interpreter).resolve().parent.parent
    for site_packages in sorted(env_root.glob("lib/python3.*/site-packages")):
        package = site_packages / "radiomics"
        if package.is_dir():
            return package
    return None


def check_available():
    """Return None if PyRadiomics can run, else a human-readable reason why not."""
    if not RADIOMICS_PYTHON.is_file():
        return (
            f"PyRadiomics interpreter not found at {RADIOMICS_PYTHON}. Create "
            f"the {RADIOMICS_ENV_NAME} env (see README) or set RADIOMICS_PYTHON."
        )
    if not RUNNER.is_file():
        return f"Radiomics runner script missing from {RUNNER}."
    if _package_dir(RADIOMICS_PYTHON) is None:
        return (
            f"pyradiomics is not installed in {RADIOMICS_PYTHON}. Run "
            f'`conda run -n {RADIOMICS_ENV_NAME} pip install "numpy<2" '
            f"pyradiomics` (see README)."
        )
    return None


def installed_version():
    """PyRadiomics' version, read from the child env without starting it."""
    package = _package_dir(RADIOMICS_PYTHON)
    if package is None:
        return None
    for dist in package.parent.glob("pyradiomics-*.dist-info"):
        return dist.name[len("pyradiomics-"):-len(".dist-info")]
    return "installed"


def runtime_summary():
    """One line describing how a run will execute, for the status bar."""
    version = installed_version()
    if version is None:
        return f"PyRadiomics: not installed in {RADIOMICS_ENV_NAME}"
    return f"PyRadiomics {version} · {RADIOMICS_ENV_NAME}"


def preset_params_path(preset):
    """The bundled YAML for a preset name, or the RADIOMICS_PARAMS override."""
    override = os.environ.get("RADIOMICS_PARAMS")
    if override:
        return Path(override)
    return PARAMS_DIR / f"{preset.lower()}.yaml"


def build_region_mask(segmentation, labels):
    """Binary mask of every voxel in `segmentation` carrying one of `labels`.

    uint8 with a single foreground value, because a PyRadiomics run extracts
    one label at a time and the app always asks for label 1.
    """
    return np.isin(segmentation, labels).astype(np.uint8)


def build_jobs(segmentation, modalities, regions=None):
    """Which (modality, region) pairs are worth extracting, and which are not.

    Returns (jobs, skipped). A region with no voxels is dropped here rather
    than in the child: PyRadiomics raises on an empty ROI, and "the enhancing
    component of this case is empty" is a fact about the case, not a failure.
    """
    if regions is None:
        regions = RADIOMICS_REGIONS

    jobs = []
    skipped = []
    for region, labels in regions.items():
        voxels = int(np.isin(segmentation, labels).sum())
        if voxels == 0:
            for modality in modalities:
                skipped.append((modality, region, "region is empty in this mask"))
            continue
        for modality in modalities:
            jobs.append((modality, region))
    return jobs, skipped


@dataclass
class RadiomicsResult:
    """One extraction run, held in memory so saving never re-runs it."""

    rows: list                       # [{"Modality":…, "Region":…, feature: value}]
    feature_names: list              # stable column order
    preset: str
    modalities: list
    skipped: list = field(default_factory=list)   # (modality, region, reason)
    info: str = ""

    def to_csv(self):
        """The feature table as CSV text, long-form: one row per (modality, region).

        Written with the stdlib csv module rather than pandas — pandas is not a
        declared dependency of this app, and this matches how SynthSeg's CSVs
        are carried as text and written out verbatim.
        """
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["Modality", "Region"] + self.feature_names)
        for row in self.rows:
            writer.writerow(
                [row["Modality"], row["Region"]]
                + [row.get(name, "") for name in self.feature_names]
            )
        return buffer.getvalue()


def _run(cmd, on_progress):
    """Run the child, streaming stdout, and return its last few lines."""
    tail = deque(maxlen=_LOG_TAIL)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        if on_progress is not None:
            on_progress(line)
    process.wait()
    return "\n".join(tail)


def _feature_order(rows):
    """Feature names in the order PyRadiomics emitted them, first row wins.

    Not sorted: PyRadiomics groups a row by class (shape, then firstorder, then
    each texture class) and that grouping is more useful to read than
    alphabetical. Later rows can only contribute names the first one lacked,
    which happens when a region was skipped for one modality but not another.
    """
    names = []
    seen = set()
    for row in rows:
        for name in row["features"]:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def extract(
    raw_volumes,
    segmentation,
    reference_path,
    modalities,
    preset,
    params_path=None,
    on_progress=None,
):
    """Extract radiomic features over the tumour regions and return a RadiomicsResult.

    raw_volumes are the un-normalized modality arrays: the display copies are
    min-max scaled per volume (core/data_loader.normalize_for_display), which
    would make every first-order feature a property of the scaling rather than
    of the tissue.
    """
    reason = check_available()
    if reason is not None:
        raise ModelUnavailableError(reason)

    if params_path is None:
        params_path = preset_params_path(preset)
    params_path = Path(params_path)
    if not params_path.is_file():
        raise RuntimeError(f"Extraction parameters not found at {params_path}")

    pairs, skipped = build_jobs(segmentation, modalities)
    if not pairs:
        raise RuntimeError(
            "The segmentation is empty — there is nothing to extract features from."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Image and mask are both written from reference_path's canonical
        # affine, so PyRadiomics sees two volumes on identical geometry. Each
        # is written once and shared by every job that needs it.
        image_paths = {}
        for modality in modalities:
            image_paths[modality] = tmp / f"image_{modality}.nii.gz"
            save_volume(raw_volumes[modality], reference_path, image_paths[modality])

        mask_paths = {}
        for _, region in pairs:
            if region in mask_paths:
                continue
            mask_paths[region] = tmp / f"mask_{region}.nii.gz"
            save_segmentation(
                build_region_mask(segmentation, RADIOMICS_REGIONS[region]),
                reference_path,
                mask_paths[region],
            )

        spec_path = tmp / "spec.json"
        output_path = tmp / "results.json"
        spec_path.write_text(
            json.dumps(
                {
                    "params": str(params_path),
                    "output": str(output_path),
                    "jobs": [
                        {
                            "image": str(image_paths[modality]),
                            "mask": str(mask_paths[region]),
                            "modality": modality,
                            "region": region,
                        }
                        for modality, region in pairs
                    ],
                }
            )
        )

        log_tail = _run(
            [str(RADIOMICS_PYTHON), str(RUNNER), str(spec_path)], on_progress
        )

        # The runner writes its output last, so its presence — not the return
        # code — is what says the run got all the way through.
        if not output_path.is_file():
            raise RuntimeError(f"PyRadiomics produced no features:\n{log_tail}")
        payload = json.loads(output_path.read_text())

    feature_names = _feature_order(payload["rows"])
    rows = [
        dict({"Modality": row["modality"], "Region": row["region"]}, **row["features"])
        for row in payload["rows"]
    ]
    skipped = skipped + [tuple(entry) for entry in payload["skipped"]]

    info = (
        f"PyRadiomics {payload['version']}, {preset} preset · "
        f"{len(rows)} region/modality rows × {len(feature_names)} features"
    )
    if skipped:
        info += f" · {len(skipped)} skipped"

    return RadiomicsResult(
        rows=rows,
        feature_names=feature_names,
        preset=preset,
        modalities=list(modalities),
        skipped=skipped,
        info=info,
    )
