"""Report whether PyRadiomics can run here. Run from the repo root:

    python check_radiomics.py

Prints every interpreter the backend probes and what it found inside the one it
picked, so a greyed-out "Extract Features" button can be traced to the specific
missing piece.

The numpy version in the child env is printed because it is the failure this
setup invites: PyRadiomics' wheels are built against numpy 1.x, numpy 2 also
supports Python 3.9, and an unpinned `pip install pyradiomics` therefore
produces an env that looks correct and fails at import.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.backends import radiomics_backend as r
except Exception as exc:
    print(f"Could not import the radiomics backend: {exc}")
    print("Run this from the repo root, with the app's conda env active.")
    sys.exit(2)


def mark(ok):
    return "OK" if ok else "MISSING"


print(f"backend module  : {r.__file__}")
print(f"env name        : {r.RADIOMICS_ENV_NAME}")
print(f"RADIOMICS_PYTHON: {os.environ.get('RADIOMICS_PYTHON') or '(unset)'}")
print(f"RADIOMICS_PARAMS: {os.environ.get('RADIOMICS_PARAMS') or '(unset)'}")
print()

print("interpreters probed, best guess first:")
for candidate in r.candidate_interpreters():
    print(f"  [{mark(candidate.is_file()):7}] {candidate}")
print(f"\nchosen          : {r.RADIOMICS_PYTHON}")
print(f"runner script   : [{mark(r.RUNNER.is_file())}] {r.RUNNER}")

package = r._package_dir(r.RADIOMICS_PYTHON) if r.RADIOMICS_PYTHON.is_file() else None
print(f"radiomics package: [{mark(package is not None)}] {package or '-'}")
print(f"version         : {r.installed_version() or '-'}")

print("\nbundled parameter files:")
for preset in ("Fast", "Standard", "Extended"):
    path = r.PARAMS_DIR / f"{preset.lower()}.yaml"
    print(f"  [{mark(path.is_file()):7}] {preset:9} {path}")

# The import is the only check that catches a numpy ABI mismatch, and it costs
# a second, so it is done last and only when there is something to import.
if package is not None:
    print("\nchild env import:")
    probe = (
        "import numpy, radiomics; "
        "print('  radiomics', radiomics.__version__, '/ numpy', numpy.__version__)"
    )
    try:
        out = subprocess.run(
            [str(r.RADIOMICS_PYTHON), "-c", probe],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        print(f"  could not run the child interpreter: {exc}")
    else:
        print(out.stdout.rstrip() or "  (no output)")
        if out.returncode != 0:
            print(out.stderr.strip()[-1500:])
            print(
                '\n  Import failed. If it mentions numpy dtype sizes or a binary\n'
                '  incompatibility, the env has numpy 2: recreate it with\n'
                f'  `conda run -n {r.RADIOMICS_ENV_NAME} pip install "numpy<2" '
                "pyradiomics`."
            )

print(f"\nverdict         : {r.check_available() or 'PyRadiomics can run'}")
print(f"summary line    : {r.runtime_summary()}")
