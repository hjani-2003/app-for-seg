"""Report whether SynthSeg can run here. Run from the repo root:

    python check_synthseg.py

Prints every path the backend looks for and whether it exists, so a greyed-out
"Run SynthSeg" button can be traced to the specific missing piece.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.backends import synthseg_backend as s


def show(label, path, ok):
    print(f"{label:<16}: {path}")
    print(f"{'':<16}  {'OK' if ok else 'MISSING'}")


print("=== SynthSeg code and weights (copied by hand; models/ is gitignored) ===")
show("SYNTHSEG_HOME", s.SYNTHSEG_HOME, s.SYNTHSEG_HOME.is_dir())
show("  predict CLI", s.SYNTHSEG_SCRIPT, s.SYNTHSEG_SCRIPT.is_file())
for weight in ("synthseg_2.0.h5", "synthseg_qc_2.0.h5",
               "synthseg_robust_2.0.h5", "synthseg_parc_2.0.h5"):
    path = s.SYNTHSEG_HOME / "models" / weight
    print(f"  {weight:<26} {'OK' if path.is_file() else 'MISSING'}")

print()
print(f"=== SynthSeg interpreter (conda env '{s.SYNTHSEG_ENV_NAME}') ===")
if os.environ.get("SYNTHSEG_PYTHON"):
    print("SYNTHSEG_PYTHON set explicitly in the environment")
else:
    print("searched, in order:")
    for candidate in s.candidate_interpreters():
        print(f"  {'OK     ' if candidate.is_file() else 'missing'}  {candidate}")
show("using", s.SYNTHSEG_PYTHON, s.SYNTHSEG_PYTHON.is_file())

print()
print("=== how a run would execute ===")
print(" ", s.runtime_summary())

print()
reason = s.check_available()
print(">>> check_available():", reason or "None  —  Run SynthSeg will be enabled")
sys.exit(1 if reason else 0)
