"""Report whether SynthSeg can run here. Run from the repo root:

    python check_synthseg.py

Prints every path the backend looks for and whether it exists, so a greyed-out
"Run SynthSeg" button can be traced to the specific missing piece.

Deliberately tolerant of an older core/backends/synthseg_backend.py: this is
the tool you reach for when something is already wrong, so a partially-synced
checkout must produce a diagnosis, not a traceback.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.backends import synthseg_backend as s
except Exception as exc:
    print(f"Could not import the SynthSeg backend: {exc}")
    print("Run this from the repo root, with the app's conda env active.")
    sys.exit(2)


def head():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def mark(ok):
    return "OK" if ok else "MISSING"


# Attributes added after the first version of this script. If the backend
# predates them the checkout is half-updated, which is itself the diagnosis.
NEWER_API = ("SYNTHSEG_ENV_NAME", "candidate_interpreters", "runtime_summary")
stale = [name for name in NEWER_API if not hasattr(s, name)]

print(f"repo commit     : {head()}")
print(f"backend module  : {s.__file__}")
if stale:
    print()
    print("!! core/backends/synthseg_backend.py is OLDER than this script.")
    print(f"!! It is missing: {', '.join(stale)}")
    print("!! The checkout is only partly updated — sync the whole repo")
    print("!! (git pull, or re-copy the tracked files), then re-run this.")
    print("!! Continuing with the checks that do work.")
print()

print("=== SynthSeg code and weights (copied by hand; models/ is gitignored) ===")
print(f"SYNTHSEG_HOME   : {s.SYNTHSEG_HOME}\n                  {mark(s.SYNTHSEG_HOME.is_dir())}")
print(f"  predict CLI   : {s.SYNTHSEG_SCRIPT}\n                  {mark(s.SYNTHSEG_SCRIPT.is_file())}")
for weight in ("synthseg_2.0.h5", "synthseg_qc_2.0.h5",
               "synthseg_robust_2.0.h5", "synthseg_parc_2.0.h5"):
    print(f"  {weight:<26} {mark((s.SYNTHSEG_HOME / 'models' / weight).is_file())}")

env_name = getattr(s, "SYNTHSEG_ENV_NAME", "synthseg_38")
print()
print(f"=== SynthSeg interpreter (conda env '{env_name}') ===")
if os.environ.get("SYNTHSEG_PYTHON"):
    print("SYNTHSEG_PYTHON set explicitly in the environment")
elif hasattr(s, "candidate_interpreters"):
    print("searched, in order:")
    for candidate in s.candidate_interpreters():
        print(f"  {'OK     ' if candidate.is_file() else 'missing'}  {candidate}")
else:
    print("(this backend hardcodes a single path rather than searching)")
print(f"using           : {s.SYNTHSEG_PYTHON}\n                  {mark(s.SYNTHSEG_PYTHON.is_file())}")

if hasattr(s, "runtime_summary"):
    print()
    print("=== how a run would execute ===")
    print(" ", s.runtime_summary())

print()
reason = s.check_available()
print(">>> check_available():", reason or "None  —  Run SynthSeg will be enabled")
sys.exit(1 if (reason or stale) else 0)
