"""Run from the repo root on the GPU machine:  python check_synthseg.py"""
import os, sys
sys.path.insert(0, os.getcwd())
from core.backends import synthseg_backend as s

print("SYNTHSEG_HOME  :", s.SYNTHSEG_HOME)
print("  exists       :", s.SYNTHSEG_HOME.is_dir())
print("SYNTHSEG_SCRIPT:", s.SYNTHSEG_SCRIPT)
print("  exists       :", s.SYNTHSEG_SCRIPT.is_file())
print("SYNTHSEG_PYTHON:", s.SYNTHSEG_PYTHON)
print("  exists       :", s.SYNTHSEG_PYTHON.is_file())
print("weights dir    :", s.SYNTHSEG_HOME / "models")
for w in ("synthseg_2.0.h5", "synthseg_qc_2.0.h5",
          "synthseg_robust_2.0.h5", "synthseg_parc_2.0.h5"):
    p = s.SYNTHSEG_HOME / "models" / w
    print(f"  {w:26s} {'OK' if p.is_file() else 'MISSING'}")
print()
print(">>> check_available():", s.check_available() or "None  (button should be enabled)")
