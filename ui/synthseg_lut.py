"""Colours and names for the FreeSurfer labels SynthSeg produces.

SynthSeg ships no colour table — data/labels table.txt says outright that the
colour of each structure is left to the viewer. The canonical aseg colours are
therefore vendored below so the overlay matches what freeview and ITK-SNAP
show. Cortical parcels (only present with --parc) are not vendored: they are
68 more entries, so they get a deterministic generated palette instead. Point
SYNTHSEG_LUT (or FREESURFER_HOME) at a real FreeSurferColorLUT.txt and every
label, parcels included, uses its canonical colour instead.
"""
import colorsys
import os
from pathlib import Path

import numpy as np

# Canonical FreeSurfer aseg colours for the 33 structures SynthSeg 2.0 labels.
# Right-hemisphere labels share their left counterpart's colour, as in aseg.
ASEG = {
    2: ("left cerebral white matter", (245, 245, 245)),
    3: ("left cerebral cortex", (205, 62, 78)),
    4: ("left lateral ventricle", (120, 18, 134)),
    5: ("left inferior lateral ventricle", (196, 58, 250)),
    7: ("left cerebellum white matter", (220, 248, 164)),
    8: ("left cerebellum cortex", (230, 148, 34)),
    10: ("left thalamus", (0, 118, 14)),
    11: ("left caudate", (122, 186, 220)),
    12: ("left putamen", (236, 13, 176)),
    13: ("left pallidum", (12, 48, 255)),
    14: ("3rd ventricle", (204, 182, 142)),
    15: ("4th ventricle", (42, 204, 164)),
    16: ("brain-stem", (119, 159, 176)),
    17: ("left hippocampus", (220, 216, 20)),
    18: ("left amygdala", (103, 255, 255)),
    24: ("csf", (60, 60, 60)),
    26: ("left accumbens area", (255, 165, 0)),
    28: ("left ventral DC", (165, 42, 42)),
    41: ("right cerebral white matter", (245, 245, 245)),
    42: ("right cerebral cortex", (205, 62, 78)),
    43: ("right lateral ventricle", (120, 18, 134)),
    44: ("right inferior lateral ventricle", (196, 58, 250)),
    46: ("right cerebellum white matter", (220, 248, 164)),
    47: ("right cerebellum cortex", (230, 148, 34)),
    49: ("right thalamus", (0, 118, 14)),
    50: ("right caudate", (122, 186, 220)),
    51: ("right putamen", (236, 13, 176)),
    52: ("right pallidum", (12, 48, 255)),
    53: ("right hippocampus", (220, 216, 20)),
    54: ("right amygdala", (103, 255, 255)),
    58: ("right accumbens area", (255, 165, 0)),
    60: ("right ventral DC", (165, 42, 42)),
}

# --parc replaces cortex (3/42) with Desikan-Killiany parcels in these ranges.
_N_PARCELS_PER_HEMISPHERE = 35
PARC_LH = range(1001, 1036)
PARC_RH = range(2001, 2036)
MAX_LABEL = 2035

_LUT_ENV = "SYNTHSEG_LUT"


def _parcellation_color(label):
    """A stable, well-spread colour for a DK parcel.

    Hemisphere counterparts (1001/2001, ...) share a hue, as in the real LUT;
    the golden-angle step keeps neighbouring parcel numbers visually distinct.
    """
    index = (label - 1001) if label in PARC_LH else (label - 2001)
    # 13 is coprime with 35, so this walks all 35 hue slots exactly once: hues
    # stay evenly spaced (1/35 apart) while consecutive parcel numbers land far
    # apart on the wheel. Neighbouring slots then take different
    # saturation/value tiers, so even the closest pair separates clearly.
    slot = (index * 13) % _N_PARCELS_PER_HEMISPHERE
    hue = slot / _N_PARCELS_PER_HEMISPHERE
    saturation = (0.80, 0.45, 0.62)[slot % 3]
    value = (0.99, 0.90, 0.68)[slot % 3]
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def _external_lut_path():
    explicit = os.environ.get(_LUT_ENV)
    if explicit:
        return Path(explicit)
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if freesurfer_home:
        return Path(freesurfer_home) / "FreeSurferColorLUT.txt"
    return None


def load_external_lut(path):
    """Parse a FreeSurferColorLUT.txt into {label: (name, (r, g, b))}."""
    entries = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        label = int(fields[0])
        if label > MAX_LABEL:
            continue
        entries[label] = (fields[1], (int(fields[2]), int(fields[3]), int(fields[4])))
    return entries


def _build_table():
    """label -> (name, (r, g, b)) for every label SynthSeg can emit."""
    table = dict(ASEG)
    for label in list(PARC_LH) + list(PARC_RH):
        side = "ctx-lh" if label in PARC_LH else "ctx-rh"
        index = (label - 1001) if label in PARC_LH else (label - 2001)
        table[label] = (f"{side}-parcel-{index + 1:02d}", _parcellation_color(label))

    path = _external_lut_path()
    if path is not None and path.is_file():
        try:
            table.update(load_external_lut(path))
        except Exception:
            # A malformed or unreadable LUT should never stop the overlay from
            # rendering — the vendored colours are a complete fallback.
            pass
    return table


TABLE = _build_table()


def build_lut_array():
    """(MAX_LABEL + 1, 3) float array so a slice colourises by fancy-indexing.

    With --parc a slice can carry ~100 distinct labels; indexing a lookup table
    avoids a Python loop over all of them per repaint.
    """
    lut = np.zeros((MAX_LABEL + 1, 3), dtype=np.float32)
    for label, (_, rgb) in TABLE.items():
        lut[label] = np.array(rgb, dtype=np.float32) / 255.0
    return lut


LUT_ARRAY = build_lut_array()


def label_name(label):
    entry = TABLE.get(int(label))
    return entry[0] if entry else f"label {int(label)}"


def label_color_hex(label):
    entry = TABLE.get(int(label))
    rgb = entry[1] if entry else (128, 128, 128)
    return "#{:02x}{:02x}{:02x}".format(*rgb)
