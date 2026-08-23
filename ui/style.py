import pyqtgraph as pg

BACKGROUND = "#15161a"
SURFACE = "#1c1e24"
BORDER = "#2c2f38"
TEXT = "#e7e9ee"
TEXT_MUTED = "#8a8f9b"
ACCENT = "#5b8def"
ACCENT_HOVER = "#6f9bf3"
ACCENT_PRESSED = "#4a78d1"
DISABLED_BG = "#25272e"
DISABLED_TEXT = "#5a5e68"

# Per-class segmentation colors. Chosen by maximising CIEDE2000 separation as
# actually displayed — alpha-blended over brain-intensity grays — between the
# three classes, under normal vision and under simulated deuteranopia and
# protanopia, then tie-broken on distance from the FreeSurfer anatomy palette
# in ui/synthseg_lut.py so tumour stays readable over a SynthSeg overlay.
# The previous blue/green/orange set sat almost exactly on top of the pallidum,
# thalamus and accumbens colors (dE 7.2, 6.5 and 2.3 — the last below the
# just-noticeable-difference threshold).
SEGMENTATION_COLORS = {
    1: "#bf0000",  # NCR — necrotic core
    2: "#26beff",  # ED  — edema
    3: "#ffff00",  # ET  — enhancing tumor
}
CLASS_LABELS = {1: "NCR", 2: "ED", 3: "ET"}


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}

QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}

QPushButton {{
    background-color: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton:disabled {{
    background-color: {DISABLED_BG};
    color: {DISABLED_TEXT};
}}

QComboBox {{
    background-color: {BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 120px;
}}

QComboBox:disabled {{
    color: {DISABLED_TEXT};
}}

QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QLabel {{
    color: {TEXT};
}}

QCheckBox {{
    color: {TEXT};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BACKGROUND};
}}

QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    /* A checkmark glyph would need an image asset, so a filled inset box
       marks the checked state instead. */
    image: none;
}}

QCheckBox::indicator:disabled {{
    background-color: {DISABLED_BG};
    border-color: {DISABLED_BG};
}}

QCheckBox:disabled {{
    color: {DISABLED_TEXT};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:disabled {{
    background: {DISABLED_TEXT};
}}

QProgressBar {{
    background-color: {BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}

QDockWidget {{
    color: {TEXT_MUTED};
    font-weight: 600;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    text-align: left;
}}

QScrollArea {{
    background-color: {BACKGROUND};
    border: none;
}}

QScrollBar:vertical {{
    background: {BACKGROUND};
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}
"""


def apply_pg_theme():
    pg.setConfigOption("background", SURFACE)
    pg.setConfigOption("foreground", TEXT)
