import pyqtgraph as pg

from core.constants import LABEL_NAMES

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

# Per-class segmentation colors — dark-mode categorical slots 1-3 (blue, aqua, yellow), used in fixed order so each class keeps a stable, CVD-safe hue.
SEGMENTATION_COLORS = {
    1: "#3987e5",  # NCR — necrotic core
    2: "#199e70",  # ED  — edema
    3: "#c98500",  # ET  — enhancing tumor
}
CLASS_LABELS = {label_id: name for label_id, name in LABEL_NAMES.items() if label_id != 0}

# RANO bidimensional-measurement line colors, one per region type — picked
# to stay visually distinct from SEGMENTATION_COLORS above.
RANO_LINE_COLORS = {"CE": "#ff2fb0", "nonCE": "#39e6ff"}


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

QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}
"""


def apply_pg_theme():
    pg.setConfigOption("background", SURFACE)
    pg.setConfigOption("foreground", TEXT)
