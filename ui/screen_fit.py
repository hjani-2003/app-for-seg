"""Sizing a window to the screen it actually opens on.

Shared by the viewer and the tool windows. Both are laid out against a design
size chosen on a desktop monitor, and a window that opens larger than its
screen has its lower-right corner — the slice slider, or the last column of a
table — off screen from the start. The screen is read at run time rather than
assumed, so the same code fits a 4K desktop and a scaled laptop panel.
"""
from PySide6.QtGui import QGuiApplication

# availableGeometry() excludes taskbars and panels but not the window's own
# title bar and borders, so a window sized to exactly the work area hangs off
# the bottom by the height of its decorations. This leaves room for them
# without having to guess their size, which is only known after the window
# manager has drawn them.
SCREEN_FRACTION = 0.92


def size_within(widget, available, width, height):
    """Shrink a size to fit a screen's work area, decorations included.

    Never goes below the widget's own minimum — past that point there is
    nothing left to give, and Qt would ignore it anyway.
    """
    minimum = widget.minimumSizeHint()
    return (
        max(min(width, int(available.width() * SCREEN_FRACTION)), minimum.width()),
        max(min(height, int(available.height() * SCREEN_FRACTION)), minimum.height()),
    )


def fit_to_screen(widget, width, height):
    """Open `widget` at the given size, or the screen's size — whichever is
    smaller — centred on the screen it is about to appear on."""
    screen = widget.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        widget.resize(width, height)
        return

    available = screen.availableGeometry()
    width, height = size_within(widget, available, width, height)
    widget.resize(width, height)
    widget.move(
        available.x() + (available.width() - width) // 2,
        available.y() + (available.height() - height) // 2,
    )
