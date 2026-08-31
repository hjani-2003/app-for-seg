"""A result table in a window of its own.

Radiomics and RANO used to be docks tabbed into the right-hand edge of the
viewer. Both want more room than that column can give: the feature table is
1500 rows deep and up to twenty columns wide, and the RANO table is eight
columns of numbers that are read against the slice they measure. Docked, they
squeezed the slices; tabbed, only one of them was visible at a time. As
separate windows they open on demand, can be dragged onto a second monitor,
and leave the viewer to show images.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ui.screen_fit import fit_to_screen
from ui.style import DARK_STYLESHEET


class ToolWindow(QWidget):
    # What the window asks for the first time it opens, clamped to the screen.
    DEFAULT_SIZE = (900, 620)

    def __init__(self, title):
        # Deliberately parentless: a child window would be pinned above the
        # viewer, and these are meant to sit beside it — or behind it — as the
        # user prefers. MRIViewer.closeEvent closes them, since a top-level
        # window left open would keep the application running after the viewer
        # has gone.
        super().__init__()
        self.setWindowTitle(title)

        # A parentless window is not in the viewer's widget tree, so it does
        # not inherit the stylesheet set there and would otherwise open in the
        # platform's default light palette.
        self.setStyleSheet(DARK_STYLESHEET)

        self._positioned = False

    def show_window(self):
        """Open it, or bring an already-open one back to the front.

        Sized on first open only: after that the window keeps wherever the user
        put it, which is the point of it being a window.
        """
        if not self._positioned:
            fit_to_screen(self, *self.DEFAULT_SIZE)
            self._positioned = True

        self.show()
        # A minimised window is still "visible" to Qt, so raise_ alone would
        # leave the button looking like it did nothing.
        if self.isMinimized():
            self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()
