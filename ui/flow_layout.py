"""A layout that wraps its widgets onto more rows instead of growing wider.

The control panels used to sit in a plain QHBoxLayout, which made the sum of
their minimum widths a hard floor for the whole window: about 2100px, more
than a 1920px screen. Maximising then left the right-hand edge off screen,
because Qt will not size a window below its minimum. Wrapping removes the
floor — the row is only ever as wide as its widest single panel, and anything
that does not fit moves to the next line.
"""
from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QScrollArea, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=10):
        super().__init__(parent)
        self.setContentsMargins(QMargins(margin, margin, margin, margin))
        self.setSpacing(spacing)
        self._items = []

    # ---------- QLayout plumbing ----------

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        # Never asks for extra space in either direction: the row is a strip of
        # controls above the images, and the images should get the slack.
        return Qt.Orientations(Qt.Orientation(0))

    # ---------- wrapping ----------

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _layout(self, rect, apply):
        """Place items left to right, wrapping at the right edge.

        Returns the total height needed, so heightForWidth can ask without
        moving anything.
        """
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y = effective.x(), effective.y()
        row_height = 0

        for item in self._items:
            hint = item.sizeHint()
            # A panel wider than the window still has to fit inside it, or the
            # oversized item reintroduces the very overflow this layout exists
            # to prevent.
            width = min(hint.width(), effective.width())
            height = hint.height()

            if row_height and x + width > effective.right() + 1:
                x = effective.x()
                y += row_height + self.spacing()
                row_height = 0

            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, height)))

            x += width + self.spacing()
            row_height = max(row_height, height)

        return y + row_height - rect.y() + margins.bottom()


class FlowRow(QWidget):
    """A widget whose children flow — use this rather than nesting FlowLayout
    inside a QVBoxLayout directly.

    A box layout only asks a *widget* for heightForWidth when its size policy
    advertises it, so without this wrapper a wrapped second row would be drawn
    on top of whatever sits below it.
    """

    def __init__(self, spacing=10, parent=None):
        super().__init__(parent)
        self._flow = FlowLayout(self, margin=0, spacing=spacing)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def addWidget(self, widget):
        self._flow.addWidget(widget)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._flow.heightForWidth(width)

    def sizeHint(self):
        minimum = self._flow.minimumSize()
        return QSize(
            minimum.width(), self.heightForWidth(self.width() or minimum.width())
        )

    def minimumSizeHint(self):
        return self._flow.minimumSize()


class FlowStrip(QScrollArea):
    """The control strip: wraps when narrow, scrolls when short.

    Wrapping alone trades a width problem for a height one — five panels
    stacked in a single column claim ~980px of minimum height, which a 768px
    laptop screen cannot give either. Scrolling caps that: the strip asks for
    the height its current wrap needs and gives it up when the window is too
    short to spare it.
    """

    # Two panels' worth, so the strip is still usable at its smallest.
    _MIN_HEIGHT = 90

    # The slices are what the window is for, so controls never take more than
    # this share of it — past that the strip scrolls instead of growing.
    _MAX_FRACTION = 0.4

    def __init__(self, spacing=10, parent=None):
        super().__init__(parent)
        self._row = FlowRow(spacing=spacing)
        self.setWidget(self._row)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.viewport().setAutoFillBackground(False)
        self._row.setAutoFillBackground(False)

    def addWidget(self, widget):
        self._row.addWidget(widget)

    def minimumSizeHint(self):
        return QSize(self._row.minimumSizeHint().width(), self._MIN_HEIGHT)

    def sizeHint(self):
        width = self.viewport().width() or self._row.minimumSizeHint().width()
        wanted = self._row.heightForWidth(width)
        window_height = self.window().height()
        if window_height > 0:
            cap = max(self._MIN_HEIGHT, int(window_height * self._MAX_FRACTION))
            wanted = min(wanted, cap)
        return QSize(self._row.minimumSizeHint().width(), wanted)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # A different width means a different number of rows, and a different
        # height moves the cap, so what the strip asks for has to be recomputed
        # rather than left at the old wrap.
        if event.oldSize() != event.size():
            self.updateGeometry()
