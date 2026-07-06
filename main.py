import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MRIViewer


def main():
    app = QApplication(sys.argv)
    viewer = MRIViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
