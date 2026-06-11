import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import QSS


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
