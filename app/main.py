import sys
import pathlib
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow  # your import

if getattr(sys, 'frozen', False):
    BASE_DIR = pathlib.Path(sys._MEIPASS)
else:
    BASE_DIR = pathlib.Path(__file__).resolve().parent

app = QApplication(sys.argv)
app.setStyleSheet((BASE_DIR / "style" / "style.qss").read_text())

window = MainWindow()
window.show()
sys.exit(app.exec())