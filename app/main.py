import sys 
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from pathlib import Path

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    qss_path = Path(__file__).parent /"style"/ "style.qss"
    app.setStyleSheet(qss_path.read_text())

    window.show()
    sys.exit(app.exec())