from PyQt5.QtWidgets import QApplication
from nml.board_manager import BoardManager
from nml.ui.demo.demo_modes import DemoMode
import sys

def main():
    app = QApplication(sys.argv)
    board_manager = BoardManager(app_type=DemoMode.VIGEM_LATCH)
    board_manager.open() # Opens the BoardManagerViewer GUI
    board_manager.start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()