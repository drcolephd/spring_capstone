from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from nml.teensy import Teensy
from nml.serial_plot_window import SerialPlotWindow
import sys

def main():
    app = QApplication(sys.argv)

    # Set up Teensy and SerialPlotWindow
    teensy = Teensy(n_channels=2)  # Request 2 channels
    plot_window = SerialPlotWindow(n_channels=2, buffer_size=300, refresh_rate=30.0)

    # Connect Teensy's newData signal to the plot window's update method
    teensy.newData.connect(plot_window.update_plot)

    # Start the plot window and polling
    plot_window.show()
    teensy.start()

    # Clean up on exit
    def on_exit():
        teensy.stop()
        teensy.close()

    app.aboutToQuit.connect(on_exit)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
