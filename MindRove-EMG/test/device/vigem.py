from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from nml.vigem import ViGEmPy, ViGEmEnum
from nml.vigemviz import ViGEmViz
import sys

def main():
    app = QApplication(sys.argv)
    gamepad = ViGEmPy()
    gamepad.initialize()
    viz = ViGEmViz()
    viz.add_gamepad(gamepad)
    viz.show()

    # List of buttons to toggle
    buttons = ["A", "B", "X", "Y"]
    current_button_index = [0]  # Use a list to allow modification within the scope

    def toggle_button():
        # Get the current button to toggle
        button = buttons[current_button_index[0]]
        byte_code = ViGEmEnum.encode([button])

        # Send button press
        print(f"Toggling {button} button.")
        gamepad.send_input(gamepad.buttons | byte_code)

        # Release button after a short delay
        QTimer.singleShot(750, lambda: gamepad.send_input(gamepad.buttons & (0xFFFF - byte_code)))  # Release only this button after 500ms

        # Move to the next button
        current_button_index[0] = (current_button_index[0] + 1) % len(buttons)

    # Set up a timer to toggle buttons every second
    timer = QTimer()
    timer.timeout.connect(toggle_button)
    timer.start(2500)

    app.exec_()
    gamepad.cleanup()


if __name__ == "__main__":
    main()