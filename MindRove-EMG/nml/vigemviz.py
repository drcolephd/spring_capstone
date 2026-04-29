from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon, QFont
from nml.vigem import ViGEmEnum
from nml.local_paths import paths
from nml.hooks import HookManager, VK_SUBTRACT, VK_ESCAPE 
import os

class ViGEmViz(QMainWindow):
    """
    Visual overlay for ViGEm virtual gamepad with mouse and keyboard interactions.
    """
    buttonUpRequest = pyqtSignal(int)  # Signal for user-generated button release
    buttonDownRequest = pyqtSignal(int)  # Signal for user-generated button press
    newTargetWindow = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        xbox_image_filename = os.path.join(paths['assets'], 'xbox-controller.png')
        self.setWindowTitle("ViGEm Visualizer Overlay")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon(xbox_image_filename))
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Ignore mouse events

        # Load controller image
        self.controller_image = QPixmap(xbox_image_filename)

        # Mapping of button byte codes to coordinates
        self.button_map = {
            0x1000: (586, 257),  # A button
            0x2000: (635, 208),  # B button
            0x4000: (538, 212),  # X button
            0x8000: (585, 163),  # Y button
            0x0100: (217, 87),  # LB button
            0x0200: (595, 87),  # RB button
            0x0020: (355, 211),  # Back button
            0x0010: (459, 211),  # Start button
            0x0040: (223, 208),  # Left Thumb button
            0x0080: (494, 317),  # Right Thumb button
            0x0001: (313, 353),  # D-Pad Up
            0x0002: (313, 292),  # D-Pad Down
            0x0004: (277, 320),  # D-Pad Left
            0x0008: (345, 320),  # D-Pad Right
        }
        self.pressed_button = []

        # Position the window at the top-left-most corner of all monitors
        self.move_to_top_left()

        # Install hooks using HookManager
        self.hook_manager = ViGEmHookManager(self)
        self.hook_manager.targetWindowUpdated.connect(self.set_new_target_window)
        self.hook_manager.install_hooks()

    def add_gamepad(self, gamepad):
        # Connect signals
        gamepad.buttonPressed.connect(self.on_button_pressed)
        self.buttonDownRequest.connect(gamepad.on_button_down_request)
        self.buttonUpRequest.connect(gamepad.on_button_up_request)
        self.newTargetWindow.connect(gamepad.set_target_window)

    def move_to_top_left(self):
        """
        Move the window to the top-left corner of the primary screen.
        """
        desktop = QApplication.desktop()
        screen_geometry = desktop.screenGeometry(0)  # Primary screen
        self.move(screen_geometry.left(), screen_geometry.top())

    def closeEvent(self, event):
        """
        Ensure the application quits when the window is closed.
        """
        print("[ViGEmViz]::Window closed. Exiting application...")
        self.hook_manager.uninstall_hooks()
        QApplication.quit()
        event.accept()

    def paintEvent(self, event):
        """
        Handle custom painting for the overlay.
        """
        super().paintEvent(event)
        painter = QPainter(self)

        # Draw the controller image
        painter.setOpacity(0.65)  # 65% transparency
        painter.drawPixmap(0, 0, self.controller_image)

        # Draw the red circles for pressed buttons
        painter.setBrush(QColor(255, 0, 0, 200))
        painter.setPen(Qt.NoPen)
        for button in self.pressed_button:
            button_bytes = ViGEmEnum.encode([button])
            if button_bytes in self.button_map:
                x, y = self.button_map[button_bytes]
                painter.drawEllipse(x - 20, y - 20, 40, 40)

        # Draw overlay instructions
        painter.setOpacity(0.85)
        painter.setBrush(QColor(255, 255, 255, 220))
        painter.drawRect(0, 0, 250, 30)
        painter.setPen(QColor(0, 0, 0))  # Black color
        painter.setFont(QFont("Consolas", 10, weight=900, italic=True))
        painter.drawText(10, 20, "Press Esc to close overlay")

    @pyqtSlot(int)
    def on_button_pressed(self, byte_code):
        """
        Update UI to show the pressed button.

        :param byte_code: The byte code of the pressed button.
        """
        self.pressed_button = ViGEmEnum.decode(byte_code)
        self.update()

    @pyqtSlot(int)
    def set_new_target_window(self, hwnd):
        self.newTargetWindow.emit(hwnd)


class ViGEmHookManager(HookManager):
    """
    Custom hook manager for ViGEmViz to handle mouse and keyboard input.
    """
    targetWindowUpdated = pyqtSignal(int)

    def __init__(self, window):
        super().__init__(window_x=window.geometry().x(), window_y=window.geometry().y())
        self.window = window

    def on_mouse_down(self, btn, x, y):
        """
        Handle mouse button down events.

        :param btn: Mouse button (1 = Left, 2 = Middle, 3 = Right).
        :param x: X-coordinate of the event relative to the window.
        :param y: Y-coordinate of the event relative to the window.
        """
        if btn == 1:  # Left mouse button
            output_mask = 0x0000
            for byte_code, (bx, by) in self.window.button_map.items():
                if (bx - 20) <= x <= (bx + 20) and (by - 20) <= y <= (by + 20):
                    print(f"[ViGEmViz]::Clicked {ViGEmEnum.decode(byte_code)} button (code: 0x{byte_code:04x})!")
                    output_mask += byte_code
            self.window.buttonDownRequest.emit(output_mask)

    def on_mouse_up(self, btn, x, y):
        """
        Handle mouse button up events.

        :param btn: Mouse button (1 = Left, 2 = Middle, 3 = Right).
        :param x: X-coordinate of the event relative to the window.
        :param y: Y-coordinate of the event relative to the window.
        """
        if btn == 1:  # Left mouse button
            output_mask = 0xFFFF
            for byte_code, (bx, by) in self.window.button_map.items():
                if (bx - 20) <= x <= (bx + 20) and (by - 20) <= y <= (by + 20):
                    print(f"[ViGEmViz]::Released {ViGEmEnum.decode(byte_code)} button!")
                    output_mask -= byte_code
            self.window.buttonUpRequest.emit(output_mask)

    def on_key_down(self, vk_code, vk_unicode, modifiers):
        """
        Handle key down events.

        :param vk_code: The virtual key code of the pressed key.
        :param vk_unicode: The Unicode representation of the key, if available.
        :param modifiers: Modifier keys (e.g., Shift, Ctrl, Alt).
        """
        if vk_code == VK_ESCAPE:
            print("[ViGEmViz]::Escape key pressed. Exiting overlay...")
            self.window.close()
        elif vk_code == VK_SUBTRACT:
            self.targetWindowUpdated.emit(self.set_active_window_hwnd())
