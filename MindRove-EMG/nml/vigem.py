import ctypes
from ctypes import wintypes
import os
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from nml.local_paths import paths

class ViGEmPy(QObject):
    buttonPressed = pyqtSignal(int)  # Signal to emit button press byte codes

    def __init__(self):
        """
        Initialize the ViGEmPy instance and load the DLL.
        """
        super().__init__()
        dll_path = os.path.join(paths['lib'], "vigem_gamepad.dll")
        self.dll = ctypes.WinDLL(dll_path)
        self.buttons = 0x0000
        self._define_prototypes()

    def _define_prototypes(self):
        """
        Define the function prototypes for the DLL.
        """
        self.dll.initialize_gamepad.restype = ctypes.c_int
        self.dll.send_gamepad_input.argtypes = [
            ctypes.c_uint16,  # buttons
            ctypes.c_int8,    # leftX
            ctypes.c_int8,    # leftY
            ctypes.c_int8,    # rightX
            ctypes.c_int8,    # rightY
        ]
        self.dll.send_gamepad_input.restype = ctypes.c_int

        self.dll.cleanup_gamepad.restype = None
        self.dll.set_target_window.argtypes = [wintypes.HWND]
        self.dll.set_target_window.restype = ctypes.c_int

    def initialize(self):
        """
        Initialize the virtual gamepad.
        :return: True if successful, False otherwise.
        """
        result = self.dll.initialize_gamepad()
        if result != 0:
            raise RuntimeError(f"Failed to initialize gamepad. Error code: {result}")
        return True
    
    @pyqtSlot(int)
    def set_target_window(self, hwnd):
        """
        Sets the target window for sending input messages.

        :param hwnd: Handle to the target window (HWND).
        """
        result = self.dll.set_target_window(hwnd)
        if result != 0:
            raise RuntimeError(f"Failed to set target window. Error code: {result}")
        else:
            print("[ViGEmPy]::New target window assigned.")

    def send_input(self, buttons, left_x=0, left_y=0, right_x=0, right_y=0):
        """
        Send input to the virtual gamepad.

        :param buttons: Button mask (uint16_t).
        :param left_x: Left joystick X-axis (-128 to 127).
        :param left_y: Left joystick Y-axis (-128 to 127).
        :param right_x: Right joystick X-axis (-128 to 127).
        :param right_y: Right joystick Y-axis (-128 to 127).
        :param emit: Default is True; set False to skip emitting buttonPressed signal. 
        """
        result = self.dll.send_gamepad_input(buttons, left_x, left_y, right_x, right_y)
        if result != 0:
            raise RuntimeError(f"Failed to send input. Error code: {result}")
        self.buttons = buttons
        self.buttonPressed.emit(buttons)
        print(f"[ViGEmPy]::Emitted updated buttons: {ViGEmEnum.decode(buttons)}")

    @pyqtSlot(int)
    def on_button_down_request(self, byte_code):
        print(f"[ViGEmPy]::Received button-down request: {ViGEmEnum.decode(byte_code)}")
        self.send_input(self.buttons | byte_code, 0, 0, 0, 0)

    @pyqtSlot(int)
    def on_button_up_request(self, byte_code):
        print(f"[ViGEmPy]::Received button-up request: {ViGEmEnum.decode(0xFFFF - byte_code)}")
        self.send_input(self.buttons & byte_code, 0, 0, 0, 0)

    def __del__(self):
        """
        Destructor to ensure cleanup is called.
        """
        print("[ViGEmPy]::Destructor called. Cleaning up...")
        self.cleanup()

    def cleanup(self):
        """
        Clean up and release resources.
        """
        self.dll.cleanup_gamepad()

class ViGEmEnum:
    # Define byte codes for buttons
    BUTTONS = {
        'A': 0x1000,
        'B': 0x2000,
        'X': 0x4000,
        'Y': 0x8000,
        'LB': 0x0100,
        'RB': 0x0200,
        'Back': 0x0020,
        'Start': 0x0010,
        'Left Thumb': 0x0040,
        'Right Thumb': 0x0080,
        'D-Pad Up': 0x0001,
        'D-Pad Down': 0x0002,
        'D-Pad Left': 0x0004,
        'D-Pad Right': 0x0008,
    }

    @staticmethod
    def decode(byte_code):
        # Decode byte code into button names
        return [button for button, code in ViGEmEnum.BUTTONS.items() if byte_code & code]

    @staticmethod
    def encode(buttons=None):
        # Encode button names into a byte code
        if (buttons is None) or (len(buttons) == 0):
            return 0x0000
        else:
            return sum(ViGEmEnum.BUTTONS[button] for button in buttons if button in ViGEmEnum.BUTTONS)


if __name__ == "__main__":
    import time

    # Example usage
    time.sleep(5)
    gamepad = ViGEmPy()
    print("[ViGEmPy]::Initializing gamepad...")
    gamepad.initialize()

    try:
        # Cycle through buttons
        buttons = ["A", "B", "X", "Y"]
        for button in buttons:
            print(f"[ViGEmPy]::Pressing '{button}' button...")
            gamepad.send_input(ViGEmEnum.encode([button]))  # Press the button
            time.sleep(0.5)  # Hold the button for 0.5 seconds
            print(f"[ViGEmPy]::Releasing '{button}' button...")
            gamepad.send_input(0x0000)  # Release all buttons
            time.sleep(1.5)  # Wait for 1.5 seconds
    except Exception as e:
        print(f"[ViGEmPy]::An error occurred: {e}")

    print("[ViGEmPy]::Exiting...")

