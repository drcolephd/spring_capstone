import ctypes
import ctypes.wintypes
import time
from PyQt5.QtCore import QObject, pyqtSignal

# Constants for Mouse and Keyboard Hooks
WH_MOUSE_LL = 14
WH_KEYBOARD_LL = 13
LLMHF_INJECTED = 0x00000001

WM_LBUTTONDOWN = 0x0201  # Left mouse button down
WM_LBUTTONUP = 0x0202    # Left mouse button up
WM_RBUTTONDOWN = 0x0204  # Right mouse button down
WM_RBUTTONUP = 0x0205    # Right mouse button up
WM_MBUTTONDOWN = 0x0207  # Middle mouse button down
WM_MBUTTONUP = 0x0208    # Middle mouse button up
WM_MOUSEMOVE = 0x0200    # Mouse movement
WM_MOUSEWHEEL = 0x020A   # Mouse wheel movement
WM_XBUTTONDOWN = 0x020B  # X button down (e.g., side buttons)
WM_XBUTTONUP = 0x020C    # X button up

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt key
VK_ESCAPE = 0x1B
VK_SUBTRACT = 0x6D 

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

TAGGED_EVENT = 0xDEADBEEF

def MAKELPARAM(x, y):
    return (y << 16) | x

# Define CallNextHookEx with proper signature
CALL_NEXT_HOOK_EX = ctypes.windll.user32.CallNextHookEx
CALL_NEXT_HOOK_EX.argtypes = [
    ctypes.wintypes.HHOOK,
    ctypes.c_int,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
]
CALL_NEXT_HOOK_EX.restype = ctypes.c_int

# Define the function signatures for hooks
LL_MOUSE_PROC = ctypes.WINFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM
)
LL_KEYBOARD_PROC = ctypes.WINFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM
)

# Define ULONG_PTR based on platform
if ctypes.sizeof(ctypes.c_void_p) == 8:  # 64-bit system
    ULONG_PTR = ctypes.c_uint64
else:  # 32-bit system
    ULONG_PTR = ctypes.c_uint32

# Define POINT structure
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# Define MSLLHOOKSTRUCT structure for mouse events
class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

def get_window_under_cursor():
    point = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    hwnd = ctypes.windll.user32.WindowFromPoint(point)
    return hwnd

def get_current_mouse_position():
    """
    Retrieve the current mouse position in screen coordinates.
    :return: A tuple (x, y) representing the mouse's position.
    """
    point = POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        return point.x, point.y
    else:
        raise ctypes.WinError(ctypes.get_last_error())


class HookManager(QObject):
    """
    A class to manage low-level keyboard and mouse hooks for handling input events.
    Provides a framework for overriding mouse and keyboard event handlers.
    """
    def __init__(self, window_x: int=0, window_y: int=0, exit_on_escape: bool = True):
        """
        Initializes the HookManager with optional window dimensions for coordinate offset correction.

        :param window_x: The X-coordinate of the window's top-left corner. Default is 0.
        :param window_y: The Y-coordinate of the window's top-left corner. Default is 0.
        :param exit_on_escape: Whether the application should exit on the Escape key press. Default is True.
        """
        super().__init__()
        self.window_x = window_x
        self.window_y = window_y
        self._exit_on_escape = exit_on_escape
        self.mouse_hook = None
        self.keyboard_hook = None
        self._hooks_installed = False
        self.target_hwnd = None
        # Initialize the current mouse position
        self._x, self._y = get_current_mouse_position()

    def __del__(self):
        """
        Ensures that hooks are uninstalled when the instance is deleted.
        """
        self.uninstall_hooks()

    def _is_vigem_message(self, message):
        """
        Determines if the message is related to ViGEmClient.
        :param message: The Windows message to check.
        :return: True if the message is related to ViGEmClient; otherwise, False.
        """
        return False
        # # Replace 0xXXXX with the actual message ID(s) used by ViGEmClient
        # VIGEM_MESSAGE_IDS = [0xXXXX, 0xYYYY]
        # return message.message in VIGEM_MESSAGE_IDS

    def _handle_vigem_message(self, message):
        """
        Handles a message from ViGEmClient.
        :param message: The Windows message to process.
        """
        pass
        # print(f"[HookManager]::Handling ViGEmClient message {message.message}")

        # # Decode or process the message based on its type
        # if message.message == 0xXXXX:
        #     # Handle specific ViGEmClient message type
        #     print(f"[HookManager]::Processed ViGEmClient message with WPARAM={message.wParam}, LPARAM={message.lParam}")
        #     # Example: Forward the message to the target window
        #     if self.target_hwnd:
        #         ctypes.windll.user32.PostMessageW(self.target_hwnd, message.message, message.wParam, message.lParam)
        #     else:
        #         print("[HookManager]::No target HWND stored for forwarding ViGEmClient messages.")
        # else:
        #     print(f"[HookManager]::Unhandled ViGEmClient message: {message.message}")


    def run(self):
        """
        Starts the Windows message loop to process input events and handle custom messages.
        This method blocks until the loop exits.
        """
        MESSAGE = ctypes.wintypes.MSG()
        while ctypes.windll.user32.GetMessageW(ctypes.byref(MESSAGE), None, 0, 0) != 0:
            if MESSAGE.message == ctypes.windll.user32.WM_QUIT:
                print("[HookManager]::WM_QUIT received. Stopped running.")
                break
            print(f"[HookManager]::Intercepted message: HWND={MESSAGE.hWnd}, MSG={MESSAGE.message}, WPARAM={MESSAGE.wParam}, LPARAM={MESSAGE.lParam}")
            # Intercept and handle custom messages here
            if self._is_vigem_message(MESSAGE):
                self._handle_vigem_message(MESSAGE)
            else:
                # Default processing for non-ViGEm messages
                ctypes.windll.user32.TranslateMessage(ctypes.byref(MESSAGE))
                ctypes.windll.user32.DispatchMessageW(ctypes.byref(MESSAGE))

        if self._hooks_installed:
            self.uninstall_hooks()


    def set_active_window_hwnd(self):
        """
        Retrieve and store the HWND of the currently active window.
        """
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            self.target_hwnd = hwnd
            print(f"[HookManager]::Stored HWND: {self.target_hwnd}")
        else:
            print("[HookManager]::Failed to retrieve HWND.")
        return hwnd

    def _send_mouse_event_to_target(self, message, lparam):
        """
        Forward a mouse event to the stored target window.
        :param message: The message type (e.g., WM_LBUTTONDOWN).
        :param lparam: The original LPARAM value from the mouse event.
        """
        if self.target_hwnd:
            # Ensure lparam is cast to a 32-bit value
            lparam_32bit = lparam & 0xFFFFFFFF
            result = ctypes.windll.user32.PostMessageW(self.target_hwnd, message, 0, lparam_32bit)
            if result == 0:
                error_code = ctypes.windll.kernel32.GetLastError()
                print(f"[HookManager]::Failed to send mouse event. Error: {error_code}")

    # Mouse Event Callbacks (To Be Overridden)
    def on_mouse_down(self, btn, x, y):
        """
        Called when a mouse button is pressed.

        :param btn: The mouse button pressed (1 = Left, 2 = Middle, 3 = Right).
        :param x: The X-coordinate of the mouse event, relative to the window.
        :param y: The Y-coordinate of the mouse event, relative to the window.
        """
        pass

    def on_mouse_up(self, btn, x, y):
        """
        Called when a mouse button is released.

        :param btn: The mouse button released (1 = Left, 2 = Middle, 3 = Right).
        :param x: The X-coordinate of the mouse event, relative to the window.
        :param y: The Y-coordinate of the mouse event, relative to the window.
        """
        pass

    def on_mouse_move(self, x, y):
        """
        Called when the mouse is moved.

        :param x: The X-coordinate of the mouse event, relative to the window.
        :param y: The Y-coordinate of the mouse event, relative to the window.
        """
        pass

    def on_mouse_wheel(self, delta, x, y):
        """
        Called when the mouse wheel is scrolled.

        :param delta: The wheel scroll amount (positive for up, negative for down).
        :param x: The X-coordinate of the mouse event, relative to the window.
        :param y: The Y-coordinate of the mouse event, relative to the window.
        """
        pass

    # keyboard key-down callback (to be overriden)
    def on_key_down(self, vk_code, vk_unicode, modifiers):
        """
        Called when a keyboard key is pressed.

        :param vk_code: The virtual key code of the key pressed.
        :param vk_unicode: The character representation of the key, if available.
        :param modifiers: A dictionary containing the state of modifier keys (Shift, Ctrl, Alt).
        """
        if vk_code == VK_SUBTRACT:
            self.set_active_window_hwnd()

    # keyboard key-release callback (to be overriden)
    def on_key_up(self, vk_code, vk_unicode, modifiers):
        """
        Called when a keyboard key is released.

        :param vk_code: The virtual key code of the key released.
        :param vk_unicode: The character representation of the key, if available.
        :param modifiers: A dictionary containing the state of modifier keys (Shift, Ctrl, Alt).
        """
        pass

    def ll_mouse_hook(self, nCode, wParam, lParam):
        """
        Main handler installed for different mouse event hooks. 
        """
        if nCode == 0:  # HC_ACTION
            mouse_event = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            x, y = mouse_event.pt.x, mouse_event.pt.y

            # Calculate relative movement (dx, dy)
            dx = x - self._x
            dy = y - self._y

            # Update the stored mouse position
            self._x, self._y = x, y

            # Check if the event is tagged and skip if so
            if mouse_event.dwExtraInfo and ctypes.cast(mouse_event.dwExtraInfo, ctypes.POINTER(ctypes.c_ulong)).contents.value == TAGGED_EVENT:
                print("[HookManager]::Skipping tagged event.")
                return CALL_NEXT_HOOK_EX(None, nCode, wParam, lParam)

            # Handle mouse events in the application and forward them
            if wParam == WM_LBUTTONDOWN:
                self.on_mouse_down(1, x - self.window_x, y - self.window_y)
            elif wParam == WM_LBUTTONUP:
                self.on_mouse_up(1, x - self.window_x, y - self.window_y)
            elif wParam == WM_RBUTTONDOWN:
                self.on_mouse_down(3, x - self.window_x, y - self.window_y)
            elif wParam == WM_RBUTTONUP:
                self.on_mouse_up(3, x - self.window_x, y - self.window_y)
            elif wParam == WM_MBUTTONDOWN:
                self.on_mouse_down(2, x - self.window_x, y - self.window_y)
            elif wParam == WM_MBUTTONUP:
                self.on_mouse_up(2, x - self.window_x, y - self.window_y)
            elif wParam == WM_MOUSEMOVE:
                self.on_mouse_move(x - self.window_x, y - self.window_y)
            elif wParam == WM_MOUSEWHEEL:
                delta = ctypes.cast(mouse_event.mouseData, ctypes.POINTER(ctypes.c_short)).contents.value
                self.on_mouse_wheel(delta, x - self.window_x, y - self.window_y)
            
            self._send_mouse_event_to_target(wParam, lParam)
        return CALL_NEXT_HOOK_EX(None, nCode, wParam, lParam)

    def ll_keyboard_callback(self, nCode, wParam, lParam):
        if nCode == 0:  # HC_ACTION
            vkCode = ctypes.cast(lParam, ctypes.POINTER(ctypes.c_ulong)).contents.value
            modifiers = HookManager._get_modifiers()
            if wParam == WM_KEYDOWN:
                if (vkCode == VK_ESCAPE) and (self._exit_on_escape):
                    print("[Keyboard Hook]::Escape key pressed. Exiting...")
                    ctypes.windll.user32.PostQuitMessage(0)
                else:
                    self.on_key_down(vkCode, HookManager._vk_code_to_char(vkCode), modifiers)
            elif wParam == WM_KEYUP:
                self.on_key_up(vkCode, HookManager._vk_code_to_char(vkCode), modifiers)
        return CALL_NEXT_HOOK_EX(None, nCode, wParam, lParam)

    # Hook Installation and Removal
    def install_hooks(self):
        """
        Installs the low-level keyboard and mouse hooks. Raises an error if installation fails.
        """
        # Install mouse hook
        self.mouse_proc = LL_MOUSE_PROC(self.ll_mouse_hook)
        self.mouse_hook = ctypes.windll.user32.SetWindowsHookExW(
            WH_MOUSE_LL, self.mouse_proc, None, 0
        )
        if not self.mouse_hook:
            error_code = ctypes.windll.kernel32.GetLastError()
            raise RuntimeError(f"Failed to install mouse hook. Error code: {error_code}")
        print("[Mouse Hook]::Installed successfully.")

        # Install keyboard hook
        self.keyboard_proc = LL_KEYBOARD_PROC(self.ll_keyboard_callback)
        self.keyboard_hook = ctypes.windll.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self.keyboard_proc, None, 0
        )
        if not self.keyboard_hook:
            ctypes.windll.user32.UnhookWindowsHookEx(self.mouse_hook)
            self.mouse_hook = None
            error_code = ctypes.windll.kernel32.GetLastError()
            raise RuntimeError(f"Failed to install keyboard hook. Error code: {error_code}")
        print("[Keyboard Hook]::Installed successfully.")
        self._hooks_installed = True

    def uninstall_hooks(self):
        """
        Uninstalls the low-level keyboard and mouse hooks.
        """
        if self.mouse_hook:
            ctypes.windll.user32.UnhookWindowsHookEx(self.mouse_hook)
            self.mouse_hook = None
        if self.keyboard_hook:
            ctypes.windll.user32.UnhookWindowsHookEx(self.keyboard_hook)
            self.keyboard_hook = None
        print("Hooks uninstalled successfully.")
        self._hooks_installed = False

    @staticmethod
    def _get_modifiers():
        """
        Check the state of modifier keys (Shift, Ctrl, Alt).
        Returns a dictionary with the state of each modifier.
        """
        # Check if modifiers are pressed
        shift = ctypes.windll.user32.GetKeyState(VK_SHIFT) & 0x8000 != 0
        ctrl = ctypes.windll.user32.GetKeyState(VK_CONTROL) & 0x8000 != 0
        alt = ctypes.windll.user32.GetKeyState(VK_MENU) & 0x8000 != 0

        return {"Shift": shift, "Ctrl": ctrl, "Alt": alt}

    @staticmethod
    def _vk_code_to_char(vk_code):
        """
        Converts a virtual key code to a human-readable character.
        """
        buffer = ctypes.create_unicode_buffer(8)  # Buffer to store the translated character
        state = (ctypes.c_ubyte * 256)()          # Keyboard state array

        # Get the current keyboard state
        if not ctypes.windll.user32.GetKeyboardState(ctypes.byref(state)):
            return None

        # Translate the virtual key code
        result = ctypes.windll.user32.ToUnicode(
            vk_code,                                # Virtual key code
            ctypes.windll.user32.MapVirtualKeyW(vk_code, 0),  # Hardware scan code
            ctypes.byref(state),                   # Keyboard state
            buffer,                                # Output buffer
            len(buffer),                           # Size of buffer
            0                                      # Flags
        )

        if result > 0:
            return buffer.value  # Return the translated character
        return None  # No character could be translated


if __name__ == "__main__":
    # Example Subclass
    class ExampleHookManager(HookManager):
        def on_mouse_down(self, btn, x, y):
            print(f"[ExampleHookManager]::Mouse-Button-{btn} Down at ({x}, {y})")

        def on_mouse_up(self, btn, x, y):
            print(f"[ExampleHookManager]::Mouse-Button-{btn} Up at ({x}, {y})")

        def on_mouse_move(self, x, y):
            print(f"[ExampleHookManager]::Mouse moved to ({x}, {y})")

        def on_mouse_wheel(self, delta, x, y):
            direction = "up" if delta > 0 else "down"
            print(f"[ExampleHookManager]::Mouse wheel scrolled {direction} at ({x}, {y})")

        def on_key_down(self, vk_code, vk_unicode, modifiers):
            print(f"[ExampleHookManager]::Key '{vk_unicode}' pressed (VK code: {vk_code})")
            if any(modifiers.values()):
                # Print non-zero fields
                active_modifiers = {key: value for key, value in modifiers.items() if value}
                print(f"Active Modifiers: {active_modifiers}")

        def on_key_up(self, vk_code, vk_unicode, _):
            print(f"[ExampleHookManager]::Key '{vk_unicode}' released (VK code: {vk_code})")

    try:
        hook_manager = ExampleHookManager(window_x=100, window_y=100, exit_on_escape=True)  # Example window offsets
        hook_manager.install_hooks()
        print("Hooks installed. Press Escape to exit.")
        hook_manager.run()
        print("Exited successfully.")
    except Exception as e:
        print(f"Error: {e}")
