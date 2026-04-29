import socket
from PyQt5.QtCore import QObject, pyqtSlot

class TCP(QObject):
    """Manages a TCP socket connection and provides methods for sending digital or analog inputs."""

    def __init__(self, ip: str = "127.0.0.1", gamepad_port: int = 6053, mouse_port: int = 6054):
        """Initialize a TCP connection."""
        super(TCP, self).__init__()
        self.ip = ip
        self.gamepad_port = gamepad_port
        self.gamepad_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.gamepad_sock.connect((self.ip, self.gamepad_port))
            print(f"Connected to {self.ip}:{self.gamepad_port}")
        except ConnectionRefusedError:
            raise ConnectionRefusedError(f"Unable to connect to {self.ip}:{self.gamepad_port}")
        
        self.mouse_port = mouse_port
        self.mouse_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.mouse_sock.connect((self.ip, self.mouse_port))
            print(f"Connected to {self.ip}:{self.mouse_port}")
        except ConnectionRefusedError:
            raise ConnectionRefusedError(f"Unable to connect to {self.ip}:{self.mouse_port}")
        
    @pyqtSlot(float, float)
    def handle_mouse(self, delta_x, delta_y):
        self.send_mouse_movement_input(round(delta_x), round(delta_y))

    def send_message_gamepad(self, message: str):
        """Send a raw message to the TCP gamepad-handler server socket."""
        try:
            self.gamepad_sock.sendall(message.encode())
            print(f"Sent: {message.strip()}")
        except Exception as e:
            print(f"Error sending message: {e}")

    def send_message_mouse(self, message: str):
        """Send a raw message to the TCP mouse-handler server socket."""
        try:
            self.mouse_sock.sendall(message.encode())
            print(f"Sent: {message.strip()}")
        except Exception as e:
            print(f"Error sending message: {e}")

    def send_digital_input(self, key: str, action: str):
        """
        Send a digital input message.

        Args:
            key: The key to press/release (e.g., 'a', 'b', 'y').
            action: The action ('0' for press, '1' for release).
        """
        message = f"{key}{action}\r\n"
        self.send_message_gamepad(message)

    def send_mouse_movement_input(self, dx: int, dy: int):
        """
        Send an analog input message.

        Args:
            direction: The direction of movement ('x', 'click', 'scroll', 'on', 'off').
            dx: The x-delta or mouse action value.
            dy: The y-delta (optional).
        """
        message = f"x,{dx},{dy}\n"
        self.send_message_mouse(message)

    def send_mouse_button_input(self, button, button_state):
        message = f"{button},{button_state}\r\n"
        self.send_message_mouse(message)

    def close(self):
        """Close the TCP connection."""
        self.gamepad_sock.close()
        print(f"Disconnected from {self.ip}:{self.gamepad_port}")
        if self.mouse_sock is not None:
            self.mouse_sock.close()
            print(f"Disconnected from {self.ip}:{self.mouse_port}")
