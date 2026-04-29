import socket
import json
import threading

class ParameterSocket:
    host: str = '127.0.0.1'
    port: int = 5211
    num_channels: int = 8
    scale_factor: float = 100.0
    client: socket.socket = None
    server: socket.socket = None
    addr: tuple = None

    def __init__(self, host: str = '127.0.0.1', port: int = 5211, num_channels: int = 8, scale_factor: float = 100.0):
        self.host = host
        self.port = port
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.client = None

        self._start_server()

    def _start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"[ParameterSocket] Waiting for Unity to connect on {self.host}:{self.port}...")

        threading.Thread(target=self._accept_connection, daemon=True).start()

    def _accept_connection(self):
        self.client, addr = self.server.accept()
        print(f"[ParameterSocket] Unity connected from {addr}")
        threading.Thread(target=self._handle_requests, daemon=True).start()
        self.send_config()  # proactively push config on connect

    def _handle_requests(self):
        while True:
            try:
                data = self.receive_parameters()
                if data.get("Name") == "request_config":
                    self.send_config()
            except Exception as e:
                print(f"[ParameterSocket] Connection closed or error: {e}")
                break

    def send_param(self, name, value):
        message = json.dumps({"Name": name, "Value": value}).encode('utf-8')
        length = len(message).to_bytes(4, byteorder='big')
        self.client.sendall(length + message)

    def send_config(self):
        self.send_param("num_channels", float(self.num_channels))
        self.send_param("scale_factor", float(self.scale_factor))

    def receive_parameters(self):
        length = int.from_bytes(self.client.recv(4), byteorder='big')
        msg = self.client.recv(length)
        return json.loads(msg.decode("utf-8"))

    def close(self):
        try:
            if self.client:
                self.client.close()
            self.server.close()
        except Exception:
            pass
