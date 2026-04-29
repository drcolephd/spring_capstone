import socket
import struct
import threading

class StreamSocket:
    host: str = '127.0.0.1'
    port: int = 5210
    client: socket.socket = None
    server: socket.socket = None
    addr: tuple = None

    def __init__(self, host='127.0.0.1', port=5210):
        self.host = host
        self.port = port
        self.client = None
        self.addr = None
        self.server = None
        self.lock = threading.Lock()

        self._start_server()

    def _start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)

        print(f"[StreamSocket] Waiting for Unity to connect on {self.host}:{self.port}...")
        threading.Thread(target=self._accept_connection, daemon=True).start()

    def _accept_connection(self):
        while True:
            try:
                self.client, self.addr = self.server.accept()
                print(f"[StreamSocket] Unity connected from {self.addr}")
            except Exception as e:
                print(f"[StreamSocket] Error in accept(): {e}")
                continue

    def send_rates(self, rate_list: list[float], scale_factor: int):
        with self.lock:
            if not self.client:
                return
            try:
                scaled = [min(max(0, int(rate * scale_factor)), 65535) for rate in rate_list]
                packet = struct.pack(f'>{len(scaled)}H', *scaled)
                self.client.sendall(packet)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                print(f"[StreamSocket] Connection lost: {e}. Waiting for reconnect...")
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None  # Mark connection as closed

    def close(self):
        with self.lock:
            try:
                if self.client:
                    self.client.close()
                if self.server:
                    self.server.close()
            except Exception:
                pass
