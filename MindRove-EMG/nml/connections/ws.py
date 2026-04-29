import asyncio
import websockets
import socket
import json
from PyQt5.QtCore import QThread, pyqtSignal

class WebSocketServer(QThread):
    """
    WebSocket server running in a separate thread, listening for integer messages
    and inserting them as markers using the MindRove board_shim.
    """

    message_received = pyqtSignal(int)  # Signal to notify received values

    def __init__(self, board_shim = None, host: str="127.0.0.1", port: int=8088, udp_port: int=9000):
        """
        Initializes the WebSocket server.

        :param board_shim: MindRove BoardShim instance for inserting markers.
        :param host: Host address for the WebSocket server (default: "127.0.0.1").
        :param port: Port number for the WebSocket server (default: 8088).
        :param udp_port: Port number for the UDP loopback socket server (default: 9000). 
        """
        super().__init__()
        self.board_shim = board_shim
        self.host = host
        self.port = port
        self.loop = None
        self.server = None
        self.udp_port = udp_port
        self.is_running = True  # Flag to stop the server cleanly
        self.active_client = None  # Track the active WebSocket client
        self.udp_socket = None

    async def handler(self, websocket):
        """
        Handles incoming WebSocket connections and listens for messages.
        """
        print("✅ Client connected.")
        self.active_client = websocket
        try:
            async for message in websocket:
                print(f"📩 Received message: {message}")
                try:
                    marker_value = float(message)  # Ensure it's a float
                    if self.board_shim is not None:
                        self.board_shim.insert_marker(marker_value)
                        print(f"✅ Inserted marker: {marker_value}")
                    self.message_received.emit(marker_value)
                    await websocket.send(message)
                except ValueError:
                    print("⚠️ Invalid message format. Expected a float.")
        except websockets.exceptions.ConnectionClosed:
            print("❌ Client disconnected.")
        finally:
            self.active_client = None  # Reset active client

    # async def start_server(self):
    #     """
    #     Starts the WebSocket server asynchronously.
    #     Uses a lambda wrapper to pass `handler` correctly.
    #     """
    #     self.server = await websockets.serve(self.handler, self.host, self.port)
    #     print(f"🚀 WebSocket server started at ws://{self.host}:{self.port}")

    #     try:
    #         await self.server.wait_closed()  # Keep the server running
    #     except asyncio.CancelledError:
    #         pass  # Graceful exit on stop request

    async def start_server(self):
        """
        Starts the WebSocket and UDP listener asynchronously.
        """
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"🚀 WebSocket server started at ws://{self.host}:{self.port}")

        # Start UDP listener
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((self.host, self.udp_port))
        print(f"📡 UDP listener started on {self.host}:{self.udp_port}")

        while self.is_running:
            await self.listen_udp()

    async def listen_udp(self):
        """
        Listens for UDP messages and forwards them to the WebSocket client.
        """
        try:
            data, addr = self.udp_socket.recvfrom(1024)  # type: ignore # Receive UDP data
            message = data.decode('utf-8').strip()
            print(f"📩 Received UDP message from {addr}: {message}")
            await self.parse_message(message)
            # Forward to WebSocket client
            if self.active_client:
                await self.active_client.send(message)
                print(f"🔄 Forwarded UDP message to WebSocket client: {message}")

        except Exception as e:
            print(f"⚠️ UDP listener error: {e}")


    async def parse_message(self, message):
        try:
            # Parse JSON message
            json_data = json.loads(message)
            if json_data.get("name") == "state":
                marker_value = float(json_data.get("value", -1)) + 2048.0
                if self.board_shim is not None:
                    self.board_shim.insert_marker(marker_value)
                    print(f"✅ Inserted UDP state marker: {marker_value}")
                    self.message_received.emit(marker_value)
                else:
                    print(f"✅ Received UDP state marker (no board_shim): {marker_value}")
                    self.message_received.emit(marker_value)
            elif json_data.get("name") == "start":
                fname = json_data.get("value", "default.tsv")
                if self.board_shim is not None:
                    if self.board_shim.is_prepared():
                        self.board_shim.start_stream(f"file://{fname}:w")
                        print(f"✅ Started recording: {fname}")
                    else:
                        print(f"⚠️ Received command to start recording ({fname}) but board_shim session has not yet been prepared!")
                else:
                    print(f"✅ Received command to start recording (no board_shim): {fname}")

            elif json_data.get("name") == "stop":
                if self.board_shim is not None:
                    if self.board_shim.is_prepared():
                        self.board_shim.stop_stream()
                        print("✅ Stopped recording.")
                else:
                    print("✅ Received command to stop recording.")
                
        except Exception as e:
            print(f"⚠️ Parsing error: {e}")

    def run(self):
        """
        Runs the WebSocket server inside a separate event loop in QThread.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start_server())

    def stop(self):
        """
        Gracefully stops the WebSocket server and closes active connections.
        """
        if not self.is_running:
            return
        self.is_running = False

        # Close active client connection if it exists
        if self.active_client:
            asyncio.run_coroutine_threadsafe(self.active_client.close(), self.loop) # type: ignore

        if self.server:
            print("🛑 Stopping WebSocket server...")
            self.server.close()

        self.quit()  # Stop the QThread
        print("✅ WebSocket server stopped.")

    def __del__(self):
        """ Destructor ensures the server is closed properly. """
        self.stop()

if __name__ == "__main__":
    import time

    # Start the WebSocket server (for standalone testing)
    ws_server = WebSocketServer()
    ws_server.start()  # Start in a separate thread

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\n🔑 Keyboard Interrupt Received...")
        ws_server.stop()
