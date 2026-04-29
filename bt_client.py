import socket
import time

PI_BT_ADDRESS = "88:A2:9E:AF:19:38"   # replace with your Pi's Bluetooth MAC
PORT = 1
EXPECTED_MESSAGE = "CONNECTED_OK"

def connect_and_receive():
    sock = socket.socket(
        socket.AF_BLUETOOTH,
        socket.SOCK_STREAM,
        socket.BTPROTO_RFCOMM
    )

    try:
        print(f"Connecting to {PI_BT_ADDRESS} on RFCOMM channel {PORT}...")
        sock.connect((PI_BT_ADDRESS, PORT))

        data = sock.recv(1024).decode("utf-8").strip()
        print(f"Received: {data}")

        if data == EXPECTED_MESSAGE:
            print("Success: received confirmation from Pi.")
            return True
        else:
            print("Connected, but message did not match expected value.")
            return False

    except Exception as e:
        print(f"Connection failed: {e}")
        return False

    finally:
        sock.close()

if __name__ == "__main__":
    connect_and_receive()