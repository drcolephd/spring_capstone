import psutil
import socket
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams


def get_mindrove_ip():
    """
    Detects the local device's IP address when connected to a WiFi network in the 192.168.4.x range.

    Returns:
        str: The assigned IP address (e.g., '192.168.4.2') or None if not found.
    """
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4 address
                ip = addr.address
                if ip.startswith("192.168.4."):
                    print(f"✅ Found MindRove WiFi on {interface}: {ip}")
                    return ip
    print("⚠️ No active connection in 192.168.4.x range found.")
    return None  # No matching network found

if __name__ == "__main__":
    # Example usage:
    mindrove_ip = get_mindrove_ip()
    print(f"🌐 Assigned MindRove IP: {mindrove_ip}" if mindrove_ip else "❌ Not connected to MindRove WiFi")
    params = MindRoveInputParams()
    print(params.to_json())
