import time
import numpy as np
from enum import Enum
from pylsl import StreamInfo, StreamOutlet


class GestureCode(Enum):
    GRASP = 0
    RELEASE = 1
    WRIST_CLOCKWISE = 2
    WRIST_COUNTERCLOCKWISE = 3
    WRIST_OUT = 4
    WRIST_IN = 5
    LOWER = 7
    RAISE = 8
    WRIST_EXTEND = 9
    WRIST_FLEX = 10
    WRIST_PRO = 11
    WRIST_SUP = 12

class StreamManager:
    def __init__(self):
        name = "decoder"
        n_channels = 1
        srate = 20
        info = StreamInfo(name, 'EEG', n_channels, srate, 'float32', 'myuid34234')
        self.outlet = StreamOutlet(info)

    def send(self, gesture: GestureCode, duration: float = 1.0):
        """Sends the corresponding LSL code for the given gesture over the specified duration.

        Args:
            gesture (GestureCode): The gesture to send.
            duration (float): Duration in seconds for how long to send the gesture.
        """
        if not isinstance(gesture, GestureCode):
            raise ValueError("Invalid gesture. Must be of type GestureCode.")

        task_arr = np.ones(int(20 * duration)) * gesture.value
        for val in task_arr:
            print(f"Sending gesture {gesture.name} ({val})")
            self.outlet.push_sample([val])
            time.sleep(0.050)  # 50 ms sleep to match the 20 Hz sampling rate

    def send_single(self, gesture: GestureCode):
        """Sends a single instance of the decoded gesture.

        Args:
            gesture (GestureCode): The gesture to send.
        """
        if not isinstance(gesture, GestureCode):
            raise ValueError("Invalid gesture. Must be of type GestureCode.")

        print(f"Sending single gesture {gesture.name} ({gesture.value})")
        self.outlet.push_sample([gesture.value])
        # time.sleep(0.010)

# Example usage
if __name__ == "__main__":
    sm = StreamManager()
    
    # Send a single grasp command
    while True:
        sm.send_single(GestureCode.WRIST_CLOCKWISE)
        time.sleep(1.5)
        sm.send_single(GestureCode.WRIST_CLOCKWISE)
        time.sleep(1.5)

    # # # Randomly send a gesture every 2 seconds
    # while True:
    #     random_gesture = np.random.choice(list(GestureCode))
    #     sm.send_single(random_gesture)
    #     time.sleep(0.25)
