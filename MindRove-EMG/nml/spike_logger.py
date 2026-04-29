import struct
from datetime import datetime
import numpy as np

class SpikeLogger:
    _open = False
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'wb')
        self._open = True
        print(f"Created SpikeLogger: {self.filename}")
    
    def __del__(self):
        if self._open:
            self.file.close()
    
    def write(self, data: "np.float32", n: int = 12, x: float = 0.0, y: float = 0.0, angle: float = 0.0, button_state: bool = False):
        """
        Write a timestamp with 100-microsecond precision and 8 float values to the binary file.
        :param data: A numpy array of n floats
        :param decode: scalar integer decode value
        "param 
        """
        # Get current timestamp with microsecond precision
        timestamp = datetime.now().timestamp()  # Timestamp in seconds with fractions
        packed_data = struct.pack(f"<d{n}ffffb", timestamp, *data, x, y, angle, button_state)  # Little-endian: 1 double (timestamp), 8 floats, 1 signed integer, 3 floats, 1 unsigned byte (boolean)
        self.file.write(packed_data)

    def write_batch(self, n: int, data_array: "np.ndarray[np.float32]", x: float = 0.0, y: float = 0.0, angle: float = 0.0, button_state: bool = False):
        """
        Write multiple rows of data to the binary file with the same timestamp, decode, and prompt values.
        :param data_array: 2D numpy array of shape (num_samples, 8), where each row has 8 floats
        :param batch: scalar integer indicating current processor batch-id
        :param sample: scalar integer indicating processor sample-counter value
        """
        # Get current timestamp with microsecond precision
        timestamp = datetime.now().timestamp()
        
        # Prepare the packed data for all rows in the batch
        for data in data_array:
            packed_data = struct.pack(f"<d{n}ffffb", timestamp, *data, x, y, angle, button_state)
            self.file.write(packed_data)
    
    def close(self):
        self.file.close()
        self._open = False