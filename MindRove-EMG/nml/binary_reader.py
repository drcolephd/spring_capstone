import struct
import numpy as np
import csv
from datetime import datetime

class BinaryReader:
    _open = False
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'rb')
        self._open = True
        self.sample_size = struct.calcsize('<d8fiifffb')
    
    def __del__(self):
        if self._open:
            self.file.close()

    def convert(self):
        """
        Converts the binary file to a CSV with named columns.
        """
        # Generate the new filename by replacing '.bin' with '.csv'
        csv_filename = self.filename.replace('.bin', '.csv')
        
        # Open the CSV file for writing
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write the header row
            header = ['date', 'time'] + [f'feature_{i+1}' for i in range(8)] + ['decode', 'prompt', 'x', 'y', 'angle', 'button_1']
            writer.writerow(header)
            
            # Read through each sample in the binary file
            self.file.seek(0)  # Start from the beginning
            while True:
                sample = self.read_next()
                if sample is None:
                    break  # End of file reached
                # print(sample)
                # Separate timestamp and feature data
                timestamp = sample[0]
                features = sample[1:9]
                decode = sample[9]
                prompt = sample[10]
                x = sample[11]
                y = sample[12]
                angle = sample[13]
                button = sample[14]
                
                # Convert timestamp to date and time with required format
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime('%Y-%m-%d')
                time_str = dt.strftime('%H:%M:%S.%f')[:-2]  # Remove last 2 digits to retain 4 decimal places
                
                # Write the row to the CSV file
                writer.writerow([date_str, time_str] + list(features) + [decode, prompt, x, y, angle, button])
        self.file.seek(0)
        print(f"Data successfully converted to {csv_filename}")

    def read_all(self):
        """
        Reads all data from the binary file and returns it as a numpy array.
        :return: Nx9 numpy array where each row contains a timestamp and 8 float values
        """
        self.file.seek(0)  # Ensure starting at the beginning
        data = []
        
        while True:
            raw_data = self.file.read(self.sample_size)
            if not raw_data:
                break
            unpacked_data = struct.unpack('<d8fiifffb', raw_data)
            data.append(unpacked_data)
        self.file.seek(0)
        return np.array(data)  # Return as a numpy array for easy manipulation
    
    def read_next(self):
        """
        Reads the next sample from the binary file.
        :return: A tuple containing (timestamp, 8 float values, integer, integer, float, float, float, byte) or None if EOF
        """
        raw_data = self.file.read(self.sample_size)
        if not raw_data:
            return None  # End of file reached
        
        unpacked_data = struct.unpack('<d8fiifffb', raw_data)
        return unpacked_data
    
    def close(self):
        self.file.close()
        self._open = False