import numpy as np
from scipy.signal import lfilter_zi, lfilter

class RingBuffer:
    def __init__(self, capacity, num_channels=1):
        self.capacity = capacity
        self.num_channels = num_channels
        self.buffer = np.zeros((capacity, num_channels), dtype=np.float64)
        self.start = 0
        self.end = 0
        self.full = False

    def append(self, data):
        data = np.atleast_2d(data)
        num_new_samples = data.shape[0]

        if num_new_samples > self.capacity:
            data = data[-self.capacity:]
            num_new_samples = self.capacity

        end_index = (self.end + num_new_samples) % self.capacity

        if end_index < self.end or (self.full and end_index == self.start):
            wrap_point = self.capacity - self.end
            self.buffer[self.end:] = data[:wrap_point]
            self.buffer[:end_index] = data[wrap_point:]
            self.start = (self.start + num_new_samples) % self.capacity
        else:
            self.buffer[self.end:end_index] = data

        self.end = end_index
        if self.end == self.start:
            self.full = True

    def get_last_n_samples(self, n):
        if n > self.capacity:
            raise ValueError("Requested samples exceed buffer capacity")
        
        if self.full or (self.end - n < 0):
            indices = np.concatenate((np.arange(self.end - n, self.capacity), np.arange(0, self.end)))
            return self.buffer.take(indices, axis=0, mode='wrap')
        else:
            return self.buffer[self.end - n:self.end]

class IIRFilterWithState:
    def __init__(self, b, a, num_channels):
        self.b = b
        self.a = a
        self.num_channels = num_channels
        self.z = np.array([lfilter_zi(b, a) for _ in range(num_channels)])

    def apply(self, new_sample):
        filtered_sample = np.zeros_like(new_sample)
        for i in range(self.num_channels):
            filtered_sample[i], self.z[i] = lfilter(self.b, self.a, [new_sample[i]], zi=self.z[i])
        return filtered_sample