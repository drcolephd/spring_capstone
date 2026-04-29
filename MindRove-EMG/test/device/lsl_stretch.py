from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time

DURATION_FOR_SAMPLED_GESTURE = 3 # in seconds

def main():
    name = "decoder"
    n_channels = 1
    srate = 20
    info = StreamInfo(name, 'EEG', n_channels, srate, 'float32', 'myuid34234')
    outlet = StreamOutlet(info)
    while True:
        decoded_gesture = int(np.random.choice([0,1,2,3]))
        task_arr = np.ones(20*DURATION_FOR_SAMPLED_GESTURE) *decoded_gesture
        for i in range(len(task_arr)):
            print(task_arr[i])
            outlet.push_sample([task_arr[i]])
            time.sleep(0.050)

if __name__ == "__main__":
    main()