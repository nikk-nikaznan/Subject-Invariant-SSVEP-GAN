import numpy as np
from scipy import signal


def data_process(input_data):

    dataref = input_data[:, :, 0]
    input_data = input_data[:, :, 6:8]

    for i in range(input_data.shape[2]):
        input_data[:, :, i] = input_data[:, :, i] - dataref

    fs = 500
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 5.0  # Quality factor
    w0 = f0/(fs/2)  # Normalized Frequency

    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    for i in range(0, len(input_data)):
        # apply along the zeroeth dimension
        input_data[i, :, :] = signal.filtfilt(
            b, a, input_data[i, :, :], axis=0)

    b, a = signal.butter(4, 9.0/(fs / 2.0), 'highpass')
    for i in range(0, len(input_data)):
        # apply along the zeroeth dimension
        input_data[i, :, :] = signal.filtfilt(
            b, a, input_data[i, :, :], axis=0)

    b, a = signal.butter(4, 60/(fs / 2.0), 'lowpass')
    for i in range(0, len(input_data)):
        # apply along the zeroeth dimension
        input_data[i, :, :] = signal.filtfilt(
            b, a, input_data[i, :, :], axis=0)

    min_data = np.min(input_data)
    range_data = np.max(input_data)-min_data
    # 0 to 1
    input_data = (input_data-min_data)/range_data

    return input_data
