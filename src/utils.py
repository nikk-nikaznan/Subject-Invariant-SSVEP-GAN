import glob

import numpy as np
import torch
from scipy import signal


def load_data() -> np.ndarray:
    # data loading
    main_path = "sample_data/Real/"
    eeg_path = glob.glob(main_path + "S0*/")

    input_data = []
    for f in eeg_path:
        eeg_files = glob.glob(f + "data/*.npy")
        eeg_data = [np.load(f) for f in (eeg_files)]
        eeg_data = np.array(np.concatenate(eeg_data))
        eeg_data = data_process(eeg_data)
        input_data.append(eeg_data)

    input_data = np.array(input_data).swapaxes(2, 3)

    return input_data


def load_label() -> np.ndarray:
    # data loading
    main_path = "sample_data/Real/"
    eeg_path = glob.glob(main_path + "S0*/")

    input_label = []
    for f in eeg_path:
        eeg_files = glob.glob(f + "label/*.npy")
        eeg_label = [np.load(f) for f in (eeg_files)]
        eeg_label = np.asarray(np.concatenate(eeg_label))
        eeg_label = eeg_label.astype(np.int64)
        input_label.append(eeg_label)

    input_label = np.array(input_label)

    return input_label


def get_accuracy(actual, predicted) -> float:
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert actual.size(0) == predicted.size(0)
    return float(actual.eq(predicted).sum()) / actual.size(0)


def save_model(
    subject_predictor,
    test_idx,
) -> None:
    """Save the model and embeddings"""

    torch.save(subject_predictor, f"pretrain_subject_unseen{int(test_idx)}.pt")
    print("Model Saved")


def data_process(input_data):
    dataref = input_data[:, :, 0]
    # input_data = input_data[:, :, 6:8]

    for i in range(input_data.shape[2]):
        input_data[:, :, i] = input_data[:, :, i] - dataref

    fs = 500
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 5.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized Frequency

    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    for i in range(0, len(input_data)):
        # apply along the zeroeth dimension
        input_data[i, :, :] = signal.filtfilt(b, a, input_data[i, :, :], axis=0)

    b, a = signal.butter(4, 9.0 / (fs / 2.0), "highpass")
    for i in range(0, len(input_data)):
        # apply along the zeroeth dimension
        input_data[i, :, :] = signal.filtfilt(b, a, input_data[i, :, :], axis=0)

    b, a = signal.butter(4, 60 / (fs / 2.0), "lowpass")
    for i in range(0, len(input_data)):
        # apply along the zeroeth dimension
        input_data[i, :, :] = signal.filtfilt(b, a, input_data[i, :, :], axis=0)

    min_data = np.min(input_data)
    range_data = np.max(input_data) - min_data
    # 0 to 1
    input_data = (input_data - min_data) / range_data

    return input_data
