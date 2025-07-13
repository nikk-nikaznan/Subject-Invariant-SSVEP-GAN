import logging.config
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from scipy import signal

logger = logging.getLogger(__name__)


def setup_logging_from_config(config_path: Path = Path("config/logging_config.yaml")) -> None:
    """Load logging configuration from YAML file."""
    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f)

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


def load_config_yaml(config_file: str) -> dict[str, Any]:
    """Load a YAML file describing the training setup."""
    with Path(config_file).open() as f:
        return yaml.safe_load(f)


def load_data() -> np.ndarray:
    """
    Load and preprocess EEG data from .npy files for all subjects.

    Returns:
        np.ndarray: Preprocessed EEG data with shape (subjects, trials, channels, samples).

    """
    main_path = Path("sample_data/Real/")
    eeg_path = list(main_path.glob("S0*/"))

    input_data = []
    for f in eeg_path:
        eeg_files = sorted((f / "data").glob("*.npy"))
        eeg_data = [np.load(str(file)) for file in eeg_files]
        eeg_data = np.array(np.concatenate(eeg_data))
        eeg_data = data_process(eeg_data)
        input_data.append(eeg_data)

    return np.array(input_data).swapaxes(2, 3)


def load_label() -> np.ndarray:
    """
    Load EEG labels from .npy files for all subjects.

    Returns:
        np.ndarray: Array of labels for each subject.

    """
    main_path = Path("sample_data/Real/")
    eeg_path = list(main_path.glob("S0*/"))

    input_label = []
    for f in eeg_path:
        eeg_files = list((f / "label").glob("*.npy"))
        eeg_label = [np.load(str(file)) for file in eeg_files]
        eeg_label = np.concatenate(eeg_label).astype(np.int64)
        input_label.append(eeg_label)

    return np.array(input_label)


def get_accuracy(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """
    Calculate the accuracy between actual and predicted labels.

    Args:
        actual (torch.Tensor): Ground truth labels.
        predicted (torch.Tensor): Predicted labels.

    Returns:
        float: Accuracy value.

    """
    if actual.size(0) != predicted.size(0):
        error_msg = "actual and predicted must have the same batch size"
        raise ValueError(error_msg)

    return float(actual.eq(predicted).sum()) / actual.size(0)


def save_subject_model(
    subject_predictor: torch.nn.Module,
    test_idx: int,
) -> None:
    """
    Save the model and embeddings to a file.

    Args:
        subject_predictor (torch.nn.Module): The model to save.
        test_idx (int): Index of the test subject.

    """
    torch.save(subject_predictor, f"pretrain_subject_unseen{int(test_idx)}.pt")
    logger.info("Model Saved")


def data_process(input_data: np.ndarray) -> np.ndarray:
    """
    Preprocess EEG data by referencing, filtering, and normalizing.

    Args:
        input_data (np.ndarray): Raw EEG data.

    Returns:
        np.ndarray: Preprocessed EEG data.

    """
    dataref = input_data[:, :, 0]

    for i in range(input_data.shape[2]):
        input_data[:, :, i] = input_data[:, :, i] - dataref

    fs = 500
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    qf = 5.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized Frequency

    # Design notch filter
    b, a = signal.iirnotch(w0, qf)
    for i in range(len(input_data)):
        input_data[i, :, :] = signal.filtfilt(b, a, input_data[i, :, :], axis=0)

    b, a = signal.butter(4, 9.0 / (fs / 2.0), "highpass")
    for i in range(len(input_data)):
        input_data[i, :, :] = signal.filtfilt(b, a, input_data[i, :, :], axis=0)

    b, a = signal.butter(4, 60 / (fs / 2.0), "lowpass")
    for i in range(len(input_data)):
        input_data[i, :, :] = signal.filtfilt(b, a, input_data[i, :, :], axis=0)

    min_data = np.min(input_data)
    range_data = np.max(input_data) - min_data

    if range_data == 0:
        range_data = 1  # Avoid division by zero

    # Normalize data to the range [0, 1]
    return (input_data - min_data) / range_data
