import argparse
import logging
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sis_gan.models import EEGCNNSubject, weights_init
from sis_gan.utils import get_accuracy, load_config_yaml, load_data, load_label, save_model

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting seeds for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.default_rng(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class SubjectClass:
    """Class for training and evaluating a subject classification model on EEG data."""

    def __init__(self, config_file: str) -> None:
        """
        Initialize the Subject_Class with configuration and data.

        Args:
            config_file (str): Path to the YAML configuration file.

        """
        self.input_data: np.ndarray = load_data()
        self.input_label: np.ndarray = load_label()
        self.config = load_config_yaml(config_file)

    def _load_model(self) -> None:
        """Load the EEG subject classification model and initialize weights."""
        self.subject_predictor: nn.Module = EEGCNNSubject(self.config).to(device)
        self.subject_predictor.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create the loss function and optimizer for training."""
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer_Pred = torch.optim.Adam(
            self.subject_predictor.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

    def _train_model(self) -> None:
        """Train the subject classification model using the provided configuration."""
        for epoch in range(self.config["num_epochs"]):
            logger.info("Starting training epoch %d", epoch)
            cumulative_accuracy = 0.0

            for _, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                self.optimizer_Pred.zero_grad()
                outputs = self.subject_predictor(inputs)
                loss = self.ce_loss(outputs, labels)
                loss.backward()
                self.optimizer_Pred.step()

                _, predicted = torch.max(outputs, 1)
                cumulative_accuracy += get_accuracy(labels, predicted)
        logger.info("Training accuracy: %2.1f", cumulative_accuracy / len(self.trainloader) * 100)

    def _test_model(self) -> None:
        """Evaluate the trained model on the test dataset and print accuracy."""
        self.subject_predictor.eval()
        test_cumulative_accuracy = 0.0
        for _, data in enumerate(self.testloader, 0):
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_inputs = test_inputs.float()

            test_outputs = self.subject_predictor(test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)

            test_acc = get_accuracy(test_labels, test_predicted)
            test_cumulative_accuracy += test_acc

        logger.info("Test Accuracy: %2.1f", test_cumulative_accuracy / len(self.testloader) * 100)

    def perform_kfold(self) -> None:
        """Prepare data and perform a single stratified shuffle split for training and testing."""
        train_subject = [np.zeros(self.input_data.shape[1]) + num_s for num_s in range(self.input_data.shape[0])]

        train_subject = np.array(train_subject).astype(np.int64)
        eeg_subject = np.concatenate(train_subject)
        eeg_data = np.concatenate(self.input_data)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        sss.get_n_splits(eeg_data, eeg_subject)

        for train_idx, test_idx in sss.split(eeg_data, eeg_subject):
            x_train = eeg_data[train_idx]
            y_train = eeg_subject[train_idx]
            x_test = eeg_data[test_idx]
            y_test = eeg_subject[test_idx]

            train_input = torch.from_numpy(x_train)
            train_label = torch.from_numpy(y_train)
            test_input = torch.from_numpy(x_test)
            test_label = torch.from_numpy(y_test)

            self.trainloader: DataLoader = DataLoader(
                dataset=TensorDataset(train_input, train_label),
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0,
            )
            self.testloader: DataLoader = DataLoader(
                dataset=TensorDataset(test_input, test_label),
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0,
            )

            self._load_model()
            self._build_training_objects()
            self._train_model()
            self._test_model()
            save_model(self.subject_predictor, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/loo_pretrain_subject.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    trainer = SubjectClass(config_file=args.config_file)
    trainer.perform_kfold()
