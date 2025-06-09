import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from sklearn.model_selection import LeaveOneOut
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import EEGCNNSSVEP, weights_init
from utils import get_accuracy, load_data, load_label

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting seeds for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.default_rng(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class SSVEPClass:
    """Class for training SSVEP classification model on EEG data."""

    def __init__(self, config_file: str) -> None:
        """
        Initialize the SSVEP class with configuration and data.

        Args:
            config_file (str): Path to the YAML configuration file.

        """
        self.config_file = config_file
        self.input_data = load_data()
        self.input_label = load_label()
        self.load_config_yaml()

    def load_config_yaml(self) -> None:
        """Load a YAML file describing the training setup."""
        with Path(self.config_file).open() as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

    def _load_model(self) -> None:
        """Load the EEG subject classification model."""
        # Build the subject classification model and initalise weights
        self.subject_predictor = EEGCNNSSVEP(self.config).to(device)
        self.subject_predictor.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create the training objects."""
        # Loss and Optimizer
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer_Pred = torch.optim.Adam(
            self.subject_predictor.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

    def _train_model(self) -> None:
        """Train a model using the provided configuration."""
        # loop through the required number of epochs
        for epoch in range(self.config["num_epochs"]):
            logger.info("Epoch : %d", epoch)
            cumulative_accuracy = 0.0

            # loop over all of the batches
            for _, data in enumerate(self.trainloader, 0):
                # format the data from the dataloader
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                # Forward + Backward + Optimize
                self.optimizer_Pred.zero_grad()
                outputs = self.subject_predictor(inputs)
                loss = self.ce_loss(outputs, labels)
                loss.backward()
                self.optimizer_Pred.step()

                # calculate the accuracy over the training batch
                _, predicted = torch.max(outputs, 1)

                cumulative_accuracy += get_accuracy(labels, predicted)
        logger.info("Training accuracy: %2.1f", cumulative_accuracy / len(self.trainloader) * 100)

    def perform_loo(self) -> None:
        """Perform the leave one out analysis for each subject in the training dataset."""
        loo = LeaveOneOut()

        for train_idx, test_idx in loo.split(self.input_data):
            logger.info("Train indices: %s, Test index: %s", train_idx, test_idx)
            self.train_idx = train_idx
            self.test_idx = test_idx

            datainput = self.input_data[self.train_idx]
            labelinput = self.input_label[self.train_idx]

            eeg_data = np.concatenate(datainput)
            eeg_label = np.concatenate(labelinput)

            # convert NumPy Array to Torch Tensor
            train_input = torch.from_numpy(eeg_data)
            train_label = torch.from_numpy(eeg_label)

            # create the data loader for the training set
            self.trainloader = DataLoader(
                dataset=TensorDataset(train_input, train_label),
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0,
            )

            self._load_model()
            self._build_training_objects()
            self._train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/loo_SSVEP_class.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    trainer = SSVEPClass(config_file=args.config_file)
    trainer.perform_loo()
