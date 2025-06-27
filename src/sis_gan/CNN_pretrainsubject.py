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

from sis_gan.models import EEGCNNSubject, weights_init
from sis_gan.utils import get_accuracy, load_data, save_model, load_config_yaml

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting seeds for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.default_rng(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class PretrainSubject:
    """Class for pretraining a subject classification model on EEG data using leave-one-out cross-validation."""

    def __init__(self, config_file: str) -> None:
        """
        Initialize the Pretrain_Subject class with configuration and data.

        Args:
            config_file (str): Path to the YAML configuration file.

        """
        self.input_data: np.ndarray = load_data()
        self.config = load_config_yaml(config_file)

    def _load_model(self) -> None:
        """Load the EEG subject classification model and initialize weights."""
        self.subject_predictor: nn.Module = EEGCNNSubject(self.config).to(device)
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
        save_model(self.subject_predictor, self.test_idx)

    def perform_loo(self) -> None:
        """Perform the leave one out analysis for each subject in the training dataset."""
        loo = LeaveOneOut()

        for train_idx, test_idx in loo.split(self.input_data):
            logger.info("Train indices: %s, Test index: %s", train_idx, test_idx)
            self.train_idx = train_idx
            self.test_idx = test_idx

            datainput = self.input_data[self.train_idx]
            train_subject = [np.zeros(datainput.shape[1]) + num_s for num_s in range(datainput.shape[0])]
            train_subject = np.array(train_subject).astype(np.int64)
            eeg_subject = np.concatenate(train_subject)
            eeg_data = np.concatenate(datainput)

            train_input = torch.from_numpy(eeg_data)
            train_label = torch.from_numpy(eeg_subject)

            self.trainloader: DataLoader = DataLoader(
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
        default="config/loo_pretrain_subject.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    trainer = PretrainSubject(config_file=args.config_file)
    trainer.perform_loo()
