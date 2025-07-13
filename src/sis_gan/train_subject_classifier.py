import argparse
import logging
import random
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sis_gan.models import EEGCNNSubject, weights_init
from sis_gan.utils import (
    get_accuracy,
    load_config_yaml,
    load_data,
    load_label,
    save_subject_model,
    setup_logging_from_config,
)

setup_logging_from_config()
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting seeds for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.default_rng(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class SubjectClassifier:
    """
    Class for training subject classification models on EEG data.

    Supports both stratified train/test split and leave-one-out cross-validation.
    """

    def __init__(
        self,
        config_file: str,
        validation_strategy: Literal["stratified", "loo"] = "stratified",
    ) -> None:
        """
        Initialize the SubjectClassifier with configuration and data.

        Args:
            config_file (str): Path to the YAML configuration file.
            validation_strategy (Literal["stratified", "loo"]): Validation strategy to use.

        """
        self.validation_strategy = validation_strategy
        self.input_data: np.ndarray = load_data()
        self.config = load_config_yaml(config_file)

        # Only load labels for stratified split (not needed for LOO)
        if validation_strategy == "stratified":
            self.input_label: np.ndarray = load_label()

    def _load_model(self) -> None:
        """Load the EEG subject classification model and initialize weights."""
        self.subject_predictor: nn.Module = EEGCNNSubject(self.config).to(device)
        self.subject_predictor.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create the loss function and optimizer for training."""
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer_pred = torch.optim.Adam(
            self.subject_predictor.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

    def _train_model(self, *, save_model: bool = True, test_idx: int = 1) -> float:
        """
        Train the subject classification model using the provided configuration.

        Args:
            save_model (bool): Whether to save the trained model.
            test_idx (int): Index for saving the model (used in LOO).

        Returns:
            float: Training accuracy.

        """
        total_accuracy = 0.0

        for epoch in range(self.config["num_epochs"]):
            logger.info("Starting training epoch %d", epoch)
            cumulative_accuracy = 0.0

            for _, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                self.optimizer_pred.zero_grad()
                outputs = self.subject_predictor(inputs)
                loss = self.ce_loss(outputs, labels)
                loss.backward()
                self.optimizer_pred.step()

                _, predicted = torch.max(outputs, 1)
                cumulative_accuracy += get_accuracy(labels, predicted)

            epoch_accuracy = cumulative_accuracy / len(self.trainloader) * 100
            total_accuracy = epoch_accuracy  # Keep last epoch accuracy

        logger.info("Training accuracy: %.1f%%", total_accuracy)

        if save_model:
            save_subject_model(self.subject_predictor, test_idx)

        return total_accuracy

    def _eval_model(self) -> float:
        """
        Evaluate the trained model on the test dataset.

        Returns:
            float: Test accuracy.

        """
        self.subject_predictor.eval()
        test_cumulative_accuracy = 0.0

        with torch.no_grad():
            for _, data in enumerate(self.testloader, 0):
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_inputs = test_inputs.float()

                test_outputs = self.subject_predictor(test_inputs)
                _, test_predicted = torch.max(test_outputs, 1)
                test_acc = get_accuracy(test_labels, test_predicted)
                test_cumulative_accuracy += test_acc

        test_accuracy = test_cumulative_accuracy / len(self.testloader) * 100
        logger.info("Test accuracy: %.1f%%", test_accuracy)
        return test_accuracy

    def _perform_stratified_split(self) -> tuple[float, float]:
        """
        Perform stratified train/test split and train the model.

        Returns:
            tuple[float, float]: Training and test accuracies.

        """
        logger.info("Using stratified train/test split validation")

        # Create subject labels for stratification
        num_subjects = self.input_data.shape[0]
        train_subject = [np.zeros(self.input_data.shape[1]) + num_s for num_s in range(num_subjects)]
        train_subject = np.array(train_subject).astype(np.int64)
        eeg_subject = np.concatenate(train_subject)
        eeg_data = np.concatenate(self.input_data)

        # Update config to match actual number of subjects
        self.config["num_subjects"] = num_subjects
        logger.info("Number of subjects in data: %d", num_subjects)
        logger.info("Model configured for %d subjects", self.config["num_subjects"])

        # Perform stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_n)
        train_idx, test_idx = next(sss.split(eeg_data, eeg_subject))

        # Prepare data loaders
        x_train, y_train = eeg_data[train_idx], eeg_subject[train_idx]
        x_test, y_test = eeg_data[test_idx], eeg_subject[test_idx]

        train_input = torch.from_numpy(x_train)
        train_label = torch.from_numpy(y_train)
        test_input = torch.from_numpy(x_test)
        test_label = torch.from_numpy(y_test)

        self.trainloader = DataLoader(
            dataset=TensorDataset(train_input, train_label),
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        self.testloader = DataLoader(
            dataset=TensorDataset(test_input, test_label),
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
        )

        # Train and evaluate
        self._load_model()
        self._build_training_objects()
        train_accuracy = self._train_model(save_model=True, test_idx=1)
        test_accuracy = self._eval_model()

        return train_accuracy, test_accuracy

    def _perform_loo(self) -> list[float]:
        """
        Perform leave-one-out cross-validation and train models.

        Returns:
            list[float]: Training accuracies for each fold.

        """
        logger.info("Using leave-one-out cross-validation")

        loo = LeaveOneOut()
        accuracies = []

        for train_idx, test_idx in loo.split(self.input_data):
            test_idx_int = test_idx[0]  # Extract single integer from array
            logger.info("Training on subjects %s, testing on subject %d", train_idx, test_idx_int)

            # Prepare training data
            datainput = self.input_data[train_idx]
            train_subject = [np.zeros(datainput.shape[1]) + num_s for num_s in range(datainput.shape[0])]
            train_subject = np.array(train_subject).astype(np.int64)
            eeg_subject = np.concatenate(train_subject)
            eeg_data = np.concatenate(datainput)

            train_input = torch.from_numpy(eeg_data)
            train_label = torch.from_numpy(eeg_subject)

            self.trainloader = DataLoader(
                dataset=TensorDataset(train_input, train_label),
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0,
            )

            # Train model for this fold
            self._load_model()
            self._build_training_objects()
            train_accuracy = self._train_model(save_model=True, test_idx=test_idx_int)
            accuracies.append(train_accuracy)

        return accuracies

    def train(self) -> tuple[float, ...] | list[float]:
        """
        Train subject classifier using the specified validation strategy.

        Returns:
            Union[tuple[float, ...], list[float]]: Accuracies based on validation strategy.

        """
        if self.validation_strategy == "stratified":
            return self._perform_stratified_split()
        if self.validation_strategy == "loo":
            return self._perform_loo()
        error_msg = f"Unknown validation strategy: {self.validation_strategy}"
        raise ValueError(error_msg)


def main() -> None:
    """Run the subject classifier training."""
    parser = argparse.ArgumentParser(description="Train subject classification model")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/subject_classification.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--validation_strategy",
        type=str,
        choices=["stratified", "loo"],
        default="stratified",
        help="Validation strategy: 'stratified' for train/test split, 'loo' for leave-one-out",
    )
    args = parser.parse_args()

    # Train the classifier
    classifier = SubjectClassifier(
        config_file=args.config_file,
        validation_strategy=args.validation_strategy,
    )

    results = classifier.train()

    # Log final results
    if args.validation_strategy == "stratified":
        train_acc, test_acc = results
        logger.info("Final results - Training: %.1f%%, Test: %.1f%%", train_acc, test_acc)
    else:  # loo
        mean_acc = np.mean(results)
        std_acc = np.std(results)
        logger.info("LOO results - Mean: %.1f%% ± %.1f%%", mean_acc, std_acc)


if __name__ == "__main__":
    main()
