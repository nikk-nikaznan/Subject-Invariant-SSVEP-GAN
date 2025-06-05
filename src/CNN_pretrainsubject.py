import argparse
import random

import numpy as np
import torch
import yaml  # type: ignore
from sklearn.model_selection import LeaveOneOut
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import EEG_CNN_Subject, weights_init
from utils import get_accuracy, load_data, save_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting seeds for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class Pretrain_Subject:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.input_data = load_data()
        self.load_config_yaml()

    def load_config_yaml(self) -> None:
        """Load a YAML file describing the training setup"""
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

    def _load_model(self) -> None:
        """Load the EEG subject classification model"""
        # Build the subject classification model and initalise weights
        self.subject_predictor = EEG_CNN_Subject(self.config).to(device)
        self.subject_predictor.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create the training objects"""
        # Loss and Optimizer
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer_Pred = torch.optim.Adam(
            self.subject_predictor.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

    def _train_model(self) -> None:
        """Train a model using the provided configuration"""
        # loop through the required number of epochs
        for epoch in range(self.config["num_epochs"]):
            print("Epoch:", epoch)
            cumulative_accuracy = 0.0

            # loop over all of the batches
            for i, data in enumerate(self.trainloader, 0):
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
        print("Training Accuracy: %2.1f" % (cumulative_accuracy / len(self.trainloader) * 100))
        save_model(self.subject_predictor, self.test_idx)

    def perform_loo(self) -> None:
        """Perform the leave one out analysis for each subject in the training dataset"""
        loo = LeaveOneOut()

        for self.train_idx, self.test_idx in loo.split(self.input_data):
            print(self.train_idx, self.test_idx)

            datainput = self.input_data[self.train_idx]
            train_subject = []
            for num_s in range(datainput.shape[0]):
                train_subject.append(np.zeros(datainput.shape[1]) + num_s)

            train_subject = np.array(train_subject).astype(np.int64)
            EEGsubject = np.concatenate(train_subject)
            EEGdata = np.concatenate(datainput)

            # convert NumPy Array to Torch Tensor
            train_input = torch.from_numpy(EEGdata)
            train_label = torch.from_numpy(EEGsubject)

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
        default="config/loo_pretrain_subject.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    trainer = Pretrain_Subject(config_file=args.config_file)
    trainer.perform_loo()
