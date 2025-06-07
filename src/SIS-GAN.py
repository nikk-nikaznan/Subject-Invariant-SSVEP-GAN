import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from sklearn.model_selection import LeaveOneOut
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import EEG_CNN_Discriminator, EEG_CNN_Generator, weights_init
from utils import load_data, load_label

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SISGAN:
    """Class for training and generating synthetic EEG data using a subject-invariant GAN."""

    def __init__(self, config_file: str) -> None:
        """
        Initialize SISGAN with configuration file and data.

        Args:
            config_file (str): Path to the YAML configuration file.

        """
        self.config_file = config_file
        self.input_data: np.ndarray = load_data()
        self.input_label: np.ndarray = load_label()
        self.load_config_yaml()
        self.rng = np.random.default_rng(42)  # For reproducibility

    def load_config_yaml(self) -> None:
        """Load a YAML file describing the training setup."""
        with Path(self.config_file).open() as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

    def _load_pretrain_model(self) -> None:
        """Load the pretrain subject classification model."""
        self.subject_predictor: torch.nn.Module = torch.load(
            f"pretrain_subject_unseen{self.test_idx}.pt",
            map_location=torch.device("cpu"),
        )

    def _load_model(self) -> None:
        """Load and initialize the GAN generator and discriminator models."""
        self.generator: torch.nn.Module = EEG_CNN_Generator(self.config).to(device)
        self.generator.apply(weights_init)
        self.discriminator: torch.nn.Module = EEG_CNN_Discriminator(self.config).to(device)
        self.discriminator.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create loss functions and optimizers for training."""
        self.ce_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        self.optimizer_Gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )
        self.optimizer_Dis = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

    def _gen_noise(self) -> None:
        """Generate random noise and set the last elements to a random selection of one-hot labels."""
        gen_noise_: np.ndarray = self.rng.normal(
            0,
            1,
            (self.config["batch_size"], self.config["nz"]),
        )
        self.gen_label: np.ndarray = self.rng.integers(
            0,
            self.config["num_aux_class"],
            self.config["batch_size"],
        )
        gen_onehot = np.zeros((self.config["batch_size"], self.config["num_aux_class"]))
        gen_onehot[np.arange(self.config["batch_size"]), self.gen_label] = 1
        gen_noise_[
            np.arange(self.config["batch_size"]),
            : self.config["num_aux_class"],
        ] = gen_onehot[np.arange(self.config["batch_size"])]
        gen_noise = torch.from_numpy(gen_noise_)
        gen_noise.data.copy_(gen_noise.view(self.config["batch_size"], self.config["nz"]))
        self.z = gen_noise.to(device).float()

    def _train_model(self) -> None:
        """Train GAN using the provided configuration."""
        for epoch in range(self.config["num_epochs"]):
            logger.info("Epoch: %d", epoch)
            for _, data in enumerate(self.trainloader, 0):
                real_data, real_aux_label = data
                real_data, real_aux_label = (
                    real_data.to(device),
                    real_aux_label.to(device),
                )
                real_data = real_data.float()
                real_aux_label = real_aux_label.long()
                input_size = real_data.size(0)

                # Configure input
                input_data = real_data.data
                dis_label = torch.empty(input_size, 1).to(device)
                self._gen_noise()

                # Train Discriminator
                self.optimizer_Dis.zero_grad()
                dis_label.data.fill_(1)
                aux_label = real_aux_label.data

                dis_output, aux_output = self.discriminator(input_data)
                dis_err_real = self.adversarial_loss(dis_output, dis_label)
                aux_err_real = self.ce_loss(aux_output, aux_label)
                err_dis_real = dis_err_real + aux_err_real
                err_dis_real.backward()

                # Train with fake
                fake_data = self.generator(self.z)
                dis_label.data.fill_(0)
                aux_label.data.resize_(input_size).copy_(torch.from_numpy(self.gen_label))
                dis_output, aux_output = self.discriminator(fake_data.detach())
                dis_err_fake = self.adversarial_loss(dis_output, dis_label)
                aux_err_fake = self.ce_loss(aux_output, aux_label)
                err_dis_fake = dis_err_fake + aux_err_fake
                err_dis_fake.backward()
                err_dis = err_dis_real + err_dis_fake
                self.optimizer_Dis.step()

                # Train Generator
                self.optimizer_Gen.zero_grad()
                dis_label.data.fill_(1)
                fake_data = self.generator(self.z)
                dis_output, aux_output = self.discriminator(fake_data)
                dis_err_gen = self.adversarial_loss(dis_output, dis_label)
                aux_err_gen = self.ce_loss(aux_output, aux_label)
                subject_output = self.subject_predictor(fake_data)
                sub_err_gen, _ = torch.max(subject_output, 1)
                sub_err_gen = self.config["lmbd"] * sub_err_gen.mean()
                err_gen = dis_err_gen + aux_err_gen + sub_err_gen
                err_gen.backward()
                self.optimizer_Gen.step()

            logger.info(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]",
                epoch,
                self.config["num_epochs"],
                err_dis.item(),
                err_gen.item(),
            )

    def _generate_data(self) -> None:
        """Generate synthetic EEG signals using the trained generator and save them."""
        new_data = []
        new_label = []
        self.generator.eval()

        with torch.no_grad():
            for nclass in range(self.config["num_aux_class"]):
                for _ in range(self.config["num_epochs"]):
                    eval_noise_ = self.rng.normal(
                        0,
                        1,
                        (self.config["batch_size"], self.config["nz"]),
                    )
                    eval_label = np.zeros((self.config["batch_size"],), dtype=int) + nclass
                    eval_onehot = np.zeros(
                        (self.config["batch_size"], self.config["num_aux_class"]),
                    )
                    eval_onehot[np.arange(self.config["batch_size"]), eval_label] = 1
                    eval_noise_[
                        np.arange(self.config["batch_size"]),
                        : self.config["num_aux_class"],
                    ] = eval_onehot[np.arange(self.config["batch_size"])]
                    eval_noise = torch.from_numpy(eval_noise_)
                    eval_noise.data.copy_(
                        eval_noise.view(self.config["batch_size"], self.config["nz"]),
                    )
                    z = eval_noise.to(device).float()
                    fake_data = self.generator(z)
                    new_data.append(fake_data.data.cpu().numpy())
                    new_label.append(eval_label)

            new_data = np.asarray(new_data)
            new_data = np.concatenate(new_data)
            new_label = np.asarray(new_label)
            new_label = np.concatenate(new_label)
            np.save(f"SISGAN_unseen{self.test_idx}.npy", new_data)
            np.save(f"SISGAN_unseen{self.test_idx}_labels.npy", new_label)
        logger.info("Generated data saved")

    def perform_loo(self) -> None:
        """Perform the leave-one-out analysis for each subject in the training dataset."""
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(self.input_data):
            logger.info("Training on subject %d, testing on subject %d", train_idx, test_idx)
            self.train_idx = train_idx
            self.test_idx = test_idx

            datainput = self.input_data[self.train_idx]
            labelinput = self.input_label[self.train_idx]
            eeg_data = np.concatenate(datainput)
            eeg_label = np.concatenate(labelinput)
            train_input = torch.from_numpy(eeg_data)
            train_label = torch.from_numpy(eeg_label)
            self.trainloader = DataLoader(
                dataset=TensorDataset(train_input, train_label),
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0,
            )
            self._load_pretrain_model()
            self._load_model()
            self._build_training_objects()
            self._train_model()
            self._generate_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/loo_SISGAN.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    trainer = SISGAN(config_file=args.config_file)
    trainer.perform_loo()
