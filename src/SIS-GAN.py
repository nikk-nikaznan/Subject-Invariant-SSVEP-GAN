import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader, TensorDataset

from models import EEG_CNN_Discriminator, EEG_CNN_Generator, weights_init
from utils import load_data, load_label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting seeds for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class SIS_GAN:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.input_data = load_data()
        self.input_label = load_label()
        self.load_config_yaml()

    def load_config_yaml(self) -> None:
        """Load a YAML file describing the training setup"""

        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def _load_pretrain_model(self) -> None:
        """Load the pretrain subject classification model"""

        # Load the pretrain subject predictor
        self.subject_predictor = torch.load(
            "pretrain_subject_unseen%i.pt" % (self.test_idx),
            map_location=torch.device("cpu"),
        )

    def _load_model(self) -> None:
        """Load the GAN model"""

        # Build the generator model and initalise weights
        self.generator = EEG_CNN_Generator(self.config).to(device)
        self.generator.apply(weights_init)
        # Build the discriminator model and initalise weights
        self.discriminator = EEG_CNN_Discriminator(self.config).to(device)
        self.discriminator.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create the training objects"""

        # Loss and Optimizer
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
        """Generate random noise and set the last elements to a random selection of one hot labels"""

        gen_noise_ = np.random.normal(
            0, 1, (self.config["batch_size"], self.config["nz"])
        )
        self.gen_label = np.random.randint(
            0, self.config["num_aux_class"], self.config["batch_size"]
        )
        gen_onehot = np.zeros((self.config["batch_size"], self.config["num_aux_class"]))
        gen_onehot[np.arange(self.config["batch_size"]), self.gen_label] = 1
        gen_noise_[
            np.arange(self.config["batch_size"]), : self.config["num_aux_class"]
        ] = gen_onehot[np.arange(self.config["batch_size"])]
        gen_noise = torch.from_numpy(gen_noise_)
        gen_noise.data.copy_(
            gen_noise.view(self.config["batch_size"], self.config["nz"])
        )
        self.z = gen_noise.to(device).float()

    def _train_model(self) -> None:
        """Train GAN using the provided configuration"""

        # loop through the required number of epochs
        for epoch in range(self.config["num_epochs"]):
            print("Epoch:", epoch)

            # loop over all of the batches
            for i, data in enumerate(self.trainloader, 0):
                # format the data from the dataloader
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
                dis_label = torch.empty(input_size, 1).to(device)  # Discriminator label
                # Intialise noise
                self._gen_noise()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # train with real
                self.optimizer_Dis.zero_grad()

                dis_label.data.fill_(1)
                aux_label = real_aux_label.data

                dis_output, aux_output = self.discriminator(input_data)
                dis_errD_real = self.adversarial_loss(dis_output, dis_label)
                aux_errD_real = self.ce_loss(aux_output, aux_label)

                errD_real = dis_errD_real + aux_errD_real
                errD_real.backward()

                # train with fake
                fake_data = self.generator(self.z)

                dis_label.data.fill_(0)
                aux_label.data.resize_(input_size).copy_(
                    torch.from_numpy(self.gen_label)
                )

                dis_output, aux_output = self.discriminator(fake_data.detach())
                dis_errD_fake = self.adversarial_loss(dis_output, dis_label)
                aux_errD_fake = self.ce_loss(aux_output, aux_label)

                errD_fake = dis_errD_fake + aux_errD_fake
                errD_fake.backward()

                errD = errD_real + errD_fake
                self.optimizer_Dis.step()

                # -----------------
                #  Train Generator`
                # -----------------`

                self.optimizer_Gen.zero_grad()

                dis_label.data.fill_(1)
                fake_data = self.generator(self.z)

                dis_output, aux_output = self.discriminator(fake_data)
                dis_errG = self.adversarial_loss(dis_output, dis_label)
                aux_errG = self.ce_loss(aux_output, aux_label)

                subject_output = self.subject_predictor(fake_data)
                sub_errG, _ = torch.max(subject_output, 1)

                sub_errG = self.config["lmbd"] * sub_errG.mean()

                errG = dis_errG + aux_errG + sub_errG
                errG.backward()
                self.optimizer_Gen.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] "
                % (
                    epoch,
                    self.config["num_epochs"],
                    i,
                    len(
                        self.trainloader,
                    ),
                    errD.item(),
                    errG.item(),
                )
            )

    def _generate_data(self) -> None:
        """Generate synthetic EEG signals using the trained generator"""

        new_data = []
        new_label = []
        self.generator.eval()

        with torch.no_grad():
            # generate n data per class
            for nclass in range(self.config["num_aux_class"]):
                for n in range(self.config["num_epochs"]):
                    eval_noise_ = np.random.normal(
                        0, 1, (self.config["batch_size"], self.config["nz"])
                    )
                    # Here we are create a vector of size BS with the chosen class
                    eval_label = (
                        np.zeros((self.config["batch_size"],), dtype=int)
                    ) + nclass
                    eval_onehot = np.zeros(
                        (self.config["batch_size"], self.config["num_aux_class"])
                    )
                    eval_onehot[np.arange(self.config["batch_size"]), eval_label] = 1
                    eval_noise_[
                        np.arange(self.config["batch_size"]),
                        : self.config["num_aux_class"],
                    ] = eval_onehot[np.arange(self.config["batch_size"])]
                    eval_noise = torch.from_numpy(eval_noise_)
                    eval_noise.data.copy_(
                        eval_noise.view(self.config["batch_size"], self.config["nz"])
                    )
                    z = eval_noise.to(device).float()

                    fake_data = self.generator(z)

                    new_data.append(fake_data.data.cpu().numpy())
                    new_label.append(eval_label)

            new_data = np.asarray(new_data)
            new_data = np.concatenate(new_data)

            new_label = np.asarray(new_label)
            new_label = np.concatenate(new_label)

            # save generated data
            np.save("SISGAN_unseen%i.npy" % (self.test_idx), new_data)
            np.save("SISGAN_unseen%i_labels.npy" % (self.test_idx), new_label)
        print("generated data saved")

    def perform_loo(self) -> None:
        """Perform the leave one out analysis for each subject in the training dataset"""

        loo = LeaveOneOut()

        for self.train_idx, self.test_idx in loo.split(self.input_data):
            print(self.train_idx, self.test_idx)

            datainput = self.input_data[self.train_idx]
            labelinput = self.input_label[self.train_idx]

            EEGdata = np.concatenate(datainput)
            EEGlabel = np.concatenate(labelinput)

            # convert NumPy Array to Torch Tensor
            train_input = torch.from_numpy(EEGdata)
            train_label = torch.from_numpy(EEGlabel)

            # create the data loader for the training set
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

    trainer = SIS_GAN(config_file=args.config_file)
    trainer.perform_loo()
