import argparse
import glob
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from utils import data_process

from models import weights_init, EEG_CNN_Discriminator, EEG_CNN_Generator, EEG_CNN_Subject

from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
    help="number of training steps for discriminator per iter",
)
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument(
    "--nz",
    type=int,
    default=103,
    help="size of the latent z vector used as the generator input.",
)
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 30
dropout_level = 0.5
nz = opt.nz
num_classes = 3
num_subjects = 8
# lambda - weight for subject predictor loss
lmbd = 0.3


def gen_noise():
    """Generate random noise and set the last elements to a random selection of one hot labels"""

    gen_noise_ = np.random.normal(0, 1, (batch_size, nz))
    gen_label = np.random.randint(0, num_classes, batch_size)
    gen_subject = np.random.randint(0, num_subjects, batch_size)
    gen_onehot = np.zeros((batch_size, num_classes))
    gen_onehot[np.arange(batch_size), gen_label] = 1
    gen_noise_[np.arange(batch_size), :num_classes] = gen_onehot[np.arange(batch_size)]
    gen_noise = torch.from_numpy(gen_noise_)
    gen_noise.data.copy_(gen_noise.view(batch_size, nz))
    z = gen_noise.to(device)

    z = z.float()

    return z, gen_label, gen_subject






# Loss function
adversarial_loss = nn.BCEWithLogitsLoss()  # dis_criterion
ce_loss = nn.CrossEntropyLoss()

real_label = 1
fake_label = 0

batches_done = 0


def training_GAN(
    dataloader,
    generator,
    discriminator,
    subject_predictor,
    optimizer_Gen,
    optimizer_Dis,
):
    discriminator.train()
    generator.train()
    subject_predictor.eval()

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):
            # Real data from data loader with matching labels
            real_data, real_aux_label, real_subject = data
            real_subject = real_subject.to(device)
            input_size = real_data.size(0)
            real_data = real_data.to(device)
            real_data = real_data.float()
            real_aux_label = real_aux_label.to(device)
            real_aux_label = real_aux_label.long()

            # Configure input
            input_data = real_data.data
            dis_label = torch.empty(input_size, 1).to(device)  # Discriminator label

            z, z_label, z_subject = gen_noise()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            if i % 1 == 0:
                # train with real
                optimizer_Dis.zero_grad()

                dis_label.data.fill_(real_label)
                aux_label = real_aux_label.data
                subject_label = real_subject.data

                dis_output, aux_output = discriminator(input_data)
                dis_errD_real = adversarial_loss(dis_output, dis_label)
                aux_errD_real = ce_loss(aux_output, aux_label)

                errD_real = dis_errD_real + aux_errD_real
                errD_real.backward()

                # train with fake
                fake_data = generator(z)

                dis_label.data.fill_(fake_label)
                aux_label.data.resize_(input_size).copy_(torch.from_numpy(z_label))
                subject_label.data.resize_(input_size).copy_(torch.from_numpy(z_subject))

                dis_output, aux_output = discriminator(fake_data.detach())
                dis_errD_fake = adversarial_loss(dis_output, dis_label)
                aux_errD_fake = ce_loss(aux_output, aux_label)

                errD_fake = dis_errD_fake + aux_errD_fake
                errD_fake.backward()

                errD = errD_real + errD_fake
                optimizer_Dis.step()

            # -----------------
            #  Train Generator
            # -----------------
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if i % 1 == 0:
                # Reset gradients
                optimizer_Gen.zero_grad()

                dis_label.data.fill_(real_label)
                fake_data = generator(z)

                dis_output, aux_output = discriminator(fake_data)
                dis_errG = adversarial_loss(dis_output, dis_label)
                aux_errG = ce_loss(aux_output, aux_label)

                subject_output = subject_predictor(fake_data)
                sub_errG, _ = torch.max(subject_output, 1)

                sub_errG = lmbd * sub_errG.mean()

                errG = dis_errG + aux_errG + sub_errG
                errG.backward()
                optimizer_Gen.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] "
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                errD.item(),
                errG.item(),
            )
        )

    return generator


def generate_GAN(generator):
    new_data = []
    new_label = []
    generator.eval()
    with torch.no_grad():
        for nclass in range(0, 3):
            for epoch in range(opt.n_epochs):
                # generate n data per class
                eval_noise_ = np.random.normal(0, 1, (batch_size, nz))
                # Here we are create a vector of size BS with the chosen class
                eval_label = (np.zeros((batch_size,), dtype=int)) + nclass
                eval_onehot = np.zeros((batch_size, num_classes))
                eval_onehot[np.arange(batch_size), eval_label] = 1
                eval_noise_[np.arange(batch_size), :num_classes] = eval_onehot[np.arange(batch_size)]
                eval_noise = torch.from_numpy(eval_noise_)
                eval_noise.data.copy_(eval_noise.view(batch_size, nz))
                z = eval_noise.to(device)
                z = z.float()

                fake_data = generator(z)

                fake_data = fake_data.data.cpu().numpy()

                new_data.append(fake_data)
                new_label.append(eval_label)

        new_data = np.asarray(new_data)
        new_data = np.concatenate(new_data)

        new_label = np.asarray(new_label)
        new_label = np.concatenate(new_label)

        return new_data, new_label


def sisgan(datatrain, subject, label, nseed, test_idx):
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)
    subject = torch.from_numpy(subject)

    dataset = TensorDataset(datatrain, label, subject)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    generator = EEG_CNN_Generator().to(device)
    discriminator = EEG_CNN_Discriminator().to(device)
    subject_predictor = EEG_CNN_Subject().to(device)

    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # Optimizer
    optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    filename_loo_subject = "pretrain_subject_unseen%i.cpt"
    state = torch.load(filename_loo_subject % (test_idx))
    subject_predictor.load_state_dict(state["state_dict"])

    # training GAN
    generator = training_GAN(
        dataloader,
        generator,
        discriminator,
        subject_predictor,
        optimizer_Gen,
        optimizer_Dis,
    )

    # generate the data
    new_data, new_label = generate_GAN(generator)

    return new_data, new_label


loo = LeaveOneOut()

# data loading
main_path = f"{Path.home()}/Data/EEG/Offline_Experiment/Train/"
eeg_path = glob.glob(main_path + "S0*/")

# data loading
main_path = "sample_data/Real/"
eeg_path = glob.glob(main_path + "S0*/")

input_data = []
input_label = []
for f in eeg_path:
    eeg_files = glob.glob(f + "data/*.npy")
    eeg_data = [np.load(f) for f in (eeg_files)]
    eeg_data = np.asarray(np.concatenate(eeg_data))
    eeg_data = data_process(eeg_data)
    input_data.append(eeg_data)

    eeg_files = glob.glob(f + "label/*.npy")
    eeg_label = [np.load(f) for f in (eeg_files)]
    eeg_label = np.asarray(np.concatenate(eeg_label))
    eeg_label = eeg_label.astype(np.int64)
    input_label.append(eeg_label)

input_data = np.asarray(input_data)
input_data = input_data.swapaxes(2, 3)
input_label = np.asarray(input_label)

for train_idx, test_idx in loo.split(input_data):
    print(train_idx, test_idx)

    train_data = input_data[train_idx]
    train_label = input_label[train_idx]

    train_subject = []
    for num_s in range(train_data.shape[0]):
        train_subject.append(np.zeros(train_data.shape[1]) + num_s)

    train_subject = np.asarray(train_subject)
    num_subjects = train_subject.shape[0]
    train_subject = train_subject.astype(np.int64)

    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    train_subject = np.concatenate(train_subject)

    seed_n = np.random.randint(500)
    print(seed_n)

    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)

    gen_data, gen_label = sisgan(train_data, train_subject, train_label, seed_n, test_idx)

    data_filename = "SISGAN_unseen%i.npy"
    label_filename = "SISGAN_unseen%i_labels.npy"

    # save generated data
    np.save(data_filename % (test_idx), gen_data)
    np.save(label_filename % (test_idx), gen_label)
