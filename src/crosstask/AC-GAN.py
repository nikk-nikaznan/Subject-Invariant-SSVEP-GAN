import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
parser.add_argument('--nz', type=int, default=103, help="size of the latent z vector used as the generator input.")
opt = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 30
dropout_level = 0.5
nz = opt.nz
num_classes = 3

def gen_noise():
    """Generate random noise and set the last elements to a random selection of one hot labels"""

    gen_noise_ = np.random.normal(0, 1, (batch_size, nz))
    gen_label = np.random.randint(0, num_classes, batch_size)
    gen_onehot = np.zeros((batch_size, num_classes))
    gen_onehot[np.arange(batch_size), gen_label] = 1
    gen_noise_[np.arange(batch_size), :num_classes] = gen_onehot[np.arange(batch_size)]
    gen_noise = (torch.from_numpy(gen_noise_))
    gen_noise.data.copy_(gen_noise.view(batch_size, nz))
    z = gen_noise.to(device)

    z = z.float()

    return z, gen_label

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class EEG_CNN_Generator(nn.Module):
    def __init__(self):
        super(EEG_CNN_Generator, self).__init__()

        self.nz = nz
        self.dense = nn.Sequential(
            nn.Linear(self.nz, 2816),
            nn.PReLU()
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=256, kernel_size=20, stride=2, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU())

        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=10, stride=2, bias=False),
            nn.PReLU())

        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.PReLU())

        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=1, bias=False),
            nn.PReLU())

        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=2, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid())

    def forward(self, z):

        out = self.dense(z)
        out = out.view(out.size(0), 16, 176)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class EEG_CNN_Discriminator(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=20, stride=4, bias=False),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.Dropout(dropout_level))

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            nn.Dropout(dropout_level))

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            nn.Dropout(dropout_level))

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
            nn.Dropout(dropout_level))

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=4, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
            nn.Dropout(dropout_level))

        self.classifier = nn.Linear(2816, 1)
        self.aux = nn.Linear(2816, 3)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        realfake = self.classifier(out)
        classes = self.aux(out)

        return realfake, classes

generator = EEG_CNN_Generator().to(device)
discriminator = EEG_CNN_Discriminator().to(device)

discriminator.apply(weights_init)
generator.apply(weights_init)

# Loss function
adversarial_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()
ce_loss.to(device)
adversarial_loss.to(device)

# Optimizer
optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 1 for real, 0 for fake
real_label = 1
fake_label = 0

batches_done = 0  

def training_GAN(dataloader):
    discriminator.train()
    generator.train()

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):
            real_data, real_aux_label = data # Real data from data loader with matching labels
            input_size = real_data.size(0)
            real_data = real_data.to(device)
            real_data = real_data.float()
            real_aux_label = real_aux_label.to(device)
            real_aux_label = real_aux_label.long()
            
            # Configure input
            input_data = real_data.data
            dis_label = torch.empty(input_size, 1).to(device) # Discriminator label

            z, z_label = gen_noise()
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
                
                dis_output, aux_output = discriminator(input_data)
                dis_errD_real = adversarial_loss(dis_output, dis_label)
                aux_errD_real = ce_loss(aux_output, aux_label)

                errD_real = dis_errD_real + aux_errD_real
                errD_real.backward()

                # train with fake
                
                fake_data = generator(z)

                dis_label.data.fill_(fake_label)
                aux_label.data.resize_(input_size).copy_(torch.from_numpy(z_label))

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

                errG = dis_errG + aux_errG
                errG.backward()
                optimizer_Gen.step()


        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (epoch, opt.n_epochs, i, len(dataloader), errD.item(), errG.item(), ))
    
    return

def generate_GAN():

    new_data = []
    new_label = []
    generator.eval()
    with torch.no_grad():
        for nclass in range (0, 3):
            for epoch in range(200):    
                # generate n data per class
                eval_noise_ = np.random.normal(0, 1, (batch_size, nz))
                eval_label = (np.zeros((batch_size,), dtype=int)) + nclass # Here we are create a vector of size BS with the chosen class
                eval_onehot = np.zeros((batch_size, num_classes))
                eval_onehot[np.arange(batch_size), eval_label] = 1
                eval_noise_[np.arange(batch_size), :num_classes] = eval_onehot[np.arange(batch_size)]
                eval_noise = (torch.from_numpy(eval_noise_))
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


def acgan(datatrain, label, nseed):
    
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # training GAN
    training_GAN(dataloader)

    # generate the data
    new_data, new_label = generate_GAN()
    
    return new_data, new_label


# training data 
# loading your own pre-processed data
input_data = [data_S01, data_S02, data_S03, data_S04, data_S05, data_S06, data_S07, data_S08, data_S09]
# data_S0X.shape = (180, 1500, 2)
input_data = np.asarray(input_data)
input_data = input_data.swapaxes(2, 3)
input_label = [label_S01, label_S02, label_S03, label_S04, label_S05, label_S06, label_S07, label_S08, label_S09]
input_label = np.asarray(input_label)
input_label = input_label.astype(np.int64)

train_data = np.concatenate(input_data)
train_label = np.concatenate(input_label)

seed_n = np.random.randint(500)
print (seed_n)

random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)

gen_data, gen_label = acgan(train_data, train_label, seed_n)

data_filename = "ACGAN_CT.npy"
label_filename = "ACGAN_CT_labels.npy"

# save generated data
np.save(data_filename, gen_data)
np.save(label_filename, gen_label)
