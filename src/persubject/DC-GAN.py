import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=320, help='dimensionality of the latent space')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
parser.add_argument('--nz', type=int, default=100, help="size of the latent z vector used as the generator input.")
opt = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 30
dropout_level = 0.5
nz = opt.nz
d_critic = 1
g_critic = 1

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
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=10, stride=2, bias=False),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=1, bias=False),
            nn.PReLU()
        )
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

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        realfake = self.classifier(out)

        return realfake

def dcgan(datatrain, label, nseed):

    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    generator = EEG_CNN_Generator().to(device)
    discriminator = EEG_CNN_Discriminator().to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    real_label = 1
    fake_label = 0
    new_data = []

    # GAN Training ---------------------------------------------------------------
    discriminator.train()
    generator.train()

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):
            real_data, _ = data
            input_size = real_data.size(0)
            real_data = real_data.to(device)
            real_data = real_data.float()
            
            # Configure input
            label = torch.empty(input_size, 1).to(device) # Discriminator label
            z = torch.randn(real_data.shape[0], nz).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            if i % 1 == 0:
                optimizer_Dis.zero_grad()
                # train with real
                label.data.fill_(real_label)
                output_real = discriminator(real_data)

                # Calculate error and backpropagate
                errD_real = adversarial_loss(output_real, label)
                errD_real.backward()
                
                # train with fake   
                fake_data = generator(z)
                label.data.fill_(fake_label)

                output_fake = discriminator(fake_data.detach())
                
                errD_fake = adversarial_loss(output_fake, label)
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

                fake_data = generator(z)
                label.data.fill_(real_label)

                output = discriminator(fake_data)
                errG = adversarial_loss(output, label)
                errG.backward()
                optimizer_Gen.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (epoch, opt.n_epochs, i, len(dataloader), errD.item(), errG.item(), ))


    # generate the data
    discriminator.eval()
    generator.eval()
    for epoch in range(150):
        z = torch.randn(real_data.shape[0], nz).to(device)

        fake_data = generator(z)
        
        output = discriminator(fake_data)
        
        fake_data = fake_data.data.cpu().numpy()
        new_data.append(fake_data)

    new_data = np.concatenate(new_data) 
    new_data = np.asarray(new_data)
    print (new_data.shape)
    return new_data

data_filename = "/home/nikkhadijah/Data/NewGANdata/DCGAN_S%i_class%i.npy"
label_filename = "/home/nikkhadijah/Data/NewGANdata/DCGAN_S%i_class%i_labels.npy"

# training data 
# loading your own pre-processed data
input_data = [data_S01, data_S02, data_S03, data_S04, data_S05, data_S06, data_S07, data_S08, data_S09]
# data_S0X.shape = (144, 1500, 2)
input_label = [label_S01, label_S02, label_S03, label_S04, label_S05, label_S06, label_S07, label_S08, label_S09]

for ns in range (0, 9):
    data_train = input_data[ns]
    label_train = input_label[ns]
    # print (data_train.shape)

    #  ns == 0 or ns == 10: 
    data_train = data_train.swapaxes(1, 2)
    # separate according to class label
    data_train0 = []
    data_train1 = []
    data_train2 = []
    label_train0 = []
    label_train1 = []
    label_train2 = []

    for i in range (len(label_train)):
        if label_train[i] == 0:
            data_train0.append(data_train[i, :, :])
            label_train0.append(label_train[i])
        
        if label_train[i] == 1:
            data_train1.append(data_train[i, :, :])
            label_train1.append(label_train[i])
        
        if label_train[i] == 2:
            data_train2.append(data_train[i, :, :])
            label_train2.append(label_train[i])
        

    data_train0 = np.asarray(data_train0)
    data_train0 = data_train0
    label_train0 = np.asarray(label_train0)

    data_train1 = np.asarray(data_train1)
    label_train1 = np.asarray(label_train1)

    data_train2 = np.asarray(data_train2)
    label_train2 = np.asarray(label_train2)


    seed_n = np.random.randint(500)
    print (seed_n)

    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)

    gen_time = 0
    for nclass in range (0, 3):

        if nclass == 0:
            data_train = data_train0
            label_train = label_train0
        if nclass == 1:
            data_train = data_train1
            label_train = label_train1
        if nclass == 2:
            data_train = data_train2
            label_train = label_train2


        # generative model
        print ("*********Training Generative Model*********")
        gen_data = dcgan(data_train, label_train, seed_n)

        # save generated data
        if nclass == 0:
            fake_data0 = gen_data
            fake_label0 = np.zeros(len(fake_data0))
            np.save(data_filename % (ns, nclass), fake_data0)
            np.save(label_filename % (ns, nclass), fake_label0)
        if nclass == 1:
            fake_data1 = gen_data
            fake_label1 = np.ones(len(fake_data1))
            np.save(data_filename % (ns, nclass), fake_data1)
            np.save(label_filename % (ns, nclass), fake_label1)
        if nclass == 2:
            fake_data2 = gen_data
            fake_label2 = (np.ones(len(fake_data2))) + 1
            np.save(data_filename % (ns, nclass), fake_data2)
            np.save(label_filename % (ns, nclass), fake_label2)
