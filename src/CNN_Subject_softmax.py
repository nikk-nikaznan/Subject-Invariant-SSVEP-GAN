import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

import Data_prep_12

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
num_epochs = 1
learning_rate = 0.00001
dropout_level = 0.0
wdecay = 0.04
batch_size = 5
num_subjects = 9

class EEG_CNN_Subject(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=20, stride=4, bias=False),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.Dropout(0.0))

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            nn.Dropout(0.0))

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            nn.Dropout(0.0))

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
            nn.Dropout(0.0))

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=4, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
            nn.Dropout(0.0))

        self.classifier = nn.Linear(2816, num_subjects)


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

subject_predictor = EEG_CNN_Subject().to(device)
state = torch.load("pretrain_subject.cpt")
subject_predictor.load_state_dict(state['state_dict'])

def sub_invariant(data_fake, label_fake):

    train_input = torch.from_numpy(data_fake)
    train_label = torch.from_numpy(label_fake)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)

    mean_outputs = []
    subject_predictor.eval()
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            # format the data from the dataloader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            outputs = subject_predictor(inputs)
            outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.detach().cpu().numpy()
            mean_outputs.append(outputs)

    mean_outputs = np.asarray(mean_outputs)
    mean_outputs = np.concatenate(mean_outputs)
    mean_outputs = np.mean(mean_outputs, 0)

    return mean_outputs

data_SISGAN = np.load('SISGAN_CT.npy')
label_SISGAN = np.load('SISGAN_CT_labels.npy')
label_SISGAN = label_SISGAN.astype(np.int64)

data_ACGAN = np.load('ACGAN_CT.npy')
label_ACGAN = np.load('ACGAN_CT_labels.npy')
label_ACGAN = label_ACGAN.astype(np.int64)

mean_SISGAN = sub_invariant(data_SISGAN, label_SISGAN)
mean_ACGAN = sub_invariant(data_ACGAN, label_ACGAN)

X = np.arange(9)

plt.bar(X + 0.05, mean_ACGAN, color='tab:blue', width = 0.4)
plt.bar(X + 0.45, mean_SIACGAN, color='tab:green', width = 0.4)

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

plt.xlabel('Subject' , fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.xticks(X + 0.50 / 2, ('S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09'))

plt.legend(handles = [mpatches.Patch(color='tab:blue', label='AC-GAN'), mpatches.Patch(color='tab:green', label='SIS-GAN')], fontsize=15)

plt.savefig('softmax.pdf')