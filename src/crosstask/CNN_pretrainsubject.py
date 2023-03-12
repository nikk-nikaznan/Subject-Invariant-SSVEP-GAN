import torch
import torch.nn as nn

import random
import numpy as np
from pathlib import Path
import glob
from utils import data_process

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class EEG_CNN_Subject(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=20, stride=4, bias=False),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=4, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

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


def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert actual.size(0) == predicted.size(0)
    return float(actual.eq(predicted).sum()) / actual.size(0)


def save_model(epoch, subject_predictor, optimizer_Pred, test_idx, filepath="pretrain_subject.cpt"):
    """Save the model and embeddings"""

    state = {"epoch": epoch, "state_dict": subject_predictor.state_dict(), "optimizer": optimizer_Pred.state_dict()}

    torch.save(state, filepath % (test_idx))
    print("Model Saved")


seed_n = np.random.randint(500)
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)

# Hyper Parameters
num_epochs = 200
learning_rate = 0.0001
dropout_level = 0.3
wdecay = 0.005
batch_size = 16

# data loading
main_path = f"{Path.home()}/Data/EEG/Offline_Experiment/Train/"
eeg_path = glob.glob(main_path + "S0*/")

input_data = []
for f in eeg_path:
    eeg_files = glob.glob(f + "data/*.npy")
    eeg_data = [np.load(f) for f in (eeg_files)]
    eeg_data = np.asarray(np.concatenate(eeg_data))
    eeg_data = data_process(eeg_data)
    input_data.append(eeg_data)

data_input = np.asarray(input_data)
data_input = data_input.swapaxes(2, 3)

train_subject = []
for num_s in range(data_input.shape[0]):
    train_subject.append(np.zeros(data_input.shape[1]) + num_s)

train_subject = np.asarray(train_subject)
num_subjects = train_subject.shape[0]
train_subject = train_subject.astype(np.int64)
EEGsubject = np.concatenate(train_subject)

EEGdata = np.concatenate(data_input)

subject_predictor = EEG_CNN_Subject().to(device)
subject_predictor.apply(weights_init)

# Loss and Optimizer
ce_loss = nn.CrossEntropyLoss()
optimizer_Pred = torch.optim.Adam(subject_predictor.parameters(), lr=learning_rate, weight_decay=wdecay)

# convert NumPy Array to Torch Tensor
train_input = torch.from_numpy(EEGdata)
train_label = torch.from_numpy(EEGsubject)

# create the data loader for the training set
trainset = torch.utils.data.TensorDataset(train_input, train_label)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# loop through the required number of epochs
for epoch in range(num_epochs):
    # print("Epoch:", epoch)

    cumulative_accuracy = 0
    for i, data in enumerate(trainloader, 0):
        # format the data from the dataloader
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.float()

        # Forward + Backward + Optimize
        optimizer_Pred.zero_grad()
        outputs = subject_predictor(inputs)
        loss = ce_loss(outputs, labels)
        loss.backward()
        optimizer_Pred.step()

        # calculate the accuracy over the training batch
        _, predicted = torch.max(outputs, 1)

        cumulative_accuracy += get_accuracy(labels, predicted)
print("Training Loss:", loss.data)
print("Training Accuracy: %2.1f" % ((cumulative_accuracy / len(trainloader) * 100)))

save_model(epoch, subject_predictor, optimizer_Pred)
