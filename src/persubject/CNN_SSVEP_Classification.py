import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=10, stride=1),
            nn.BatchNorm1d(num_features=8),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_level),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(5960, 600),
            nn.PReLU(),
            nn.Dropout(dropout_level),
            nn.Linear(600, 60),
            nn.PReLU(),
            nn.Dropout(dropout_level),
            nn.Linear(60, num_classes),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert actual.size(0) == predicted.size(0)
    return float(actual.eq(predicted).sum()) / actual.size(0)


seed_n = np.random.randint(500)
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)

# Hyper Parameters
num_epochs = 150
learning_rate = 0.0001
dropout_level = 0.5
wdecay = 0.04
num_classes = 3
batch_size = 5

# real data
# loading your own pre-processed data
real_data = [data_S01, data_S02, data_S03, data_S04, data_S05, data_S06, data_S07, data_S08, data_S09]
# data_S0X.shape = (144, 1500, 2)
real_data = np.asarray(real_data)
# data_input.shape = (9, 144, 1500, 2)
real_data = real_data.swapaxes(2, 3)
# frequency label
real_label = [label_S01, label_S02, label_S03, label_S04, label_S05, label_S06, label_S07, label_S08, label_S09]
real_label = np.asarray(real_label)
real_label = real_label.astype(np.int64)

# fake data
fake_data = [
    data_fakeS01,
    data_fakeS02,
    data_fakeS03,
    data_fakeS04,
    data_fakeS05,
    data_fakeS06,
    data_fakeS07,
    data_fakeS08,
    data_fakeS09,
]
# data_fakeS0X.shape = (144, 2, 1500)
fake_data = np.asarray(fake_data)
# data_input.shape = (9, 144, 2, 1500)
fake_label = [
    label_fakeS01,
    label_fakeS02,
    label_fakeS03,
    label_fakeS04,
    label_fakeS05,
    label_fakeS06,
    label_fakeS07,
    label_fakeS08,
    label_fakeS09,
]
fake_label = np.asarray(fake_label)
fake_label = fake_label.astype(np.int64)

for ns in range(real_data.shape[0]):
    print("Subject", ns)

    # ************************************************************
    # For augmentation
    # EEGdata = [real_data[ns, :, :, :], fake_data[ns, :, :, :]]
    # EEGdata = np.asarray(EEGdata)
    # EEGdata = np.concatenate(EEGdata)
    # EEGlabel = [real_label[ns, :], fake_label[ns, :]]
    # EEGlabel = np.asarray(EEGlabel)
    # EEGlabel = np.concatenate(EEGlabel)

    # ************************************************************
    # For non-augmentation
    EEGdata = real_data[ns, :, :, :]
    EEGdata = np.concatenate(EEGdata)
    EEGlabel = real_label[ns, :]
    EEGlabel = np.concatenate(EEGlabel)

    # kfold validation with random shuffle
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    sss.get_n_splits(EEGdata, EEGlabel)
    test_meanAcc = []

    for train_idx, test_idx in sss.split(EEGdata, EEGlabel):
        # print (train_idx, test_idx)

        cnn = CNN().to(device)
        cnn.apply(weights_init)
        cnn.train()

        # Loss and Optimizer
        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=wdecay)

        train_input = EEGdata[train_idx]
        train_label = EEGlabel[train_idx]

        test_input = EEGdata[test_idx]
        test_label = EEGlabel[test_idx]

        # convert NumPy Array to Torch Tensor
        train_input = torch.from_numpy(train_input)
        train_label = torch.from_numpy(train_label)

        # create the data loader for the training set
        trainset = torch.utils.data.TensorDataset(train_input, train_label)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        for epoch in range(num_epochs):
            # print("Epoch:", epoch)

            cumulative_accuracy = 0
            cumulative_loss = 0
            for i, data in enumerate(trainloader, 0):
                # format the data from the dataloader
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = ce_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                # calculate the accuracy over the training batch
                _, predicted = torch.max(outputs, 1)

                cumulative_accuracy += get_accuracy(labels, predicted)
            # print("Training Loss:", loss.item())
            # print("Training Accuracy: %2.1f" % ((cumulative_accuracy/len(trainloader)*100)))

        cnn.eval()
        test_cumulative_accuracy = 0

        test_input = torch.from_numpy(test_input)
        test_label = torch.from_numpy(test_label)

        # create the data loader for the test set
        testset = torch.utils.data.TensorDataset(test_input, test_label)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

        for i, data in enumerate(testloader, 0):
            # format the data from the dataloader
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_inputs = test_inputs.float()

            test_outputs = cnn(test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)
            test_acc = get_accuracy(test_labels, test_predicted)
            test_cumulative_accuracy += test_acc

        u, u_counts = np.unique(test_label, return_counts=True)
        test_meanAcc.append(test_cumulative_accuracy / len(testloader))

    test_meanAcc = np.asarray(test_meanAcc)
    print("Mean Test Accuracy: %f" % test_meanAcc.mean())
    print("Standard Deviation: %f" % np.std(test_meanAcc, dtype=np.float64))
