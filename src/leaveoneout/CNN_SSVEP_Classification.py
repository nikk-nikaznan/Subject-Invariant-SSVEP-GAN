import torch
import torch.nn as nn
import random
import numpy as np
import glob
from sklearn.model_selection import LeaveOneOut
from utils import data_process

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
        self.dense_layers = nn.Sequential(nn.Linear(2816, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert actual.size(0) == predicted.size(0)
    return float(actual.eq(predicted).sum()) / actual.size(0)


seed_n = np.random.randint(500)
print(seed_n)
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)

num_epochs = 1
learning_rate = 0.0001
dropout_level = 0.2
wdecay = 0.0001
num_classes = 3
batch_size = 32

loo = LeaveOneOut()

# data loading
# fake data
main_path = "sample_data/Fake/"
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

    cnn = CNN().to(device)
    cnn.apply(weights_init)
    cnn.train()

    # Loss and Optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=wdecay)

    # For non-augmentation -- real
    train_input = input_data[train_idx]
    train_label = input_label[train_idx]

    # ***************************************************************

    train_input = np.concatenate(train_input)
    train_label = np.concatenate(train_label)

    test_input = input_data[test_idx]
    test_label = input_label[test_idx]

    test_input = np.concatenate(test_input)
    test_label = np.concatenate(test_label)

    # convert NumPy Array to Torch Tensor
    train_input = torch.from_numpy(train_input)
    train_label = torch.from_numpy(train_label)
    test_input = torch.from_numpy(test_input)
    test_label = torch.from_numpy(test_label)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = torch.utils.data.TensorDataset(test_input, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

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

    for i, data in enumerate(testloader, 0):
        # format the data from the dataloader
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        # inputs, labels = Variable(inputs), Variable(labels)
        test_inputs = test_inputs.float()

        test_outputs = cnn(test_inputs)
        _, test_predicted = torch.max(test_outputs, 1)
        test_acc = get_accuracy(test_labels, test_predicted)
        test_cumulative_accuracy += test_acc

    u, u_counts = np.unique(test_label, return_counts=True)
    # print (u_counts)
    print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy / len(testloader) * 100)))
