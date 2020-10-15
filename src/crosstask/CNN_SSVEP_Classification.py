import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import LeaveOneOut

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        self.dense_layers = nn.Sequential(
            nn.Linear(2816, num_classes)
            )

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
    assert(actual.size(0) == predicted.size(0))
    return float(actual.eq(predicted).sum()) / actual.size(0)

seed_n = np.random.randint(500)
print(seed_n)
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)

num_epochs = 120
learning_rate = 0.0001
dropout_level = 0.2
wdecay = 0.0001
num_classes = 3
batch_size = 32

# loading your own pre-processed data
# real data -- training
real_data = [data_S01, data_S02, data_S03, data_S04, data_S05, data_S06, data_S07, data_S08, data_S09]
# data_S0X.shape = (180, 1500, 2)
real_data = np.asarray(real_data)
# real_data.shape = (9, 180, 1500, 2)
real_data = real_data.swapaxes(2, 3)
# frequency label
real_label = [label_S01, label_S02, label_S03, label_S04, label_S05, label_S06, label_S07, label_S08, label_S09]
real_label = np.asarray(real_label)
real_label = real_label.astype(np.int64)

# fake data
fake_data = [data_fake_unseenS01, data_fake_unseenS02, data_fake_unseenS03, data_fake_unseenS04, data_fake_unseenS05, data_fake_unseenS06, data_fake_unseenS07, data_fake_unseenS08, data_fake_unseenS09]
# data_fake_unseenS0X.shape = (1440, 2, 1500)
fake_data = np.asarray(fake_data)
fake_label = [label_fake_unseenS01, label_fake_unseenS02, label_fake_unseenS03, label_fake_unseenS04, label_fake_unseenS05, label_fake_unseenS06, label_fake_unseenS07, label_fake_unseenS08, label_fake_unseenS09]
fake_label = np.asarray(fake_label)
fake_label = fake_label.astype(np.int64)

cnn = CNN().to(device)
cnn.apply(weights_init)
cnn.train()

# Loss and Optimizer
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=wdecay)

# ************************************************************
# # For augmentation
# train_input = [real_data, fake_data]
# train_input = np.asarray(train_input)
# train_label = [real_label[train_idx], fake_label[test_idx]]
# train_label = np.asarray(train_label) 

# For non-augmentation -- real
train_input = real_data
train_label = real_label

# # For non-augmentation -- fake
# train_input = fake_data
# train_label = fake_label
# ***************************************************************

train_input = np.concatenate(train_input)
train_label = np.concatenate(train_label) 

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

# real data - testing
test_data = [data_T01, data_T02, data_T03]
test_data = np.asarray(test_data)
test_data = test_data.swapaxes(2, 3)
# frequency label
test_label = [label_T01, label_T02, label_T03]
test_label = np.asarray(test_label)
test_label = test_label.astype(np.int64)

test_input = np.concatenate(test_data)
test_label = np.concatenate(test_label)

# convert NumPy Array to Torch Tensor
test_input = torch.from_numpy(test_input)
test_label = torch.from_numpy(test_label)

# create the data loader for the training set
testset = torch.utils.data.TensorDataset(test_input, test_label)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

for i, data in enumerate(testloader, 0):
    # format the data from the dataloader
    test_inputs, test_labels = data
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    # inputs, labels = Variable(inputs), Variable(labels)
    test_inputs = test_inputs.float()    

    test_outputs = cnn(test_inputs)
    _, test_predicted = torch.max(test_outputs, 1)    
    test_acc = get_accuracy(test_labels,test_predicted)
    test_cumulative_accuracy += test_acc

print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy/len(testloader)*100)))
