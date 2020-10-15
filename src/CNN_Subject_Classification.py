import torch
import torch.nn as nn
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def plot_error_matrix(cm, classes, cmap=plt.cm.Blues):
   """ Plot the error matrix for the neural network models """

   from sklearn.metrics import confusion_matrix
   import itertools

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   #plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   print(cm)

   thresh = cm.max()
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   
   plt.ylabel('True subject')
   plt.xlabel('Predicted subject')
   plt.tight_layout()
   
class CNN(nn.Module):
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
    assert(actual.size(0) == predicted.size(0))
    return float(actual.eq(predicted).sum()) / actual.size(0)

seed_n = 42
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)

# Hyper Parameters
num_epochs = 200
learning_rate = 0.0001
dropout_level = 0.35
wdecay = 0.005
batch_size = 16

# data input
# loading your own pre-processed data
data_input = [data_S01, data_S02, data_S03, data_S04, data_S05, data_S06, data_S07, data_S08, data_S09]
# data_S01.shape = (180, 1500, 2)
data_input = np.asarray(data_input)
# data_input.shape = (9, 180, 1500, 2)
data_input = data_input.swapaxes(2, 3)

# set subject label (S01 -- 0, S02 -- 1, ... , S09 -- 8)
subject_input = []
for num_s in range (data_input.shape[0]):
    subject_input.append(np.zeros(data_input.shape[1]) + num_s)
subject_input = np.asarray(subject_input)
subject_input = subject_input.astype(np.int64)
num_subjects = subject_input.shape[0]

EEGdata = np.concatenate(data_input)
EEGsubject = np.concatenate(subject_input)
correct = 0
total = 0
test_meanAcc = []

# kfold validation with random shuffle
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
sss.get_n_splits(EEGdata, EEGsubject)

for train_idx, test_idx in sss.split(EEGdata, EEGsubject):

    cnn = CNN().to(device)
    cnn.apply(weights_init)

    # Loss and Optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=wdecay)

    X_train = EEGdata[train_idx]
    y_train = EEGsubject[train_idx]
    X_test = EEGdata[test_idx]
    y_test = EEGsubject[test_idx]

    # convert NumPy Array to Torch Tensor
    train_input = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)
    test_input = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # create the data loader for the test set
    testset = torch.utils.data.TensorDataset(test_input, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # loop through the required number of epochs
    for epoch in range(num_epochs):
        # print("Epoch:", epoch)
        cnn.train()

        # loop through the batches yo!!!
        cumulative_accuracy = 0
        for i, data in enumerate(trainloader, 0):
            # format the data from the dataloader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(inputs)
            # print(outputs)
            # print(labels)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the accuracy over the training batch
            # print (outputs.shape)
            _, predicted = torch.max(outputs, 1)
            
            cumulative_accuracy += get_accuracy(labels, predicted)
        print("Training Loss:", loss.item())
        print("Training Accuracy: %2.1f" % ((cumulative_accuracy/len(trainloader)*100)))

    label_list=[]
    prediction_list=[]
    cnn.eval()
    test_cumulative_accuracy = 0
    for i, data in enumerate(testloader, 0):
        # format the data from the dataloader
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_inputs = test_inputs.float()    

        test_outputs = cnn(test_inputs)
        _, test_predicted = torch.max(test_outputs, 1)    

        test_acc = get_accuracy(test_labels,test_predicted)
        test_cumulative_accuracy += test_acc

        label_list.append(test_labels.data.cpu().numpy())
        prediction_list.append(test_predicted.data.cpu().numpy())
    
    u, u_counts = np.unique(test_label, return_counts=True)
    test_meanAcc.append(test_cumulative_accuracy/len(testloader))
    print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy/len(testloader)*100)))

test_meanAcc = np.asarray(test_meanAcc)
print("Mean Test Accuracy: %f" % test_meanAcc.mean())
print ("Standard Deviation: %f" % np.std(test_meanAcc, dtype=np.float64))

label_list = np.array(label_list)
cnf_labels = np.concatenate(label_list)
prediction_list = np.array(prediction_list)
cnf_predictions = np.concatenate(prediction_list)

class_names = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09"]
# Compute confusion matrix
cnf_matrix = confusion_matrix(cnf_labels, cnf_predictions)
# np.set_printoptions(precision=2)

# Normalise
cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cnf_matrix = cnf_matrix.round(2)
matplotlib.rcParams.update({'font.size': 10})

# Plot normalized confusion matrix
plt.figure()
plot_error_matrix(cnf_matrix, classes=class_names)
plt.tight_layout()
filename = "filename.pdf"
plt.savefig(filename, format='PDF', bbox_inches='tight')
plt.show()
