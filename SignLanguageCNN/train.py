import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing


#------------------------- CNN Architectures ----------------------#
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25 * 25 * 25, 100)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(100, 5)
        # must be 10 because of # of class labels + 1

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 25 * 25 * 25)  # reshape
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out

#THIS MODEL WAS THE BEST
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(50)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(25)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25*25*25, 100)
        self.drop = nn.Dropout(0.5) #50 % probability
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(100, 10)
        # must be 10 because of # of class labels + 1

    def forward(self, x):
        out = self.pool1(self.act1(self.batch1(self.conv1(x))))
        out = self.pool2(self.act2(self.batch2(self.conv2(out))))
        out = out.view(-1, 25*25*25)  # reshape
        out = self.act3(self.drop(self.fc1(out)))
        out = self.fc2(out)
        return out

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25 * 25 * 25, 100)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 25 * 25 * 25) #reshape
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out

class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(50)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(25)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25*25*25, 100)
        self.drop = nn.Dropout(0.5) #50 % probability
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        # must be 10 because of # of class labels + 1

    def forward(self, x):
        out = self.pool1(self.act1(self.batch1(self.conv1(x))))
        out = self.pool2(self.act2(self.batch2(self.conv2(out))))
        out = out.view(-1, 25*25*25)  # reshape
        out = self.act3(self.drop(self.fc1(out)))
        out = self.fc2(out)
        return out

#---------------------------------- ARCHITECTURES DONE -------------------------------

import datetime
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch, loss_train / len(train_loader)))


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))


def Split_Train_Model(X_org, y_org):
    # Encode labels A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8, I=9, unknown=-1
    le = preprocessing.LabelEncoder()
    le.fit(y_org.ravel())
    y_encoded = le.transform(y_org.ravel())
    add = np.ones(y_encoded.shape, dtype=int)
    y_encoded = y_encoded + add
    i = 0
    for y in y_encoded:
        if (1 <= y <= 9):
            i = i + 1
        else:
            y = -1
            y_encoded[i] = y
            i = i + 1

    # 80/20 train test split for Easy test sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_org, y_encoded, test_size=0.2, random_state=42)

    # Normalize and convert the Xtrain and Xtest to tensors
    Xtrain_tensor = torch.tensor(Xtrain / 255, dtype=torch.float32)
    Xtrain_tensor = torch.transpose(Xtrain_tensor, 1, 3)
    Xtest_tensor = torch.tensor(Xtest / 255, dtype=torch.float32)
    Xtest_tensor = torch.transpose(Xtest_tensor, 1, 3)

    # Convert the training and test sets into a list of tuples
    tensor_list = []
    i = -1
    for x in Xtrain_tensor:
        i = i + 1
        tensor_list.append((x, ytrain[i]))
    tensor_list_test = []
    i = -1
    for x in Xtest_tensor:
        i = i + 1
        tensor_list_test.append((x, ytest[i]))

    return tensor_list, tensor_list_test


#--------------------------------TRAIN OUR MODEL-------------------------------------
#Load the dataset and labels and do a 80/20 training test split
X_org = np.load('train_data.npy')
y_org = np.load('train_labels.npy')
tensor_list, tensor_list_test = Split_Train_Model(X_org,y_org)

#Train the model with the training set (80% of dataset)
train_loader = torch.utils.data.DataLoader(tensor_list, batch_size=64, shuffle=True)
model = Net2()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
)

#SAVE THE MODEL
torch.save(model.state_dict(), "Model")





