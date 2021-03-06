import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing

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



def validate(model, val_loader):
    for name, loader in [("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))

    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    NumberofLabels = np.array(labels)
    unique = np.unique(NumberofLabels)

    class_correct = list(0. for i in range(len(unique)))
    class_total = list(0. for i in range(len(unique)))
    predicted_labels = []
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.append(predicted)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i] - 1
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(unique)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    test_predicted = []
    for p in predicted_labels:
        for i in range(len(p)):
            test_predicted.append(p[i])

    return test_predicted


def Format_data(X_org, y_org):
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

    # Normalize and convert the X_org test data
    Xtest_tensor = torch.tensor(X_org / 255, dtype=torch.float32)
    Xtest_tensor = torch.transpose(Xtest_tensor, 1, 3)

    #Convert the training and test sets into a list of tuples
    tensor_list_test = []
    i = -1
    for x in Xtest_tensor:
        i = i + 1
        tensor_list_test.append((x, y_encoded[i]))

    return  tensor_list_test



#Load the dataset and labels and do 80/20 training test split
X_org = np.load('train_data.npy')
y_org = np.load('train_labels.npy')
tensor_list_test = Format_data(X_org,y_org)

#Load the trained model from the train.py function
model = Net2()
model.load_state_dict(torch.load("Model"))
model.eval()

# VALIDATE OUR MODEL
val_loader = torch.utils.data.DataLoader(tensor_list_test, batch_size=64, shuffle=True)
test_predicted = validate(model, val_loader)


