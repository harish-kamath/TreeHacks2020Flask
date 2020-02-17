import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

'''
STEP 1: LOADING DATASET
'''

train = open('train_data.txt', 'r')
lines = train.readlines()
data = []
out = []
# temp = []
for line in lines:
    temp = line.split(",")
    conv = []
    for i in range(len(temp)):
        if i in [1,2,3,4,5,6,7,8,9,10,11,16,17]:
            conv.append(float(temp[i]))
        elif i == len(temp) - 1:
            out.append(int(temp[i]))
    data.append(conv) 
a = list(zip(data, out))
print(a[0])
#a = np.array(a)
#X_train = torch.tensor(a, dtype=torch.float) # 3 X 2 tensor
#y_train = torch.tensor(tuple(out), dtype=torch.float) # 3 X 1 tensor
train_loader = a
print(train_loader)
# xPredicted = torch.tensor((data[0]), dtype=torch.float) # 1 X 2 tensor
# print(X.size(), y.size())


test = open('test_data.txt', 'r')
lines = test.readlines()
data = []
out = []
# temp = []
for line in lines:
    temp = line.split(",")
    conv = []
    # print(temp)
    for i in range(len(temp)):
        if i in [1,2,3,4,5,6,7,8,9,10,11,16,17]:
            conv.append(float(temp[i]))
        elif i == len(temp) - 1:
            out.append([int(temp[i])])
    data.append(conv)
X_test = torch.tensor(tuple(data), dtype=torch.float) # 3 X 2 tensor
y_test = torch.tensor(tuple(out), dtype=torch.float) # 3 X 1 tensor
test_loader = (X_test, y_test)

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_loader) / batch_size)
num_epochs = int(num_epochs)

'''
STEP 3: CREATE MODEL CLASS
'''
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.tanh(out)
        # Linear function (readout)
        out = self.fc2(out)
        return out
'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 13
hidden_dim = 100
output_dim = 1

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images with gradient accumulation capabilities
                images = images.view(-1, 28*28).requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))