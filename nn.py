
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

def trainModel():


	train = open('train_data.txt', 'r')
	lines = train.readlines()
	data = []
	out = []
	for line in lines:
		temp = line.split(",")
		conv = []
		for i in range(len(temp)):
			if i in [1,2,3,4,5,6,7,8,9,10,11,16,17]:
				conv.append(float(temp[i]))
			elif i == len(temp) - 1:
				out.append([int(temp[i])])
		data.append(conv)
	X = torch.tensor(tuple(data), dtype=torch.float) # 3 X 2 tensor
	y = torch.tensor(tuple(out), dtype=torch.float) # 3 X 1 tensor
	xPredicted = torch.tensor((data[0]), dtype=torch.float) # 1 X 2 tensor
	print(X.size(), y.size())
	# Construct our model by instantiating the class defined above
	model = Feedforward(13,15)
	# Construct our loss function and an Optimizer. Training this strange model with
	# vanilla stochastic gradient descent is tough, so we use momentum
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
	model.train()
	epoch = 30000
	for epoch in range(epoch):
	    optimizer.zero_grad()
	    # Forward pass
	    y_pred = model(X)
	    # print(y_pred)
	    # Compute Loss
	    loss = criterion(y_pred.squeeze(), y)
	   
	    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
	    # Backward pass
	    loss.backward()
	    optimizer.step()
	torch.save(model.state_dict(), 'model')


def testing():
	model = Feedforward(13,15)
	model.load_state_dict(torch.load('model'))
	model.eval()
	'''
	#TESTING
	'''
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
	# print(data)
	# print(out)

	correct, total = 0, 0
	for t, ans in zip(data, out):
		test = torch.tensor(tuple(t), dtype=torch.float)
		y_pred = model(test)
		guess = y_pred.detach().numpy()[0]
		if guess >= 0.5: guess = 1
		else: guess = 0
		# print(guess, ans)

		if guess == ans[0]:
			correct += 1.0
		total += 1.0
	print("Testing Accuracy", correct/total)

def percentChance(sounds):
	model = Feedforward(13,15)
	model.load_state_dict(torch.load('model'))
	model.eval()

	test = torch.tensor(tuple(sounds), dtype=torch.float)
	y_pred = model(test)
	percentage = y_pred.detach().numpy()[0]
	return percentage
