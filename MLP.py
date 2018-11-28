# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:45:41 2018

@author: Muhammad Dawood
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F

X = Variable(torch.Tensor([[0,0], [0,1], [1,0], [1,1]]))
Y = Variable(torch.Tensor([[0], [1], [1], [0]]))
Xt = Variable(torch.Tensor([[0,0], [0,1]]))
Yt = Variable(torch.Tensor([0,1]))


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        input layer have 2 neuron follow by hidden layer with 4 neuron
        output layer have one neuron
        """
        super(Model, self).__init__()
        self.input = torch.nn.Linear(2, 4)
        self.hidden = torch.nn.Linear(4,1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        out = F.relu(self.input(x))
        return F.sigmoid(self.hidden(out))

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X)
#    import pdb; pdb.set_trace()

    # Compute and print loss
    loss = criterion(y_pred, Y)
    print('Epoch: ',epoch, float(loss))

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
prediction = torch.argmax(model(Xt),1)
print('Accuracy is ', float(torch.sum(Yt.long()==prediction.long()))/len(Yt))

