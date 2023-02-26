#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

x = [20]
y = [-3]
w0 = [-0.5]
w1 = [-0.5]
w2 = [0.5]

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Dont forget BIASES IF THERE ARE ANY
        
        self.l1 = nn.Linear(1, 1, bias=False)
        self.l1.weight.data = torch.tensor(w0).float().reshape_as(l1.weight.data)
        self.a1 = nn.PReLU(init=0.1)
        self.l2 = nn.Linear(1, 1, bias=False)
        self.l2.weight.data = torch.tensor(w1).float().reshape_as(l2.weight.data)
        self.a2 = nn.PReLU(init=0.1)
        self.l3 = nn.Linear(1, 1, bias=False)
        self.l3.weight.data = torch.tensor(w2).float().reshape_as(l3.weight.data)
        # a_3 was identity
    
    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        return x

model = Network()


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0)
criterion = nn.L1Loss(reduction='mean')

# data type + shape conversion
x = torch.tensor(x).float()
y = torch.tensor(y).float()

# forward pass
optimizer.zero_grad()
pred = model(x)
print(f"Forward pass:\n model(x) = {pred}")

# backward pass
loss = criterion(pred, y)
loss.backward()
print(f"\n Loss: {loss}")

print("\nGradients:")
print(f" ∇w_0 = {model.l1.weight.grad}")
#print(f" ∇b = {model.bias.grad}")

optimizer.step()
print("\nUpdated weights:")
print(f" w = {model.l1.weight.data}")
#print(f" b = {model.bias.data}")