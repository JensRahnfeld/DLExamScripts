import torch
import torch.nn as nn
import torch.optim as optim


x = [[[1, 2, 3, 4]]]
y = [[[1, 0]]]
w = [1, 2]
b = 1

conv = nn.Conv1d(1, 1, kernel_size=2, stride=2, padding=0, bias=True)

optimizer = optim.SGD(conv.parameters(), lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0)

# initial weights
print("\Initial weights:")
print(f" w0 = {conv.weight.data}")
print(f" b0 = {conv.bias.data}")

# data type + shape conversion
conv.weight.data = torch.tensor(w).float().reshape_as(conv.weight.data)
conv.bias.data = torch.tensor(b).float().reshape_as(conv.bias.data)
x = torch.tensor(x).float()
y = torch.tensor(y).float()

# forward pass
optimizer.zero_grad()
pred = conv(x)
print("\nForward pass:")
print(f" x          = {x}")
print(f" conv(x)    = {pred}")

# backward pass
loss = 0.5 * torch.sum((pred - y)**2)
print(f" loss       = {loss}")
loss.backward()

print("\nGradients:")
print(f" ∇w = {conv.weight.grad}")
print(f" ∇b = {conv.bias.grad}")

optimizer.step()
print("\nUpdated weights:")
print(f" w = {conv.weight.data}")
print(f" b = {conv.bias.data}")
