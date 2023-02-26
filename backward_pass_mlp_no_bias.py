import torch
import torch.nn as nn
import torch.optim as optim

x = [[20]]
y = [[-3]]

w0 = [[-0.5]]
w1 = [[-0.5]]
w2 = [[0.5]]


class MLP(nn.Module):

    def __init__(self, w0, w1, w2):
        super().__init__()

        self.fc0 = nn.Linear(1, 1, bias=False)
        self.g0 = nn.PReLU(init=0.1)
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.g1 = nn.PReLU(init=0.1)
        self.fc2 = nn.Linear(1, 1, bias=False)

        self.fc0.weight.data = torch.tensor(w0).float().reshape_as(self.fc0.weight.data)
        self.fc1.weight.data = torch.tensor(w1).float().reshape_as(self.fc1.weight.data)
        self.fc2.weight.data = torch.tensor(w2).float().reshape_as(self.fc2.weight.data)

    def forward(self, x):
        z0 = self.fc0(x)
        g0 = self.g0(z0)

        z1 = self.fc1(g0)
        g1 = self.g1(z1)

        z2 = self.fc2(g1)

        print(f" z0         = {z0}")
        print(f" g0         = {g0}")
        print(f" z1         = {z1}")
        print(f" g1         = {g1}")
        print(f" z2         = {z2}")

        return z2


mlp = MLP(w0, w1, w2)

optimizer = optim.SGD(mlp.parameters(), lr=1.0, momentum=0.0, dampening=0.0, weight_decay=0.0)

# initial weights
print("\Initial weights:")
print(f" w0 = {mlp.fc0.weight.data}")
print(f" w1 = {mlp.fc1.weight.data}")
print(f" w2 = {mlp.fc2.weight.data}")

# forward pass
optimizer.zero_grad()
x = torch.tensor(x).float()
y = torch.tensor(y).float()
print("\nForward pass:")
print(f" x          = {x}")
pred = mlp(x)
print(f" conv(x)    = {pred}")

# backward pass
loss = 0.5 * torch.sum((pred - y)**2)
print(f" loss       = {loss}")
loss.backward()

print("\nGradients:")
print(f" ∇w0 = {mlp.fc0.weight.grad}")
print(f" ∇w1 = {mlp.fc1.weight.grad}")
print(f" ∇w2 = {mlp.fc2.weight.grad}")

optimizer.step()
print("\nUpdated weights:")
print(f" w0 = {mlp.fc0.weight.data}")
print(f" w1 = {mlp.fc1.weight.data}")
print(f" w2 = {mlp.fc2.weight.data}")
