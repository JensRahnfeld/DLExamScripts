import torch
import torch.nn as nn
import torch.optim as optim

x = [[1]]
y = [[-3]]

w0 = [[0.5]]
w1 = [[0.5]]
w2 = [[0.5]]
w_skip = [[0.5]]


class MLP(nn.Module):

    def __init__(self, w0, w1, w2, w_skip):
        super().__init__()

        self.fc0 = nn.Linear(1, 1, bias=False)
        self.g0 = nn.ReLU()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.g1 = nn.ReLU()
        self.fc2 = nn.Linear(1, 1, bias=False)
        self.skip = nn.Linear(1, 1, bias=False)

        self.fc0.weight.data = torch.tensor(w0).float().reshape_as(self.fc0.weight.data)
        self.fc1.weight.data = torch.tensor(w1).float().reshape_as(self.fc1.weight.data)
        self.fc2.weight.data = torch.tensor(w2).float().reshape_as(self.fc2.weight.data)
        self.skip.weight.data = torch.tensor(w_skip).float().reshape_as(self.skip.weight.data)

    def forward(self, x):
        z0 = self.fc0(x)
        g0 = self.g0(z0)

        z1 = self.fc1(g0)
        g1 = self.g1(z1)

        y_hat = self.fc2(g1) + self.skip(g0)

        print(f" z0         = {z0}")
        print(f" g0         = {g0}")
        print(f" z1         = {z1}")
        print(f" g1         = {g1}")
        print(f" y_hat      = {y_hat}")

        z0.register_hook(lambda grad: print(f" ∂L/∂z0: {grad}"))
        g0.register_hook(lambda grad: print(f" ∂L/∂g0: {grad}"))
        z1.register_hook(lambda grad: print(f" ∂L/∂z1: {grad}"))
        g1.register_hook(lambda grad: print(f" ∂L/∂g1: {grad}"))
        y_hat.register_hook(lambda grad: print(f" ∂L/∂y_hat: {grad}"))

        return y_hat


mlp = MLP(w0, w1, w2, w_skip)

optimizer = optim.SGD(mlp.parameters(), lr=1.0, momentum=0.0, dampening=0.0, weight_decay=0.0)

# initial weights
print("\Initial weights:")
print(f" w0     = {mlp.fc0.weight.data}")
print(f" w1     = {mlp.fc1.weight.data}")
print(f" w2     = {mlp.fc2.weight.data}")
print(f" w_skip = {mlp.fc2.weight.data}")

# forward pass
optimizer.zero_grad()
x = torch.tensor(x).float()
y = torch.tensor(y).float()
print("\nForward pass:")
print(f" x          = {x}")
pred = mlp(x)
print(f" conv(x)    = {pred}")

# backward pass
loss = torch.sum(torch.abs(pred - y))
print(f" loss       = {loss}")
print("\nBackward Pass:")
loss.backward()

print("\nGradients:")
print(f" ∇w0        = {mlp.fc0.weight.grad}")
print(f" ∇w1        = {mlp.fc1.weight.grad}")
print(f" ∇w2        = {mlp.fc2.weight.grad}")
print(f" ∇w_skip    = {mlp.skip.weight.grad}")

optimizer.step()
print("\nUpdated weights:")
print(f" w0     = {mlp.fc0.weight.data}")
print(f" w1     = {mlp.fc1.weight.data}")
print(f" w2     = {mlp.fc2.weight.data}")
print(f" w_skip = {mlp.skip.weight.data}")
