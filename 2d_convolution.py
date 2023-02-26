import torch
import torch.nn as nn


conv = nn.Conv2d(3, 10, kernel_size=(5, 5), stride=1, padding=0, bias=False)

# data batch
B = 8
C = 3
H = 32
W = 32

x = torch.ones(B, C, H, W)
# input shape
print(f"Input shape                             : {x.shape}")

# number of trainable parameters
num_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
print(f"Trainable parameters                    : {num_params}")

# number of multiplications
conv.weight.data = conv.weight.data * 0.0 + 1.0  # reset weights to 1.0 for counting
# conv.bias.data = conv.bias.data * 0.0
num_mults = torch.sum(conv(x)).long()
print(f"Number of multiplications for conv(x)   : {num_mults}")

# output shape
out = conv(x)
print(f"Output shape                            : {out.shape}")
