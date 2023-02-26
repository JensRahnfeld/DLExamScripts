import torch.nn as nn

gru_net = nn.GRU(input_size=256, hidden_size=512, bias=False)
num_params = sum(p.numel() for p in gru_net.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")
