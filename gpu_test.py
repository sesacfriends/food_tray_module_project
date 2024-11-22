import torch
print(torch.__version__)
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(device)