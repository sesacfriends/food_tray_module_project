import torch
print(torch.__version__)

# windows, linux -> nvidia gpu
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mac
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(device)