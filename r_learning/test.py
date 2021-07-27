import torch.nn as nn
import torch




i = torch.randn(1, 1, 5,5)

conv1 = nn.Conv2d(1, 1, 3, 1, padding=0)

m = conv1(i)

print(m)