import torch
import torch.nn as nn

m = nn.Linear(5,10)
for pn,n in m.named_parameters():
    print(pn,n)