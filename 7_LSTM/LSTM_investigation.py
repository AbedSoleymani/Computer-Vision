# %%

import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

# %%

input_dim = 4
hidden_dim = 3
num_layers = 1
seq_length = 5
batch_size = 1

lstm = nn.LSTM(input_size=input_dim,
               hidden_size=hidden_dim,
               num_layers=num_layers)

# %%

torch.manual_seed(2)
inputs = torch.randn(seq_length,
                     batch_size,
                     input_dim)
print(inputs)
h0 = Variable(torch.randn(num_layers,
                          batch_size,
                          hidden_dim))

c0 = Variable(torch.randn(num_layers,
                          batch_size,
                          hidden_dim))

# %%

for input in inputs:
    print(input.shape)
# for input in inputs:
#     out, hidden = lstm(Variable(input), (h0, c0))
#     print('out: \n', out)
#     print('hidden: \n', hidden)

# %%
