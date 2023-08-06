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
    output, (hidden, cell) = lstm(Variable(torch.Tensor(input).unsqueeze(1)),
                                  (h0, c0))
    print(output)
    # print(hidden)
    # print(cell)

# %%

"""
A for loop is not very efficient for large sequences of data
So we can also, process all of these inputs at once.

    1. concatenate all our input sequences into one big tensor,
    with a defined batch_size
    2. define the shape of our hidden state
    3. get the outputs and the most recent hidden state
    (created after the last word in the sequence has been seen)
"""

output, (hidden, cell) = lstm(inputs, (h0, c0))
print(output)
# print(hidden)
# print(cell)
