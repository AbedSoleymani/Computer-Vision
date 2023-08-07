'''
In this notebook, we will train a network character by character on some text,
then using the trained net, we will generate new text character by character.
'''

# %%

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import read_txt, get_batches, sample
from model import CharLSTM

# %%

(encoded,
 int2char,
 char2int,
 text,
 chars) = read_txt(path='books/shakespeare.txt') # anna/shakespeare.txt

print(text[:20])
print(encoded[:20])
print(encoded.shape)
# %%

batches = get_batches(arr=encoded,
                      n_seqs=10,
                      seq_len=50)
x, y = next(batches)

print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])

# %%
charLSTM = CharLSTM(tokens=chars,
                    input_size=len(chars),
                    n_hidden=512,
                    n_layers=2)
print(charLSTM)

# %%

n_seqs, n_steps, epochs = 128, 100, 1
optimizer = torch.optim.Adam(charLSTM.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

charLSTM.train_model(data=encoded,
                     optimizer=optimizer,
                     criterion=criterion,
                     epochs=epochs,
                     n_seqs=n_seqs,
                     n_steps=n_steps)

charLSTM.save_model()

# %%

with open('lstm_1_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded_model = CharLSTM(tokens=checkpoint['tokens'],
                        input_size=len(checkpoint['tokens']),
                        n_hidden=checkpoint['n_hidden'],
                        n_layers=checkpoint['n_layers'])

loaded_model.load_state_dict(checkpoint['state_dict'])

print(sample(net=loaded_model,
             size=1000,
             prime='Abed',
             top_k=5))

# %%
