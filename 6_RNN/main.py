import numpy as np
import torch
import torch.nn as nn
from data_generator import Data_generator
from rnn import RNN

seq_length = 20
data_gen = Data_generator(seq_length=seq_length)
data, time_steps = data_gen.generate()

rnn_model = RNN(input_size=1,
                output_size=1, # just predicting the next time step
                hidden_dim=10,
                n_layers=2)

test_input = torch.Tensor(data).unsqueeze(0) # adds the batch_size of 1 as first dimension

output, hidden = rnn_model.forward(test_input, None)
data_gen.display(input=data,
                 prediction=output.detach().numpy(),
                 time_steps=time_steps,
                 title='RNN prediction before training!')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(),
                             lr=0.01) 
rnn_model.train(criterion=criterion,
                optimizer=optimizer,
                training_data=data,
                epochs=300,
                print_every=30)

"""
As you see, the overall prediction for first few steps are relatively bad!
This is because the model has no information about the history of the time-series.

Moreover, the prediction for time steps with sharp changes (beginning and and of
the input time-series) are relatively bad.
"""
