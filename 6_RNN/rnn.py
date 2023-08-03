import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_dim,
                 n_layers):
        
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        """
        batch_first: If True, then the input and output tensors are provided
        as (batch, seq, feature) instead of (seq, batch, feature).
        Default: False
        """
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          nonlinearity='tanh',
                          bias=True,
                          batch_first=True,
                          dropout=0)
        
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=output_size)
        
    def forward(self, x, hidden):
        # x.shape = (batch_size, seq_length, input_size)
        # hidden.shape = (n_layers, batch_size, hidden_dim)
        # r_out.shape = (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)  
        output = self.fc(r_out)
        
        return output, hidden
    
    def train(self,
              criterion,
              optimizer,
              training_data,
              epochs=100,
              print_every=20):
        
        hidden = None  # initializing the hidden state 
        
        for epoch in range(epochs):
            input = training_data[:-1]
            target = training_data[1:]
            
            input_tensor = torch.Tensor(input).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
            target_tensor = torch.Tensor(target)

            prediction, hidden = self.forward(input_tensor, hidden)

            ## Representing Memory ##
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data

            loss = criterion(prediction, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # display loss and predictions
            time_steps = np.linspace(0, np.pi, len(input))
            if epoch%print_every == print_every-1:        
                print('Epoch {}, Loss: {}'.format(epoch+1, loss.item()))
                plt.plot(time_steps,
                         input,
                         'r.',
                         label='input')
                
                plt.plot(time_steps,
                         target,
                         'g*',
                         label='target')
                
                plt.plot(time_steps,
                         prediction.data.numpy().flatten(),
                         'b.',
                         label='prediction')
                
                plt.legend(loc='best')
                plt.title('RNN prediction after {} epochs.'.format(epoch+1))
                plt.show()
