import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import one_hot_encode, get_batches

class CharLSTM(nn.Module):
    
    def __init__(self,
                 tokens,
                 input_size=83,
                 n_hidden=512,
                 n_layers=2,
                 drop_prob=0.5):
        
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # we will assign number to epochs in train_model
        # function and use it in save_model
        self.epochs = None
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=n_hidden,
                            num_layers=n_layers, 
                            dropout=drop_prob,
                            batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, input_size)
        
        self.init_weights()
    
    def forward(self, x, hc):
        
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        
        # Stack up LSTM outputs using view
        x = x.reshape(x.size()[0]*x.size()[1], self.n_hidden)
        x = self.fc(x)
        
        return x, (h, c)
    
    
    def predict(self,
                char,
                h=None,
                cuda=False,
                top_k=None):
        '''
        Given a character, predict the next character.
        Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()
        
        if h is None:
            h = self.init_hidden(1)
        
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, self.input_size)
        inputs = torch.from_numpy(x)

        if cuda:
            inputs = inputs.cuda()
        
        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        if cuda:
            p = p.cpu()
        
        if top_k is None:
            top_ch = np.arange(self.input_size)
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
            
        return self.int2char[char], h
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, n_seqs):
        # Create two new tensors with sizes (n_layers, n_seqs, n_hidden)
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())
    
    def train_model(self,
                    data,
                    optimizer,
                    criterion,
                    epochs=10,
                    n_seqs=10,
                    n_steps=50,
                    clip=5,
                    val_frac=0.1,
                    cuda=False,
                    print_every=10):
        ''' 
            Arguments
            ---------
            data: training data
            n_seqs: Number of mini-sequences per mini-batch, aka batch size
            n_steps: Number of character steps per mini-batch
            clip: gradient clipping
            val_frac: Fraction of data to hold out for validation
            cuda: Train with CUDA on a GPU
            print_every: Number of steps for printing training and validation loss
        '''
        
        self.train()
        self.epochs = epochs
        
        val_idx = int(len(data)*(1-val_frac))
        data, val_data = data[:val_idx], data[val_idx:]
        
        if cuda:
            self.cuda()
        
        counter = 0
        n_chars = self.input_size
        for e in range(self.epochs):
            h = self.init_hidden(n_seqs)
            for x, y in get_batches(data, n_seqs, n_steps):
                counter += 1
                
                # One-hot encode our data and make them Torch tensors
                x = one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                
                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                self.zero_grad()
                
                output, h = self.forward(inputs, h)
                loss = criterion(output, targets.view(n_seqs*n_steps))

                loss.backward()
                
                # `clip_grad_norm` helps prevent the
                # exploding gradient problem in RNNs/LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)

                optimizer.step()
                
                if counter % print_every == 0:
                    
                    val_h = self.init_hidden(n_seqs)
                    val_losses = []
                    for x, y in get_batches(val_data, n_seqs, n_steps):
                        x = one_hot_encode(x, n_chars)
                        x, y = torch.from_numpy(x), torch.from_numpy(y)
                        
                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])
                        
                        inputs, targets = x, y
                        if cuda:
                            inputs, targets = inputs.cuda(), targets.cuda()

                        output, val_h = self.forward(inputs, val_h)
                        val_loss = criterion(output, targets.view(n_seqs*n_steps))
                    
                        val_losses.append(val_loss.item())
                    
                    print("Epoch: {}/{}...".format(e+1, self.epochs),
                        "Step: {}...".format(counter),
                        "Loss: {:.4f}...".format(loss.item()),
                        "Val Loss: {:.4f}".format(np.mean(val_losses)))
                    
    def save_model(self):
        model_name = 'lstm_{}_epoch.net'.format(self.epochs)

        checkpoint = {'n_hidden': self.n_hidden,
                    'n_layers': self.n_layers,
                    'state_dict': self.state_dict(),
                    'tokens': self.chars}

        with open(model_name, 'wb') as f:
            torch.save(checkpoint, f)
    