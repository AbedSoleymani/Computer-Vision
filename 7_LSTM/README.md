In PyTorch an LSTM can be defined as:
```
lstm = nn.LSTM(input_size=input_dim,
               hidden_size=hidden_dim,
               num_layers=n_layers,
               bias=True,
               batch_first=False,
               dropout=False,
               bidirectional=False)
```
where