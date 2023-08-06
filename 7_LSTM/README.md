In PyTorch an LSTM can be defined as:
```
lstm = nn.LSTM(input_size=input_dim,
               hidden_size=hidden_dim,
               num_layers=n_layers,
               bias=True,
               batch_first=False,
               dropout=False,
               bidirectional=False,
               proj_size=0)
```
where
* `input_size` – The number of expected features in the input x
* `hidden_size` – The number of features in the hidden state h
* `num_layers` – Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: `1`
* `bias` – If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`
* `batch_first` – If `True`, then the input and output tensors are provided as (batch, seq, feature) instead of `(seq, batch, feature)`. Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: `False`
* `dropout` – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: `0`
* `bidirectional` – If `True`, becomes a bidirectional LSTM. Default: `False`.
* `proj_size` – If `> 0`, will use LSTM with projections of corresponding size. Default: `0`

<br>Once an LSTM has been defined with input and hidden dimensions, we can call it and retrieve the output and hidden state at every time step.
```
output, (hidden, cell) = lstm(input, (h0, c0))
```

The inputs to an LSTM are `(input, (h0, c0))`.
* `input` = a Tensor containing the values in an input sequence; this has values: `(seq_len, batch, input_size)`
* `h0` = a Tensor containing the initial hidden state for each element in a batch
* `c0` = a Tensor containing the initial cell memory for each element in the batch

<br>`h0` nd `c0` will default to `0`, if they are not specified. Their dimensions are: `(n_layers, batch, hidden_dim)`.

`input`: tensor of shape `(seq_length, batch_size, input_size)` when `batch_first=False` and `(batch_size, seq_length, input_size)` when `batch_first=True`.

`output`: tensor of shape `(seq_length, batch_size, D*H)` when `batch_first=False` and `(batch_size, seq_length, D*H)` when `batch_first=True`. `D=2` if `bidirectional=True`, otherwise 1. `H=proj_size` if `proj_size>0`, otherwise `H=hidden_dim`.