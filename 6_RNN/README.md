# RNN tips

A recurrent neural network or RNN is a neural network which maps from an input space of
sequences to an output space of sequences _in a stateful way_. That is, the prediction of output $y_t$
depends not only on the input $x_t$, but also on the hidden state of the system, $h_t$, which gets updated
over time, as the sequence is processed.

When training a language model, we condition on the ground truth labels from the past, not labels generated from the model.
This is called _teacher forcing_, since the teacher’s values are “force fed” into the model as input at
each step.
Unfortunately, teacher forcing can sometimes result in models that perform poorly at test time.
The reason is that the model has only ever been trained on inputs that are “correct”, so it may not
know what to do if, at test time, it encounters an input sequence $w_{1:t-1}$ generated from the previous
step that deviates from what it saw in training.
A common solution to this is known as _scheduled sampling_. This starts off using
teacher forcing, but at random time steps, feeds in samples from the model instead; the fraction of
time this happens is gradually increased.
An alternative solution is to use other kinds of models where MLE training works better, such as
_1d CNNs_ and _transformers_.

Unforunately, the activations in an RNN can decay or explode as we go forwards in time, since
we multiply by the weight matrix $W_{hh}$ at each time step. Similarly, the gradients in an RNN can
decay or explode as we go backwards in time, since we multiply the Jacobians at each time step. A simple heuristic is to use gradient clipping.