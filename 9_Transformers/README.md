# Transformers
The transformer model is a seq2seq model which uses attention in the encoder as well as
the decoder, thus eliminating the need for RNNs.

### Self attention
We have seen that the decoder of an RNN could use attention to the input sequence in
order to capture contexual embeddings of each input. However, rather than the decoder attending to
the encoder, we can modify the model so the encoder attends to itself. This is called self attention.
The self-ttention component helps the encoder comprehend its inputs by focusing on other parts of the input sequence that are relevant to each input element it processes.
At training time, all the outputs are already known, so we can evaluate the
above function in parallel, overcoming the sequential bottleneck of using RNNs.
In addition to improved speed, self-attention can give improved representations of context.<br>
For example, consider translating the English sentences “The animal didn’t cross the street because it was too tired” and “The animal didn’t cross the street because it was too wide” into French. To
generate a pronoun of the correct gender in French, we need to know what “it” refers to (this is called
coreference resolution). In the first case, the word “it” refers to the animal. In the second case,
the word “it” now refers to the street. Apparently, self attention in encoder plays an important role.

### Multi-headed attention
If we think of an attention matrix as like a kernel matrix (as discussed in Section 15.4.2), it is natural
to want to use multiple attention matrices, to capture different notions of similarity. This is the
basic idea behind multi-headed attention (MHA).

### Positional encoding
The performance of “vanilla” self-attention can be low, since attention is _permutation invariant_, and
hence ignores the input word ordering. To overcome this, we can concatenate the word embeddings
with a positional embedding, so that the model knows what order the words occur in.<br>
One way to do this is to represent each position by an integer. However, neural networks cannot
natively handle integers. To overcome this, we can encode the integer in binary form.

For example, if we assume the sequence length is `n = 3`, we get the following sequence of `d = 3-dimensional bit` vectors for each location: `000, 001, 010, 011, 100, 101, 110, 111`. We see that the right most index
toggles the fastest (has highest frequency), whereas the left most index (most significant bit) toggles
the slowest.

We can think of the above representation as using a set of _basis functions_ (corresponding to powers
of 2), where the coefficients are 0 or 1. We can obtain a more compact code by using a different set
of basis functions, and real-valued weights. The original paper proposed to use a sinusoidal basis.
In this case, the left-most columns toggle fastest. We see that each row has a real-valued “fingerprint” representing its location in the sequence.
The advantage of this representation is two-fold. First, it can be computed for arbitrary length
inputs (up to $T\leq C$), unlike a learned mapping from integers to vectors. Second, the representation
of one location is linearly predictable from any other, given knowledge of their relative distance.

### Putting it all together
The overall encoder is defined by applying positional encoding to the embedding of the input sequence,
following by $N$ copies of the encoder block, where $N$ controls the depth of the block.<br>
The decoder has a somewhat more complex structure. It is given access to the encoder via
another multi-head attention block. But it is also given access to previously generated outputs: these
are shifted, and then combined with a positional embedding, and then fed into a masked (causal)
multi-head attention model. Finally the output distribution over tokens at each location is computed
in parallel.

During training time, all the inputs X to the decoder are known in advance, since they are derived
from embedding the lagged target output sequence. During inference (test) time, we need to decode
sequentially, and use masked attention, where we feed the generated output into the embedding
layer, and add it to the set of keys/values that can be attended to.

### Comparing transformers, CNNs and RNNs/LSTMs


