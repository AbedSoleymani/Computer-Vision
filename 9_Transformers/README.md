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
<div align="center">
<img width="712" alt="Screenshot 2023-08-07 at 6 11 12 PM" src="https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/e8ff4b44-7f4f-4b50-a2c5-9e8b783ed9d3">
<div align="left">

### Transformers for images
CNNs are the most common model type for processing image data, since they have
**useful built-in inductive bias**, such as `locality` (due to small kernels), `equivariance` (due to weight
tying), and `invariance` (due to pooling). Suprisingly, it has been found that transformers can also do
well at image classification, at least if trained on enough data. (They need a lot of data to overcome
their lack of relevant inductive bias.)

**ViT** models (vision transformer), that chops the input up into
16x16 patches, projects each patch into an embedding space, and then passes this set of embeddings
$x_{1:T}$ to a transformer, analogous to the way word embeddings are passed to a transformer. The input
is also prepended with a special [CLASS] embedding, $x_0$. The output of the transformer is a set of
encodings $e_{0:T}$ ; the model maps e0 to the target class label $y$, and is trained in a supervised way.

After supervised pretraining, the model is fine-tuned on various downstream classification tasks,
an approach known as transfer learning. When trained on “small”
datasets such as ImageNet (which has 1k classes and 1.3M images), they find that they cannot
outperform a pretrained CNN ResNet model known as BiT (big transfer) [Kol+20].
However, when trained on larger datasets, such as ImageNet-21k (with 21k classes and 14M images),
or the Google-internal JFT dataset (with 18k classes and 303M images), they find that ViT does
better than BiT at transfer learning. It is also cheaper to train than ResNet at this scale. (However,
training is still expensive: the large ViT model on ImageNet-21k takes 30 days on a Google Cloud
TPUv3 with 8 cores!)

Self-attention ViT architecture is used to capture relationships between different patches (i.e., non-overlapping unified-length square parts of the input image) of an image and enable the model to make better sense of the input image and focus on relevant regions during image analysis. Self-attention is a key component of the Transformer's ability to capture both _local_ and _global_ dependencies between patches.

Here are steps in ViT:
1. **Flattening Patches:** Each image is divided into non-overlapping patches, often represented as square regions. Each patch is then flattened into a one-dimensional vector by concatenating the pixel values along rows or columns.<br>
<div align="center">
<img src=https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/0277215c-ca11-42c7-b3ba-153bc15a4c43 height=350>
<div align="left">

3. **Linear Projection (Embedding):** The flattened patch vectors are passed through a linear projection (also known as an embedding layer) to map them to higher-dimensional representations. This projection aims to transform the pixel values into more informative and expressive embeddings that the model can work with.<br>
<div align="center">
<img src=https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/ab5f782a-00f3-4879-b754-0d3ac3e2cb9d height=350>
<div align="left">

5. **Positional Encodings:** Since ViT doesn't have inherent knowledge of the spatial arrangement of patches, positional encodings are added to the patch embeddings. These positional encodings provide information about the location of each patch within the image and help the model understand the spatial relationships.
6. **Tokenization:** After obtaining the patch embeddings with positional encodings, these embeddings are treated as tokens and fed into the transformer model. Each patch embedding serves as a token that the self-attention mechanism processes. These tokens are usually the input to the model's self-attention layers.
7. **Self-Attention and Processing:** The self-attention mechanism in the transformer processes the tokenized patch embeddings to capture relationships between different patches. This enables the model to attend to relevant patches and capture context from the entire image.
8. **Layer Stacking and Processing:** Multiple self-attention and feedforward layers are stacked to create a deep hierarchy within the transformer model. Each layer refines and aggregates information from previous layers, allowing the model to learn increasingly abstract features.
9. **Classification Head:** The final token representations output by the self-attention layers are used for classification. A classification head (often a fully connected layer) takes these token representations and produces class predictions based on the learned features.
<div align="center">
<img src=https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/f3d59ede-b0f8-48e5-ac11-15ecc8a9b46f height=350>
<div align="left">

The original Vision Transformer (ViT) architecture, as introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," does not include a separate decoder like the one found in the traditional Transformer model used for natural language processing tasks as the main task of ViT is extracting meaningful features and understaing spatial relationships between different patches of the input image for the downstream tasks.
