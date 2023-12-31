# Transformers
The transformer model is a seq2seq model which uses attention in the encoder as well as
the decoder, thus eliminating the need for RNNs.

### Self attention
We have seen that the decoder of an RNN could use attention to the input sequence in
order to capture contexual embeddings of each input. However, rather than the decoder attending to
the encoder, we can modify the model so the encoder attends to itself. This is called self attention.
The self-attention component helps the encoder comprehend its inputs by focusing on other parts of the input sequence that are relevant to each input element it processes.
At training time, all the outputs are already known, so we can evaluate the
above function in parallel, overcoming the sequential bottleneck of using RNNs.
In addition to improved speed, self-attention can give improved representations of context.<br>
For example, consider translating the English sentences “The animal didn’t cross the street because it was too tired” and “The animal didn’t cross the street because it was too wide” into French. To
generate a pronoun of the correct gender in French, we need to know what “it” refers to (this is called
coreference resolution). In the first case, the word “it” refers to the animal. In the second case,
the word “it” now refers to the street. Apparently, self attention in encoder plays an important role.

**Query, Key, Value**

The self-attention mechanism uses three matrices - query (Q), key (K), and value (V) - to help the system understand and process the relationships between words in a sentence. These three matrices serve distinct purposes:

1. Query (Q): This matrix represents the focus word for which the context is being determined. By transforming the word representation using the query matrix, the system generates a query vector that will be used to compare against other words in the sentence.
2. Key (K): The key matrix is used to create key vectors for all words in the sentence. These key vectors help the system measure the relevance or similarity between the focus word (using the query vector) and other words in the sentence. A higher similarity score between the query vector and a key vector indicates a stronger relationship between the corresponding words.
3. Value (V): The value matrix generates value vectors for all words in the sentence. These vectors hold the contextual information of each word. After calculating the similarity scores using query and key vectors, the system computes a weighted sum of the value vectors. The weights for each value vector are determined by the similarity scores, ensuring that the final contextual representation is influenced more by relevant words.

In summary, the three matrices - query, key, and value - play different roles in the self-attention mechanism. The query matrix helps focus on the word of interest, the key matrix measures relevance between words, and the value matrix provides the context that will be combined to create the final contextual representation of the focus word. Using these three matrices together enables the self-attention mechanism to effectively capture the relationships and dependencies between words in a sentence.

### Multi-headed attention
If we think of an attention matrix as like a kernel matrix, it is natural
to want to use multiple attention matrices, to capture different notions of similarity. This is the
basic idea behind multi-headed attention (MHA).

In MHA, we use layer nomalization instead of batch normalization. This is because BN results in bad performance in transformers. Here are some reasons:

1. **Sequence Length Variability**: In NLP tasks, the length of input sequences can vary. Since BN normalizes across the batch dimension, when the sequence length varies, it can lead to issues. For instance, if sequences are padded to a certain length, the normalization might not be effective for shorter sequences, and vice versa.
2. **Autoregressive Nature of Transformers**: Transformers, especially in autoregressive models where each token is generated one at a time, may not benefit significantly from batch normalization. The statistics used for normalization across the batch dimension might not be consistent across different positions in the sequence. When Batch Normalization (BN) is applied to the input of each layer in a neural network, it normalizes the activations by the mean and standard deviation calculated over the entire batch dimension. In the context of transformers, which are commonly used for natural language processing (NLP) tasks, positional information is typically added to the input embeddings to encode the position of each token in the sequence.
The positional information is crucial for transformers to understand the order and relationships between different tokens. However, when BN is applied to the entire batch, it doesn't differentiate between tokens at different positions. The positional encodings, which are part of the input, are affected by the normalization process.
This can lead to a loss of positional information, as the mean and standard deviation calculated by BN treat positional encodings as just another set of features without considering their special role in encoding position. In essence, BN assumes that the statistics it computes over the batch dimension are equally applicable to all positions in the sequence, which might not be the case when dealing with positional encodings.
To preserve positional information better, layer normalization or other normalization techniques that operate independently on each position in the sequence may be preferred over BN in transformer architectures for NLP tasks. Layer normalization normalizes each position independently, making it more suitable for tasks where positional information is crucial.

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
their lack of relevant inductive bias. For more detail, please see the final discussion.) It was observed that despite heavy regularization effort, ViT overfits the ImageNet (10M images) and performs worse than CNN-based models, but is much better than CNNs on larger datasets (e.g., with 300M images).

**ViT** models (vision transformer), that chops the input up into
16x16 patches, projects each patch into an embedding space, and then passes this set of embeddings
$x_{1:T}$ to a transformer, analogous to the way word embeddings are passed to a transformer. The input
is also prepended with a special [CLASS] embedding, $x_0$. The output of the transformer is a set of
encodings $e_{0:T}$ ; the model maps $e_0$ to the target class label $y$, and is trained in a supervised way.

After supervised pretraining, the model is fine-tuned on various downstream classification tasks,
an approach known as transfer learning. When trained on “small”
datasets such as ImageNet (which has 1k classes and 1.3M images), they find that they cannot
outperform a pretrained CNN ResNet model known as BiT (big transfer).
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

3. **Linear Projection (Embedding):** The flattened patch vectors are passed through a linear projection (also known as an embedding layer) to map them to higher-dimensional representations. This projection aims to transform the pixel values into more informative and expressive embeddings that the model can work with. Moreover, such dimensionality reduction improves the computational and memory complexity of the model and helps the model to become more robust (e.g., against input noise) for the downstream task which benefits the generalization.<br>
<div align="center">
<img src=https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/ab5f782a-00f3-4879-b754-0d3ac3e2cb9d height=350>
<div align="left">

5. **Positional Encodings:** Since ViT doesn't have inherent knowledge of the spatial arrangement of patches, positional encodings are added to the patch embeddings. These positional encodings provide information about the location of each patch within the image and help the model understand the spatial relationships. Please note that unlike NLP transformer that uses sine/cosine basis function which delivers fixed embeddings, positional embeddings in ViT are learnable. In fact, the positional embeddings are a set of vectors for each patch location that get trained with gradient descent along with other parameters. As you can see in the image below, the learned embedding is pretty interesting. During training, the ViT finds out that which patch belongs to which part of the image and which patches are neighbours based on their similarity! Please note that the order of feeding patches to the ViT should remain the same for all inputs in training and test phase.
<div align="center">
<img width="500" alt="image" src="https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/6b16d6e6-fc91-4957-9c2a-9aba0ee2b3f6">
<div align="left">
  
7. **Tokenization:** After obtaining the patch embeddings with positional encodings, these embeddings are treated as tokens and fed into the transformer model. Each patch embedding serves as a token that the self-attention mechanism processes. These tokens are usually the input to the model's self-attention layers.
8. **Self-Attention and Processing:** The self-attention mechanism in the transformer processes the tokenized patch embeddings to capture relationships between different patches. This enables the model to attend to relevant patches and capture context from the entire image.
9. **Layer Stacking and Processing:** Multiple self-attention and feedforward layers are stacked to create a deep hierarchy within the transformer model. Each layer refines and aggregates information from previous layers, allowing the model to learn increasingly abstract features.
10. **Classification Head:** The final token representations output by the self-attention layers are used for classification. A classification head (often a fully connected layer) takes these token representations and produces class predictions based on the learned features.
<div align="center">
<img src=https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/f3d59ede-b0f8-48e5-ac11-15ecc8a9b46f height=350>
<div align="left">

The original Vision Transformer (ViT) architecture, as introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," does not include a separate decoder like the one found in the traditional Transformer model used for natural language processing tasks as the main task of ViT is extracting meaningful features and understaing spatial relationships between different patches of the input image for the downstream tasks.

ViT can be pre-trained with unlabeled data. To this end, we can do:
* Patch Prediction: The model learns to predict missing patches within an image. It masks out some patches and learns to reconstruct them from the surrounding patches. This is similar to predicting missing words in a sentence, a task used in natural language processing.
* Contrastive Learning: This involves creating positive pairs (patches from the same image) and negative pairs (patches from different images) and training the model to bring positive pairs closer together in the embedding space while pushing negative pairs apart. This encourages the model to learn features that are unique to each image while being invariant to common transformations.
* Rotation Prediction: The model learns to predict the rotation applied to an image. This task encourages the model to capture spatial relationships between patches.

Moreover, we can further pre-train ViT by large datasets and transfer/fine-tune the learned model for smaller datasets.
**ATTENTION DISTANCE**
To understand how ViT uses self-attention to integrate information across the image, we can analyze the _average distance_ spanned by attention weights at different layers:
<div align="center">
<img width="350" alt="Screenshot 2023-08-09 at 12 55 25 AM" src="https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/cc623fd8-765d-40bc-b6ef-538fbebf7fa0">
<div align="left">

This “attention distance” is analogous to **receptive field size in CNNs**. Average attention distance is highly variable across heads in lower layers, with some heads attending to much of the image, while others attend to small regions at or near the query location. As depth increases, attention distance increases for all heads. In the second half of the network, most heads attend widely (globally) across tokens.<br>
This result indicate that ViT has the advantage of paying attention to things in the input image that are far a way even at shallow layers of their processing (CNNs do not have this luxury!).

### What's the big deal with transformers and ViT?
Both insist that they do not need RNN/LSTM or CNN in their structure; just MLP! This insistence blocks the inductive bias presented in both RNN/LSTM and CNN. LSTMs/RNNs inherently assume that the order of elements in a sequence matters. This inductive bias helps them learn temporal relationships and patterns within sequences, making them suitable for tasks with sequential inputs. CNNs assume that local patterns (such as edges, textures, and shapes) are important building blocks for recognizing objects. This assumption is embedded in the convolutional and pooling layers, which focus on capturing local features and reducing spatial dimensions. Moreover, CNNs learn to recognize features at different levels of abstraction. Lower layers capture low-level features like edges, while higher layers capture more complex and abstract features. This hierarchical structure mimics the idea that visual information is processed in a similar manner in the human visual system.

Before the advent of transformers, these inductive biases were effective. They helped mitigate limitations stemming from the restricted training data. It was akin to informing these models that while we acknowledge their struggles with learning from relatively small datasets, we would guide them by supplementing with domain-specific knowledge. This approach led us to _biased_ models that showed good performances, but not as good as an unbiased model like the transformer, which was trained on more extensive datasets. The reason behind the fact that transformers requiring larger datasets is their reduced bias compared to RNNs/LSTMs/CNNs. Yes, they are indeed _less_ biased! Transformers remain biased to some extent due to their residual layers.
A residual layer (or residual block) within a neural network architecture can be viewed as an inductive bias since it introduces a specific _structural bias_ that aids in the training of exceedingly deep networks.

Residual layers assume that a "good" transformation should be relatively small. This assumption stems from the notion that, in an ideal scenario, a layer should learn an identity mapping (or a close approximation) if the optimal transformation is not substantially deviating from the identity. Additionally, residual layers provide an implicit prior that encourages the model to learn incremental, fine-tuned alterations to the input data, rather than undergoing drastic changes.

