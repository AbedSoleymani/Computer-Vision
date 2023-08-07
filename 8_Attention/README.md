# Attention

In all of the neural networks we have considered so far (CNN, 1D CNN, RNN, and LSTM), the hidden activations are a linear combination of the input activations, followed by a nonlinearity and a fixed set of weights that are learned on a training set.
However, we can imagine a more flexible model in which we have a set of $m$ feature vectors or
values $V$ and the model dynamically decides (in an input dependenent way) which one to
use, based on how similar the input query vector $q$ is to a set of $m$ keys $K$.
If $q$ is most similar to key $i$, then we use value $v_i$. This is the basic idea behind _attention mechanisms_.

We can think of attention as a _dictionary lookup_, in which we compare the query $q$ to each key $k_i$,
and then retrieve the corresponding value $v_i$. To make this lookup operation _differentiable_, instead
of retrieving a single value $v_i$, we compute a convex combination of the values.
The attention weights can be computed from an attention score function, that
computes the similarity of query $q$ to key $k_i$.
Given the scores, we can compute the attention weights using the _softmax_ function.

## Soft vs Hard attention
If we force the attention heatmap to be sparse, so that each output can only attend to one input
location instead of a weighted combination of all of them, the method is called hard attention. We
compare these two approaches for an image captioning problem in the image below:
<img width="1368" alt="Screenshot 2023-08-07 at 1 47 44 PM" src="https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/49679d7d-6821-4de9-becf-d5fe68c0437e">
It seems from the above examples that these attention heatmaps can “explain” why the model
generates a given output. However, the interpretability of attention is controversial.
Unfortunately, hard attention results in a nondifferentiable training objective, and requires methods such as reinforcement learning to fit the model.
