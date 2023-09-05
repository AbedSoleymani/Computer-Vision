# ViT
## How run the code
please provide a description about how to run
1. run `transfer_learning_verification.py` to creat the model and load the pre-trained weights based on HuggingFace transformers
2. run `test.py` to validate the accuracy and generalization of your model. You can download and save other images in `imgs` folder and check them as well.

## Data normalization in ViT
One may wonder that why we use layer normalization in ViT, not batch normalization which is a very common choice in deep computer vision applications.
While both normalization techniques help in improving the training stability and convergence of the model, here we will provide a few reasons why layer normalization is often favored over batch normalization in visual transformers:

1. Autoregressive Nature: Visual transformers often operate in an autoregressive manner, where each position attends to previous positions. Batch normalization relies on statistics computed over the entire batch, which can lead to information leakage from future positions during autoregressive decoding. Layer normalization is more suitable in this case because it normalizes each position independently.
2. Variable Sequence Lengths: Visual transformers often process sequences of varying lengths, such as in image captioning tasks. Batch normalization requires fixed batch sizes, which can be challenging to handle when sequences have different lengths. Layer normalization can be applied independently to each position, making it more flexible for varying sequence lengths.
3. Positional Encoding: Visual transformers typically incorporate positional encodings to provide position information to the model. Batch normalization may interfere with the positional encoding information, while layer normalization does not.
4. Stability and Convergence: Layer normalization is better suited for the self-attention mechanism's stability and convergence properties, as it normalizes the activations along the feature dimension rather than across the batch.
5. Memory Efficiency: Layer normalization requires less memory compared to batch normalization, as it does not need to store statistics for the entire batch.
6. Generalization: Layer normalization's per-position normalization can lead to better generalization across different examples in the batch and across different tasks.

## Class token in ViT?
In ViTs, the "class token" is a special token that represents the entire image as a whole. It's a vector that is added to the sequence of patch embeddings before being fed into the transformer encoder. The class token is typically used to provide a global context about the image and allow the transformer to attend to it during the self-attention mechanism.

When processing images using ViTs, the image is divided into patches, and each patch is embedded into a vector. These patch embeddings are treated as the input sequence for the transformer encoder. However, the transformer doesn't have a direct way to capture global information about the entire image. To address this, a class token is added to the sequence, which represents a summary or aggregation of the entire image content.

The class token is usually initialized as a learnable parameter and is concatenated to the sequence of patch embeddings. It allows the self-attention mechanism to attend to the global context while processing the patch-level features. This global context is important for tasks like image classification, where the model needs to make a decision based on the entire image.

In summary, the class token in Vision Transformers is a mechanism to provide the transformer model with a representation of the entire image so that it can capture global context and make informed decisions during the self-attention process.