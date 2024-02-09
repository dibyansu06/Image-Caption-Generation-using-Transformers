# Image Captioning with Transformers

## Overview

This repository hosts the implementation of an image captioning model using transformer-based architectures. Image captioning is the task of generating textual descriptions for images automatically. The model presented here combines convolutional neural networks (CNNs) for image feature extraction with transformer-based decoders for generating captions.

## Model Architecture

The architecture consists of several key components:

- **CNN Model**: The Convolutional Neural Network is used as a feature extractor. For this project, we experimented with various pre-trained CNN models, including ResNet, VGG, and Inception, to extract high-level features from input images. These features are then passed to the transformer encoder.

- **Transformer Encoder**: The encoder processes the image features obtained from the CNN model. It consists of multiple Transformer Encoder Blocks, each performing multi-head self-attention followed by feed-forward neural network layers. The encoder extracts contextualized representations of the input features.

- **Transformer Decoder**: The decoder takes the encoder outputs along with the tokenized captions as input. It consists of multiple Transformer Decoder Blocks, each performing masked self-attention over the generated tokens and cross-attention over the encoder outputs. The decoder generates captions token by token.

- **Positional Embedding**: To incorporate positional information into the transformer model, positional embeddings are added to the input tokens. This helps the model learn the sequential order of words in the generated captions.

## Usage

1. **Setup**: I recommend to use Google colab for ease of use.

2. **Model Configuration**: Instantiate the CNN model of your choice and configure the transformer encoder and decoder blocks with desired parameters such as embedding dimensions, number of attention heads, etc.

3. **Data Preprocessing**: Prepare your dataset by preprocessing images and their corresponding captions. This may involve resizing images, tokenizing captions, and creating training-validation splits.

4. **Training**: Train the image captioning model using the preprocessed dataset. Fine-tune the CNN model and transformer blocks to optimize performance.

5. **Inference**: Use the trained model to generate captions for new images. Provide an image as input, and the model will output descriptive captions.

## Results

We evaluated the performance of our model primarily on the Flickr30k dataset.

Test Sample 1:

![Sample Caption 1](/results_sc/Screenshot%202023-12-17%20085648.png)

Test Sample 2:

![Sample Caption 2](/results_sc/Screenshot%202023-12-17%20085822.png)

Test Sample 3:

![Sample Caption 3](/results_sc/Screenshot%202023-12-17%20090132.png)

Test Sample 4:

![Sample Caption 4](/results_sc/Screenshot%202023-12-17%20092027.png)

Test Sample 5:

![Sample Caption 5](/results_sc/Screenshot%202023-12-17%20092101.png)

Test Sample 6:

![Sample Caption 6](/results_sc/Screenshot%202023-12-17%20092200.png)

Test sample 7:

![Sample Caption 7](/results_sc/Screenshot%202023-12-17%20093116.png)


## Learning Resources

This project was undertaken as a learning project, drawing inspiration and knowledge from various sources, including:

- Research papers on image captioning and transformer architectures.
- The [Keras Vision Image Captioning](https://keras.io/examples/vision/image_captioning/), which provided valuable insights into model implementation and training.
- "Attention Is All You Need", the seminal paper introducing the transformer architecture, served as a foundational resource.


**Important Note**: Due to a known issue, the model may become unusable after saving and loading it. This could be due to compatibility issues or limitations in the current implementation. I are actively working on resolving this issue, but in the meantime, please be cautious when saving and loading the model. I recommend retraining the model if you encounter any issues with its usability after loading.


---

By combining the power of convolutional neural networks with transformer-based architectures, our image captioning model achieves impressive results in generating descriptive captions for diverse images. I hope this repository serves as a valuable resource for researchers and practitioners in the field of computer vision and natural language processing.
