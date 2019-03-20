Autoencoder
============

Basic ideas
-----------

**What is an autoencoder?**

An autoencoder is a type of neural network that learns a latent representation of the input data.
It is consisted of two components:

1. An *encoder* that compresses the input into a latent-space representation;
2. A *decoder* that reconstruct the input from the latent-space representation.

**Why do we want such a representation?**

We define the network architecture such that the latent representation has a smaller dimension than the original input.
This dimensionality reduction technique removes noise from the input signal and help the network learn more useful features.


*G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality of data with neural networks. Science,
313(5786):504–507, 2006.*

Variations
----------

* Vanilla autoencoder

* Denoising autoencoder (Vincent et al., 2008)
  * Vincent, H. Larochelle Y. Bengio and P.A. Manzagol, Extracting and Composing Robust Features with Denoising Autoencoders,
  Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML‘08), pages 1096 - 1103, ACM, 2008.
  
  * Idea: since the autoencoder might overfit when there are more network parameters than the number of data points,
  we can modify the network by using a corrupted input with added noises or some of its values masked in a stochastic manner.
  We then train the model to recover the original input.
  
* Sparse autoencoder

* Variational autoencoder (VAE)