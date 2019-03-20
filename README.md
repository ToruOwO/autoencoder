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

The autoencoder basically tries to learn an approximation to the identity function.
By placing constraints on the network, such as by limiting the number of hidden units,
we can discover interesting structure about the data.
Autoencoder can therefore be seen as a dimensionality reduction technique that removes
noise from the input and helps the network learn more useful features.
(In fact, simple autoencoder often ends up learning a low-dimensional representation very
similar to PCA's.)

**Reference**

*G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality of data with neural networks. Science,
313(5786):504–507, 2006.* [[paper]](https://www.cs.toronto.edu/~hinton/science.pdf)

*I. Goodfellow, Y. Bengio and A. Courville. Deep learning.
MIT Press (www.deeplearningbook.org), 2016.* [[Chapter 14. autoencoder]](https://www.deeplearningbook.org/contents/autoencoders.html)

Variations
----------

* Vanilla autoencoder

* Denoising autoencoder (Vincent et al., 2008)
  * Vincent, H. Larochelle Y. Bengio and P.A. Manzagol, Extracting and Composing Robust Features with Denoising Autoencoders,
  Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML‘08), pages 1096 - 1103, ACM, 2008.
  [[paper]](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
  
  * Idea: since the autoencoder might overfit when there are more network parameters than the number of data points,
  we can modify the network by using a corrupted input with added noises or some of its values masked in a stochastic manner.
  We then train the model to recover the original input.
  
* Sparse autoencoder
  * Idea: if we impose a sparsity constraint on the hidden units of an autoencoder,
  the autoencoder is forced to encode unique statistical features of the data rather than simple
   approximating an identity function.
   
  * See [notes](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) by Andrew Ng

* Variational autoencoder (VAE)

  * Kingma, Diederik P., and Max Welling. “Auto-encoding variational bayes.” arXiv preprint arXiv:1312.6114 (2013).
  [[paper]](http://www.cs.columbia.edu/~blei/seminar/2016_discrete_data/readings/KingmaWelling2013.pdf)
  
  * Different from autoencoder, VAE is trained to learn the *probability distribution* that models
  the input data rather than the identity function that maps input to output. It then *samples* points from
  this distribution and feed them to the decoder to generate new input data samples.
  
  * Implementation techniques
    1. Loss function = reconstruction loss (measures different the reconstructed data is from the original data)
       \+ KL-divergence (a regularizer that encourages the posterior to match the prior)
    2. The reparameterization trick
