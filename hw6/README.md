# Homework 6 — A richer VAE prior, and GAN augmentation

Question 1 was submitted by Abbas Khoshdooni Farahani. Question 2 was submitted by Faeze Khanmohammadi.

| Notebook | Question |
| --- | --- |
| [vampprior_vae.ipynb](vampprior_vae.ipynb) | VAE with a VampPrior on binarized MNIST |
| [gan_data_augmentation.ipynb](gan_data_augmentation.ipynb) | GAN augmentation for a small classifier |

## 1. VAE with a VampPrior

### The two objectives

A standard VAE trains

```text
L = −log p(x | z) + β KL(q(z | x) || p(z))
```

with p(z) = N(0, I) and β = 1. On binarized images the reconstruction term is binary cross-entropy. The KL term keeps the encoder near the prior, which is what makes sampling from N(0, I) meaningful, and it limits overfitting. For a diagonal Gaussian encoder,

```text
KL(q(z | x) || N(0, I)) = −0.5 Σ (1 + log σ² − μ² − σ²)
```

A VampPrior replaces the fixed Gaussian with a mixture of encoder distributions on learned pseudo-inputs u_k:

```text
p(z) = (1 / K) Σ_{k=1}^{K} q(z | u_k)
```

K is 500 in this run. The KL becomes the gap between log q(z | x) and log p(z), and log p(z) is a log-sum-exp over the K components minus log K. That log-sum-exp is what keeps the mixture numerically stable. The prior can now be multimodal, which is the capacity the fixed Gaussian does not have. The same flexibility is what reduces posterior collapse and pulls digit classes apart in the latent space.

### Setup

MNIST, 28×28, binarized at 0.5, so each pixel is 0 or 1 and binary cross-entropy is the right reconstruction loss. 60,000 training images, 10,000 test images, batch size 128. The low-dimensional run uses a latent size of 2, one hidden layer of width 400, Adam at learning rate 0.001, and 50 epochs. Both models’ total loss and reconstruction loss fall. The VampPrior ends lower. Its KL curve is different because the prior is moving too.

### What the 2-D space looks like

The standard VAE packs almost every digit into one cloud around the origin, roughly −5 to 3.5 on the first axis and −4.5 to 3.5 on the second. Digits 2 through 9 overlap heavily. Digit 1 sits a little up and to the left, digit 0 a little up and to the right, and neither is cleanly separate. Prior samples are a round cloud on [−4, 4]². Posterior samples are a tighter cloud on the same center, and the contour plot is two sets of circles.

The VampPrior spreads the codes out and separates several digits. Digit 0 forms a compact cluster up and to the left. Digit 1 forms a large cluster down and to the right, around latent coordinates (10 to 20, −10 to −15). Digits 2 and 3 form smaller clusters near (2, 0) and (5, 0). Digits 4 through 9 still touch, but they break into smaller groups instead of one pile. The prior contours are elongated, with vertical lines at the component means. That is the mixture, not a single Gaussian.

At higher latent sizes (10, 20, 50) reconstruction loss drops for both models, and the VampPrior is still ahead, by less than in two dimensions. Collapse becomes the thing to watch. In the standard VAE one reported latent axis has mean log σ² of about −6.06, so the variance is about 0.0038. That axis is effectively unused. The VampPrior’s learned prior is what keeps more axes alive.

Pseudo-inputs, a PCA walk through the latent space, and a log-likelihood comparison are in the notebook.

## 2. GAN augmentation

The generator and discriminator are convolutional, with self-attention and channel attention, spectral normalization, and a projection discriminator for the class label. The loss is WGAN-GP plus a drift penalty. A classifier is then trained twice: once on the original training set, and once on that set plus images from the trained GAN.

On the original data, 15 epochs:

| Epoch | Train accuracy | Test accuracy |
| --- | --- | --- |
| 1 | 43.31% | 60.60% |
| 8 | 77.77% | 77.93% |
| 15 | 82.23% | 80.87% |

With GAN images added, the same classifier starts higher and finishes higher:

| Epoch | Train accuracy | Test accuracy |
| --- | --- | --- |
| 1 | 66.95% | 75.87% |
| 9 | 88.21% | 84.40% |
| 15 | 91.46% | 86.00% |

Final test accuracy is **80.87%** without the synthetic images and **86.00%** with them. The training accuracy also rises, from 82.23% to 91.46%, so part of the gain is simply more gradient steps on a larger set. The test gap of about five points is the part that counts. Loss curves and samples through training are in the notebook. The plotting cell was run after the fact from saved logs, which is noted in the notebook.
