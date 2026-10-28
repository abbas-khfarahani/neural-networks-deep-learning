# Homework 1 — Threshold neurons, linear units, and a first autoencoder

Questions 1 and 2 were submitted by Abbas Khoshdooni Farahani. Questions 3 and 4 were submitted by Faeze Khanmohammadi.

| Notebook | Question |
| --- | --- |
| [adaline_breast_cancer.ipynb](adaline_breast_cancer.ipynb) | Adaline on the Wisconsin breast-cancer set |
| [madaline_moons_circles.ipynb](madaline_moons_circles.ipynb) | Madaline Rule II on moons and circles |
| [housing_mlp.ipynb](housing_mlp.ipynb) | House-price regression |
| [fashionmnist_autoencoder.ipynb](fashionmnist_autoencoder.ipynb) | Autoencoder and clustering on FashionMNIST |

## 1. McCulloch–Pitts neurons

A threshold neuron computes a weighted sum plus a bias and fires when that sum crosses zero. With real weights and no hidden layer, the decision boundary is a single hyperplane, so the unit can only cut the input plane in two.

AND, OR, and NAND are linearly separable. For each gate the truth table is plotted in the (x1, x2) plane, and weights and a bias are chosen so the hyperplane sits between the 0s and the 1s. The acceptable weights are the solutions of the four inequalities that the truth table imposes.

XOR is not linearly separable. In three-input XOR the 1s and the 0s are not on opposite sides of any single plane, so one threshold neuron cannot implement it. Majority is linearly separable. Parity, like XOR, is not: the labels alternate in a way no half-space can match.

## 2. Adaline and Madaline

### Adaline on breast cancer

Adaline is trained from scratch with mean squared error and a linear activation, then thresholded for classification. Three learning rates were tried: 0.001, 0.005, and 0.02.

The stable small step size finishes at a training MSE of about 14 and a training accuracy of 0.937. The larger rates diverge: at 0.005 the training MSE is about 340. On the test set the three rates score 0.9006, 0.9064, and 0.8947. The selected run (learning rate 0.005 on the reported test comparison) has:

| Metric | Adaline | Logistic regression |
| --- | --- | --- |
| Accuracy | 0.9064 | 0.9591 |
| Precision | 0.9333 | 0.9717 |
| Recall | 0.9159 | 0.9626 |
| F1 | 0.9245 | 0.9671 |
| ROC-AUC | 0.9648 | 0.9956 |

LDA is in the same comparison table in the notebook. Adaline is a reasonable linear baseline, and it loses to logistic regression and LDA because those models are fit to a probabilistic objective on a set that is only approximately linearly separable. Adaline is the better of the three only when the classes really are separated by a wide margin and squared error on the raw targets is a fair loss. On this data the probabilistic linear models use the same features more effectively.

### Madaline on moons and circles

Madaline Rule II trains a hidden layer of Adaline units and combines their half-planes into one decision. The notebook implements the rule, draws the boundary on the moons and circles sets, and varies the number of hidden units and the learning rate. Moons and circles are not linearly separable, so a single Adaline fails and the hidden hyperplanes are what carve the boundary: each hidden unit cuts the plane, and the output unit ORs those cuts together into the final region.

## 3. House-price regression

The California housing table is cleaned in three steps: missing entries are handled, text columns are encoded as numbers, and outliers are removed. The split is 80/20 with `random_state` 42.

A small MLP has 89 parameters. A wider network, trained at a learning rate 100 times smaller (1.0 down to 0.01), fits the prices better. In the reported comparison the complex model reaches about R² 0.71 and MAE about 39,100, against about R² 0.62 and MAE about 45,300 for the simpler run. Both training curves fall, but the small model is noisy. The complex model is smoother, with a mild gap between training and validation loss. Validation loss sitting below training loss is possible here because dropout and the smaller learning rate make the training loss a harder, noisier objective than the evaluation loss.

## 4. FashionMNIST autoencoder

FashionMNIST is 60,000 training images and 10,000 test images, 10 balanced classes, pixels scaled from 0–255 into [0, 1]. Scaling keeps the activations from saturating when values are multiplied through the layers.

The network is a fully connected autoencoder. The encoder compresses each image to 32 features. Those features are clustered with K-means, projected with t-SNE, and compared with the same clustering on the raw 784 pixels.

| Representation | ARI | AMI |
| --- | --- | --- |
| Autoencoder, 32 features | 0.4042 | 0.5764 |
| Raw pixels, 784 features | 0.3684 | 0.5087 |

The compressed codes cluster slightly better than the pixels (about 1.1× on both scores) and clear the targets set for the assignment, ARI at least 0.3 and AMI at least 0.5. Reconstructions and the t-SNE plots are in the notebook.
