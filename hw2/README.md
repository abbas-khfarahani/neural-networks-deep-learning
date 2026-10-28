# Homework 2 — Image classification

Question 1 was submitted by Faeze Khanmohammadi. Question 2 was submitted by Abbas Khoshdooni Farahani.

| Notebook | Question |
| --- | --- |
| [rice_leaf_disease.ipynb](rice_leaf_disease.ipynb) | Rice-leaf disease, AlexNet and INC-VGG |
| [remote_sensing_scenes.ipynb](remote_sensing_scenes.ipynb) | Remote-sensing scene classification |

## 1. Rice-leaf disease

The images come from the Kaggle rice-disease set and fall into six classes, including brown spot, healthy leaf, leaf blast, and leaf scald. A fixed seed keeps the runs comparable. Some of the publisher’s own augmented images were filled by copying the nearest pixel after a shift or a small rotation, which stretches the borders. Those copies are easy for a person to spot and they are a poor match for a real photograph, so the training augmentation used here is separate from that preprocessing.

AlexNet is implemented from scratch. On the test set it reaches **90.98%** accuracy. Per-class accuracy on the reported table is 0.9253 for brown spot, 0.9923 for healthy leaf, 0.9407 for leaf blast, and 0.9613 for leaf scald. Healthy leaves are the easy class: the model essentially does not miss them.

INC-VGG follows the paper’s mix of an Inception-style block, a VGG-style stack, and a Swish activation, with dropout 0.3, SGD with momentum 0.9, and a learning rate of 10⁻⁴ reduced by a plateau schedule. Test accuracy is **96.13%**. The same four classes score 0.9820, 1.0000, 0.9639, and 0.9768. Healthy leaf is perfect on that table, and the other three move up with the deeper model. Confusion matrices for both networks are in the notebook.

## 2. Remote-sensing scenes

The second question classifies overhead scenes with an Xception backbone from `timm`. Three controls are in the notebook: augmentation, the optimizer, and partial fine-tuning.

Without augmentation the validation accuracy after 10 epochs is **0.7667**. With augmentation it is **0.9095**. The gain is mostly early: the unaugmented run is still at 0.60 training accuracy when the augmented run has already generalized.

An MLP head trained with Adagrad, backbone frozen, finishes at validation accuracy **0.9095**. The same head with Adam finishes at **0.8976**. Adam fits the training set faster (training accuracy 0.96 by epoch 10) and does not beat Adagrad on the validation set in this setup.

Unfreezing part of the backbone and fine-tuning with Adam for five more epochs is the best of the three. Validation accuracy moves from 0.9095 at epoch 1 to **0.9310** at epoch 5, against 0.9095 for the frozen Adagrad baseline. Training that stage takes about four minutes on the Colab GPU used for the run.
