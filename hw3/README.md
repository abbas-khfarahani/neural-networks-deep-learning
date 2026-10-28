# Homework 3 — Segmentation and detection

Question 1 was submitted by Faeze Khanmohammadi. Question 2 was submitted by Abbas Khoshdooni Farahani.

| Notebook | Question |
| --- | --- |
| [brain_tissue_segmentation.ipynb](brain_tissue_segmentation.ipynb) | Brain-tissue segmentation |
| [vehicle_detection.ipynb](vehicle_detection.ipynb) | Vehicle detection with an improved Fast R-CNN |

## 1. Brain-tissue segmentation

Volumes are NIfTI MRI scans with matching segmentation masks, loaded with nibabel. Slices are taken in all three anatomical planes, axial, sagittal, and coronal, instead of one view. The classes are background, cerebrospinal fluid (CSF), gray matter, and white matter.

The notebook implements an M-Net, the U-Net variant with extra side paths into the decoder, and reports the parameter count. Training uses LeCun weight initialization. Before training, a single forward pass scores every class near chance, Dice about 0.24 to 0.25.

After 10 epochs the validation Dice scores are:

| Class | Dice |
| --- | --- |
| CSF | 0.7346 |
| Gray matter | 0.8723 |
| White matter | 0.8106 |

Training loss ends at 0.0356 and validation loss at 0.0429. Gray matter is the easiest of the three tissues. CSF stays the weak class: it is a thin, variable boundary, and the Dice gap against gray matter is still about 0.14 at the last epoch. The best CSF Dice during the run was 0.7543, at epoch 8, before a small drop. Reconstructed slices are in the notebook.

## 2. Vehicle detection

The detector is an improved Fast R-CNN. Training uses SGD with momentum 0.9. The learning rate is 10⁻⁴. Proposals are filtered with a score threshold of 0.5, and non-maximum suppression uses an IoU threshold of 0.5.

Validation mAP at IoU 0.5 climbs through training:

| Checkpoint | mAP@50 |
| --- | --- |
| Early (loss 0.124) | 13.84% |
| Mid (loss 0.086) | 18.90% |
| Later | 27.68% |
| Epoch 30 | 28.54% |

The best validation mAP@50 is **28.54%**, at epoch 30, with validation loss 0.0767. On the held-out test set, mAP@50 is **23.24%**. That is below the 60% mark the assignment asked for. The loss is small and stable, so the network is fitting the training objective, but the box-level metric has not caught up: localization and the proposal stage are still the limit, not the classification loss.
