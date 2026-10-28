# Homework 5 — Re-identification and a discrete diffusion language model

Question 1 was submitted by Faeze Khanmohammadi. Question 2 was submitted by Abbas Khoshdooni Farahani.

| Notebook | Question |
| --- | --- |
| [cattle_reid.ipynb](cattle_reid.ipynb) | Cattle re-identification with ResNet and BoTNet |
| [diffusion_text_to_sql.ipynb](diffusion_text_to_sql.ipynb) | Masked diffusion-style text-to-SQL |

## 1. Cattle re-identification

The set is the cattle collection from the paper: 89 folders, one animal each, plus the usual dataset readme files. The class sizes are badly skewed. A few animals have about 50 photos. Many have a single photo (classes 69 and 72 among them), and a large number have fewer than five.

Images are resized to 384×192 so the frame matches the shape of a cow, then normalized. Folders with fewer than five images are dropped: **32 classes** removed, **57** left, **856** images. The training split is then augmented until each class has about 50 images.

| Split | Images |
| --- | --- |
| Train, before balancing | 684 |
| Train, after balancing | 2,850 |
| Test, left untouched | 172 |

The batch size is 32, so a batch is `[32, 3, 384, 192]`. That is 90 training batches and 6 test batches.

Graph sampling, as used in the paper, builds a graph of visual similarity and draws hard neighbors instead of a random batch. Random pairs of black-and-white cattle are often too easy, and the model can separate them from color alone. Hard negatives force it to use the hide pattern. The submitted training run uses the ordinary 80/20 split plus the class balancing above, and discusses graph sampling as the method the paper prefers for this set.

ResNet50 is trained from scratch with cross-entropy and Adam at learning rate 10⁻⁴. Skip connections are there so a deep stack can be trained at all: the block learns a residual F(x), and x is added back, which keeps the gradient from vanishing. At epoch 1 the test accuracy is 7.56%, which is expected with no pretrained weights. The best test point is epoch 20: test loss **0.1538**, test accuracy **96.51%**. Late in training the training accuracy stays above 99%, and the test accuracy stays in the 90–96% range, so the model is not only memorizing the augmented training images.

BoTNet replaces the deep convolutions with multi-head self-attention inside a ResNet-style bottleneck. The same optimizer and loss, for 30 epochs, go from about 12% test accuracy in the early epochs to about 72% by epoch 10. The best test accuracy is **91.28%** at epoch 29 (training loss 0.1348, training accuracy 95.51%, test loss 0.3113). Attention does learn global pattern, and it does not beat the convolutional baseline on this set.

| Model | Best test accuracy | When |
| --- | --- | --- |
| ResNet50 | 96.51% | Epoch 20 |
| BoTNet | 91.28% | Epoch 29 |

ResNet converges faster and with less oscillation. On class 45, which has 38 images, ResNet scores 87.5% and BoTNet drops to 50%. Attention heatmaps for the BoTNet layer are in the notebook.

## 2. Diffusion-style text-to-SQL

The model is a masked language model (DistilBERT), not an autoregressive decoder. A forward process masks tokens with a schedule in t. The loss is cross-entropy on the masked positions only, with labels set to −100 elsewhere, multiplied by a weight that depends on t. Training uses 5,000 examples, with 500 for development and 500 for test.

Generation is iterative and blockwise, in `masked_iterative_generate`. The prompt is tokenized, a span of mask tokens is appended for the SQL, and that span is split into blocks. At each step the model scores the still-masked positions, and a fraction of the most confident positions is committed. The rest stay masked. A key-value cache does not transfer from left-to-right decoding: earlier tokens are not frozen, so the whole sequence is scored again at every step.

The raw string is not scored directly. `extract_sql_from_text` keeps the text from the first SQL keyword (`SELECT`, `WITH`, `UPDATE`, `DELETE`, `INSERT`) through the first semicolon. Masked models trained on general text often emit a preamble or a second sentence. DistilBERT was not trained as a generator, it has no end-of-sequence token in this loop, and a small model drops or invents clauses. Cutting at the first keyword and the first semicolon is the cleanup used for scoring.

Normalized exact match uppercases the SQL keywords, collapses whitespace, and strips a trailing semicolon, then checks string equality. On the 500 test examples the notebook reports:

```text
Normalized EM on test set: 0.0000 over 500 examples
```

So none of the 500 predictions matched the gold query after normalization. The qualitative errors are the ones this setup produces: a right-looking `SELECT`/`FROM`/`WHERE` with the wrong column or table, a missing filter, a wrong comparison, a broken `GROUP BY` or `ORDER BY`, leftover prose, and SQL that does not parse. One epoch of masked training on a small DistilBERT is not enough to emit exact queries. The notebook still implements the forward mask, the t-weighted loss, the block sampler, and the normalized metric the question asked for.
