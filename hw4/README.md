# Homework 4 — Sequences

Question 1 was submitted by Faeze Khanmohammadi. Question 2 was submitted by Abbas Khoshdooni Farahani.

| Notebook | Question |
| --- | --- |
| [atis_intent_slot.ipynb](atis_intent_slot.ipynb) | ATIS intent detection and slot filling |
| [stock_forecasting.ipynb](stock_forecasting.ipynb) | Stock-price forecasting |

## 1. ATIS intent and slot filling

The ATIS flight-query set is loaded and split so that 10% of the training data is validation.

| Split | Sentences |
| --- | --- |
| Train | 3,846 |
| Validation | 428 |
| Test | 586 |

Each row has an id, tokens, slot labels, and an intent. The first training sentence is “what flights are available from boston to denver on wednesday in early morning”, with intent `atis_flight` and slot tags such as `B-fromloc.city_name` and `B-toloc.city_name`.

Tokenization is a whitespace split. Three vocabularies are built, and `<PAD>` and `<UNK>` are added to each.

| Vocabulary | Size |
| --- | --- |
| Words | 837 |
| Slots | 102 |
| Intents | 18 |

`<PAD>` makes the sentences in a batch the same length so they fit in one tensor. `<UNK>` is the id for any word that appears in validation or test and was never seen in training. Padding is done inside `collate_fn`, to the longest sentence in the batch rather than the longest sentence in the corpus. Two batches then have different lengths. That wastes less computation than padding every sentence to the global maximum.

Three models are trained.

| Model | Intent accuracy (test) | Slot F1 |
| --- | --- | --- |
| Separate base models, epoch 10 | 96.05% | 94.64% |
| Joint BiLSTM | 97.10% | 93.26% |
| Encoder–decoder, non-aligned | 97.61% | 66.09% |

The base intent model is at about 94.05% validation accuracy by epoch 10, and the base slot model at about 98.48%. Frequent slots such as `fromloc.city_name` and `toloc.city_name` score above 98 F1. The joint model sums the two losses with equal weight. Its validation intent accuracy reaches 96.13% and its slot accuracy 98.10%. The encoder–decoder starts at a loss of 2.704 and ends at 0.468. Intent stays high, at 97.61%, but slot F1 falls to 66.09% because the slots are no longer aligned token by token with the input. The per-slot `seqeval` tables are in the notebook.

## 2. Stock-price forecasting

The series is differenced so that the mean and variance do not drift, then standardized with a z-score, z = (x − μ) / σ. The scaler is fit on the training window only. Fitting it on the whole series would leak future prices into the training statistics. After scaling, the training mean is about 0 and the training standard deviation is about 1. Validation and test moments differ slightly, which is what you want if the split is honest.

Min-max scaling was not used. After differencing, a few large moves would stretch the whole series toward the ends of [0, 1]. A z-score keeps those moves in units of standard deviation.

The input window is 20 steps and the model forecasts 10 steps of 4 series. Three architectures are compared.

The baseline is an MLP (a NARMAX-style feedforward net). Its validation loss stays under 0.07 on the reported run. Plots for 2024–2025 are in the notebook.

The LSTM model is a 1-D CNN followed by a two-layer LSTM. The convolution stack maps 1 channel to 64 and then to 128, with kernel size 3, batch norm, and dropout 0.2. The LSTM has hidden size 128. A linear head maps 128 to 64 to 40, which is 10 steps times 4 series. The network has **300,520** parameters.

The GRU model uses the same front end and three GRU layers so the parameter count stays in the same range: **333,544** parameters. On some epochs its validation loss is under 0.02.

In both recurrent models the head reads the last time step of the recurrent output, `out[:, -1, :]`, not the raw final hidden state. That vector has already mixed every step of the window, which is what a 10-step forecast needs. Using only the hidden state drops part of that context.

A CNN-only network is included as a bonus, with no recurrence. Multi-step forecasts still accumulate error: a 1% mistake on the first day compounds, and a 100-day rollout is not a serious price. The notebooks show the short-horizon comparison rather than a long free run.
