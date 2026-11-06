# Sentiment classification: BPE, log counts and MLP

We classify IMDB movie reviews as *negative* or *positive* and compare:

- **3 representations** of a review over a BPE vocabulary of 5,000 tokens: token indicators, bag of words, and log counts
- **3 architectures:** logistic (log-linear) regression, an MLP with 128 hidden units, and a small MLP with 16 hidden units
- **Evaluation:** 5-fold cross-validation repeated with 3 seeds (0, 3, 5)

| Validation accuracy (%) | indicator | bow | logcounts |
|---|---|---|---|
| **logistic** | 85.56 ± 0.40 | 86.00 ± 0.53 | 86.85 ± 0.30 |
| **mlp** (128) | 87.77 ± 0.29 | 87.94 ± 0.33 | 88.15 ± 0.26 |
| **small-mlp** (16) | 87.77 ± 0.32 | 88.03 ± 0.37 | **88.17 ± 0.26** |

Mean ± std over 15 folds (5 folds × 3 seeds). Slides: [PDF](docs/presentation.pdf) · [PowerPoint](docs/presentation.pptx) · [HTML](docs/presentation.html).

## Contents

```
├── notebooks/
│   └── presentation_experiments.ipynb   the experiments, with all outputs
├── src/
│   ├── experiment.py    BPE tokenizer training, cross-validation splits and training loop
│   ├── loglin.py        logistic regression (scikit-learn)
│   ├── smlp.py          one-hidden-layer MLP on sparse inputs (PyTorch)
│   └── clslib.py        base classes (NamedClasses, Encoder, ...)
├── results/
│   ├── cv_accuracies.json       validation accuracy per system / seed / fold
│   ├── training_history.json    MLP: train/val loss and accuracy per epoch and the epoch kept
│   │                            logistic: final train/val loss and accuracy
│   └── models/                  BPE tokenizer and the 135 trained models (each fold with its history.json)
└── docs/
    ├── presentation.pdf         the slides (PDF)
    ├── presentation.pptx        the slides (PowerPoint)
    └── presentation.html        the slides (HTML, open in a browser)
```

## Setup

We are using python 3.15, here is how you reproduce our setup.

```bash
python3.15 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Then open `notebooks/presentation_experiments.ipynb`. The IMDB dataset ([mteb/imdb](https://huggingface.co/datasets/mteb/imdb))
is downloaded from Hugging Face on the first run. Re-running the training cell retrains all 135 models
and overwrites `results/cv_accuracies.json` and `results/models/` (it does not regenerate `training_history.json`).

### Using `uv`

```bash
uv init --bare .
uv add -r requirements.txt
uv run jupyter lab
```

## License and acknowledgements

The code in `src/` is adapted from
[nlp-illc/nlp1-student](https://github.com/nlp-illc/nlp1-student), released under the Apache License 2.0.
This project is released under the same license; see [`LICENSE`](LICENSE).
