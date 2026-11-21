# Build a GPT from Scratch

This project implements a byte-level BPE tokenizer, a decoder-only GPT in PyTorch,
a training pipeline, a smoothed bigram baseline, and autoregressive generation.

## Setup

The training corpus is Tiny Shakespeare. Download it to `input.txt` in the project
root before training:

```text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Install PyTorch, then run from the project root:

```bash
python train.py --config configs/M.json
python bigram.py
```

The training script caches token IDs under `cache/`, writes evaluation metrics to
`logs/<run_id>.csv`, appends completed runs to `runs.csv`, and saves the best
checkpoint under `checkpoints/`. The assignment's required 137-step warmup is
used regardless of the config's `warmup_steps` value.

For the full experiment suite, run the configs on a GPU runtime such as Google
Colab. Dataset, cache, logs, and checkpoints are local/generated files and are
not part of the final submission archive.
