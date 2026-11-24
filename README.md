# Build a GPT from Scratch

This project implements a byte-level BPE tokenizer, a decoder-only GPT in PyTorch,
a training pipeline, a smoothed bigram baseline, and autoregressive generation.

## Setup

The training corpus is Tiny Shakespeare. Download it to `input.txt` in the project
root before training:

```text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Install PyTorch and Matplotlib, then run from the project root:

```bash
pip install torch matplotlib
python train.py --config configs/M.json
python bigram.py
```

The training script caches token IDs under `cache/`, writes evaluation metrics to
`logs/<run_id>.csv`, appends completed runs to `runs.csv`, and saves the best
checkpoint under `checkpoints/`. The assignment's required 137-step warmup is
used regardless of the config's `warmup_steps` value.

For the full experiment suite, run the configs on a GPU runtime such as Google
Colab. `configs/M.json` is the reference run; the remaining configs cover the
scaling and ablation experiments. After running all eight configs, generate the
two experiment figures with:

```bash
for run in S L A1 A2 A3 A4 A5; do python train.py --config "configs/${run}.json"; done
python plot_experiments.py
python generate_samples.py
python benchmark_generation.py
```

Dataset, cache, and checkpoints are generated/local files and are not part of
the final submission archive. Keep the CSV logs and figures for the report.
