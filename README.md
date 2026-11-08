# Custom NLP Model & BPE Tokenizer from Scratch

A lightweight transformer-based language model and custom Byte-Pair Encoding (BPE) tokenizer built and trained entirely from scratch in PyTorch and Python.

## 📁 Project Structure

- `model.py`: Transformer architecture definition (Multi-Head Attention, FeedForward, blocks).
- `tokenizer.py`: Custom BPE tokenizer implementation.
- `train.py`: Training script with loss logging and validation checkpoints.
- `generate.py`: Text generation and inference script.
- `configs/`: Configuration files for model hyperparameters.
- `tokenizer.json`: Serialized tokenizer vocabulary and merges (768 merges).
- `model_M_best.pt`: Saved model checkpoint weights.

## ⚙️ Model Architecture (`model_M`)

- **Layers (`n_layer`)**: 4[cite: 11]
- **Attention Heads (`n_head`)**: 4[cite: 11]
- **Embedding Dimension (`n_embd`)**: 128[cite: 11]
- **Block Size / Context Length**: 128[cite: 11]
- **Total Parameters**: ~936,192 (~936k)[cite: 11]

## 📈 Training Performance

Trained for 1,000 steps with a peak learning rate of $3 \times 10^{-4}$[cite: 11]:

* **Final Training Loss**: $4.5020$[cite: 11]
* **Best Validation Loss**: $4.6491$[cite: 11]
* **Throughput**: ~190,955 tokens/sec[cite: 11]

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip install torch pandas