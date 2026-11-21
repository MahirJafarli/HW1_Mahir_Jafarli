# hw1-build: caspian-7
import csv
import math
import os

from tokenizer import BPETokenizer


RUNS_COLUMNS = [
    "run_id", "n_layer", "n_head", "n_embd", "block_size",
    "max_lr", "max_steps", "ablation", "params",
    "best_val_loss", "final_train_loss", "tokens_per_sec", "wall_time_min",
]


def main():
    with open("input.txt", "r", encoding="utf-8") as dataset_file:
        text = dataset_file.read()
    split_idx = int(0.9 * len(text))
    train_text, val_text = text[:split_idx], text[split_idx:]
    if not train_text or not val_text:
        raise ValueError("input.txt must have non-empty training and validation splits.")

    tokenizer_path = "tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, vocab_size=1024)
        tokenizer.save(tokenizer_path)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    vocab_size = len(tokenizer.vocab)

    counts = [[1.0] * vocab_size for _ in range(vocab_size)]
    row_totals = [float(vocab_size)] * vocab_size
    for previous, current in zip(train_ids[:-1], train_ids[1:]):
        counts[previous][current] += 1.0
        row_totals[previous] += 1.0

    val_pairs = list(zip(val_ids[:-1], val_ids[1:]))
    if not val_pairs:
        raise ValueError("The validation split must contain at least two tokens.")

    total_val_loss = 0.0
    for previous, current in val_pairs:
        probability = counts[previous][current] / row_totals[previous]
        total_val_loss -= math.log(probability)
    val_loss = total_val_loss / len(val_pairs)
    print(f"Bigram baseline validation loss: {val_loss:.4f} nats per token")

    runs_file = "runs.csv"
    write_header = not os.path.exists(runs_file) or os.path.getsize(runs_file) == 0
    with open(runs_file, "a", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        if write_header:
            writer.writerow(RUNS_COLUMNS)
        writer.writerow(
            ["bigram", 0, 0, 0, 0, 0, 0, "none", 0,
             f"{val_loss:.4f}", f"{val_loss:.4f}", 0, 0]
        )


if __name__ == "__main__":
    main()
