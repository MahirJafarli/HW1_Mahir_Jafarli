import os
import csv
from tokenizer import BPETokenizer

def main():
    # 1. Load and read the dataset
    if not os.path.exists("input.txt"):
        raise FileNotFoundError("input.txt not found. Please download it first.")
        
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Split into 90% training, 10% validation (matching Section 3)
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # 3. Train or load tokenizer (vocab_size = 1024)
    print("Training BPETokenizer for baseline...")
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=1024)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    vocab_size = 1024

    # 4. Count bigram frequencies with add-one smoothing
    # bigram_counts[w1][w2] = count
    counts = [[1.0 * vocab_size for _ in range(vocab_size)] for _ in range(vocab_size)]
    uni_counts = [1.0 * vocab_size for _ in range(vocab_size)]

    for w1, w2 in zip(train_ids[:-1], train_ids[1:]):
        counts[w1][w2] += 1.0
        uni_counts[w1] += 1.0

    # Convert counts to log probabilities for validation loss calculation
    import math
    
    total_val_loss = 0.0
    num_tokens = len(val_ids) - 1

    for w1, w2 in zip(val_ids[:-1], val_ids[1:]):
        # Probability with add-one smoothing
        row_sum = uni_counts[w1]
        prob = counts[w1][w2] / row_sum
        total_val_loss -= math.log(prob)

    val_loss = total_val_loss / num_tokens
    print(f"Bigram Baseline Validation Loss: {val_loss:.4f} nats per token")

    # 5. Append baseline result to runs.csv as required by instructions
    os.makedirs("logs", exist_ok=True)
    runs_file = "runs.csv"
    
    file_exists = os.path.exists(runs_file)
    with open(runs_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id", "n_layer", "n_head", "n_embd", "block_size", 
                "max_lr", "max_steps", "ablation", "params", 
                "best_val_loss", "final_train_loss", "tokens_per_sec", "wall_time_min"
            ])
        writer.writerow([
            "baseline", 0, 0, 0, 0, 0, 0, "none", 0, f"{val_loss:.4f}", f"{val_loss:.4f}", 0, 0
        ])
    print("Baseline logged to runs.csv successfully!")

if __name__ == "__main__":
    main()