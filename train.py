import os
import sys
import json
import csv
import math
import time
import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import AdamW

from tokenizer import BPETokenizer
from model import GPT, GPTConfig

def get_batch(data, block_size, batch_size, device):
    # Random starting indices for batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_batches=50):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out['train'], out['val']

def get_lr(it, max_lr, warmup_steps, max_steps):
    # 1. Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2. If it exceeds max_steps, return min lr (0.1 * max_lr)
    if it > max_steps:
        return max_lr * 0.1
    # 3. Cosine decay down to 0.1 * max_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * (0.1 + 0.9 * coeff)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_cmd", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    run_id = config_data["run_id"]
    n_layer = config_data["n_layer"]
    n_head = config_data["n_head"]
    n_embd = config_data["n_embd"]
    block_size = config_data["block_size"]
    dropout = config_data["dropout"]
    batch_size = config_data["batch_size"]
    max_steps = config_data["max_steps"]
    max_lr = config_data["max_lr"]
    warmup_steps = config_data["warmup_steps"]
    eval_interval = config_data["eval_interval"]
    seed = config_data["seed"]
    ablation = config_data.get("ablation", "none")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 1. Load dataset & Tokenize
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    tokenizer_path = "tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, vocab_size=1024)
        tokenizer.save(tokenizer_path)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    # 2. Initialize Model
    cfg = GPTConfig(
        vocab_size=1024,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    model = GPT(cfg)
    model.to(device)
    params = model.num_params()
    print(f"Model initialized with {params:,} parameters.")

    # 3. Optimizer setup (Weight decay only on params with >= 2 dimensions)
    decay_params = []
    nodecay_params = []
    for pn, p in model.named_parameters():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95))

    os.path.exists("logs") or os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", f"{run_id}.csv")

    with open(log_file_path, mode="w", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_seen"])

    best_val_loss = float("inf")
    tokens_seen = 0
    start_time = time.time()

    # 4. Training loop
    model.train()
    X, Y = get_batch(train_ids, block_size, batch_size, device)

    for step in range(max_steps + 1):
        t0 = time.time()
        lr = get_lr(step, max_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip global gradient norm to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Fetch next batch
        X, Y = get_batch(train_ids, block_size, batch_size, device)
        tokens_seen += batch_size * block_size
        t1 = time.time()

        # Evaluation interval
        if step % eval_interval == 0 or step == max_steps:
            train_loss_est, val_loss_est = estimate_loss(model, train_ids, val_ids, block_size, batch_size, device)
            print(f"Step {step}/{max_steps} | Train Loss: {train_loss_est:.4f} | Val Loss: {val_loss_est:.4f} | LR: {lr:.5e}")

            with open(log_file_path, mode="a", newline="", encoding="utf-8") as lf:
                writer = csv.writer(lf)
                writer.writerow([step, f"{train_loss_est:.4f}", f"{val_loss_est:.4f}", f"{lr:.5e}", tokens_seen])

            if val_loss_est < best_val_loss:
                best_val_loss = val_loss_est
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{run_id}_best.pt")

    wall_time_min = (time.time() - start_time) / 60.0
    tokens_per_sec = tokens_seen / (time.time() - start_time)
    final_train_loss, final_val_loss = estimate_loss(model, train_ids, val_ids, block_size, batch_size, device)

    # Append run summary to runs.csv
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
            run_id, n_layer, n_head, n_embd, block_size, max_lr, max_steps,
            ablation, params, f"{best_val_loss:.4f}", f"{final_train_loss:.4f}",
            f"{tokens_per_sec:.1f}", f"{wall_time_min:.2f}"
        ])
    print(f"Run {run_id} finished successfully in {wall_time_min:.2f} minutes!")

if __name__ == "__main__":
    main()