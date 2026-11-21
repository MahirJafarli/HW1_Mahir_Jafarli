# hw1-build: caspian-7
import argparse
import csv
import hashlib
import json
import math
import os
import time

import torch
from torch.optim import AdamW

from tokenizer import BPETokenizer
from model import GPT, GPTConfig


RUNS_COLUMNS = [
    "run_id", "n_layer", "n_head", "n_embd", "block_size",
    "max_lr", "max_steps", "ablation", "params",
    "best_val_loss", "final_train_loss", "tokens_per_sec", "wall_time_min",
]


def get_batch(data, block_size, batch_size, device):
    data = torch.as_tensor(data, dtype=torch.long)
    if data.numel() <= block_size:
        raise ValueError("Token data must contain more tokens than block_size.")

    starts = torch.randint(data.numel() - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_batches=50):
    if eval_batches < 1:
        raise ValueError("eval_batches must be at least 1.")

    was_training = model.training
    model.eval()
    try:
        losses = {}
        for split, data in (("train", train_data), ("val", val_data)):
            batch_losses = torch.zeros(eval_batches)
            for index in range(eval_batches):
                inputs, targets = get_batch(data, block_size, batch_size, device)
                _, loss = model(inputs, targets)
                batch_losses[index] = loss.item()
            losses[split] = batch_losses.mean().item()
        return losses["train"], losses["val"]
    finally:
        model.train(was_training)


def get_lr(step, max_lr, warmup_steps, max_steps):
    if warmup_steps <= 0 or max_steps <= warmup_steps:
        raise ValueError("max_steps must be greater than positive warmup_steps.")
    if step <= 0:
        return 0.0
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return max_lr * 0.1

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * (0.1 + 0.9 * coeff)


def load_token_ids(train_text, val_text, tokenizer, cache_dir="cache"):
    tokenizer_json = json.dumps(tokenizer.merges, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(
        train_text.encode("utf-8") + b"\0" +
        val_text.encode("utf-8") + b"\0" + tokenizer_json
    ).hexdigest()
    cache_path = os.path.join(cache_dir, f"tokenized_{fingerprint}.pt")

    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cached["train_ids"], cached["val_ids"]

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"train_ids": train_ids, "val_ids": val_ids}, cache_path)
    return train_ids, val_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON run config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config_data = json.load(config_file)

    run_id = config_data["run_id"]
    n_layer = config_data["n_layer"]
    n_head = config_data["n_head"]
    n_embd = config_data["n_embd"]
    block_size = config_data["block_size"]
    dropout = config_data["dropout"]
    batch_size = config_data["batch_size"]
    max_steps = config_data["max_steps"]
    max_lr = config_data["max_lr"]
    eval_interval = config_data["eval_interval"]
    seed = config_data["seed"]
    ablation = config_data.get("ablation", "none")
    warmup_steps = 137

    if max_steps <= warmup_steps:
        raise ValueError("max_steps must be greater than the required 137-step warmup.")
    if eval_interval <= 0 or batch_size <= 0 or max_lr <= 0:
        raise ValueError("eval_interval, batch_size, and max_lr must be positive.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    with open("input.txt", "r", encoding="utf-8") as dataset_file:
        text = dataset_file.read()
    if not text:
        raise ValueError("input.txt is empty.")

    split_idx = int(0.9 * len(text))
    train_text, val_text = text[:split_idx], text[split_idx:]
    if not train_text or not val_text:
        raise ValueError("input.txt must have non-empty training and validation splits.")

    tokenizer_path = "tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer on the training split...")
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, vocab_size=1024)
        tokenizer.save(tokenizer_path)

    train_ids, val_ids = load_token_ids(train_text, val_text, tokenizer)
    vocab_size = len(tokenizer.vocab)
    if len(train_ids) <= block_size or len(val_ids) <= block_size:
        raise ValueError("Both tokenized splits must contain more tokens than block_size.")
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(config).to(device)
    params = model.num_params()
    print(f"Model initialized with {params:,} parameters.")

    decay_params = []
    no_decay_params = []
    for _, parameter in model.named_parameters():
        if parameter.dim() >= 2:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=max_lr,
        betas=(0.9, 0.95),
    )

    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", f"{run_id}.csv")
    with open(log_file_path, "w", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(
            ["step", "train_loss", "val_loss", "lr", "tokens_seen"]
        )

    best_val_loss = float("inf")
    tokens_seen = 0
    start_time = time.perf_counter()
    model.train()

    for step in range(1, max_steps + 1):
        learning_rate = get_lr(step, max_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        inputs, targets = get_batch(train_ids, block_size, batch_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += batch_size * block_size

        if step % eval_interval == 0 or step == max_steps:
            train_loss, val_loss = estimate_loss(
                model, train_ids, val_ids, block_size, batch_size, device
            )
            print(
                f"Step {step}/{max_steps} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | LR: {learning_rate:.5e}"
            )
            with open(log_file_path, "a", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow(
                    [step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                     f"{learning_rate:.5e}", tokens_seen]
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{run_id}_best.pt")

    final_train_loss, _ = estimate_loss(
        model, train_ids, val_ids, block_size, batch_size, device
    )
    wall_time_seconds = time.perf_counter() - start_time
    wall_time_min = wall_time_seconds / 60.0
    tokens_per_sec = tokens_seen / wall_time_seconds

    runs_file = "runs.csv"
    write_header = not os.path.exists(runs_file) or os.path.getsize(runs_file) == 0
    with open(runs_file, "a", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        if write_header:
            writer.writerow(RUNS_COLUMNS)
        writer.writerow(
            [
                run_id, n_layer, n_head, n_embd, block_size, max_lr, max_steps,
                ablation, params, f"{best_val_loss:.4f}", f"{final_train_loss:.4f}",
                f"{tokens_per_sec:.1f}", f"{wall_time_min:.2f}",
            ]
        )
    print(f"Run {run_id} finished successfully in {wall_time_min:.2f} minutes.")


if __name__ == "__main__":
    main()
