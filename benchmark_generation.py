# hw1-build: caspian-7
import json
import time

import torch

from generate import generate
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def main():
    tokenizer = BPETokenizer.load("tokenizer.json")
    with open("configs/M.json", "r", encoding="utf-8") as config_file:
        run_config = json.load(config_file)
    config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=run_config["block_size"],
        n_layer=run_config["n_layer"],
        n_head=run_config["n_head"],
        n_embd=run_config["n_embd"],
        dropout=run_config["dropout"],
    )
    model = GPT(config)
    state_dict = torch.load(
        "checkpoints/M_best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    prompt_ids = tokenizer.encode("ROMEO:\n")
    new_tokens = 120
    if len(prompt_ids) + new_tokens > config.block_size:
        raise ValueError("The benchmark prompt and generated tokens exceed block_size.")

    print(f"device={device}, prompt_tokens={len(prompt_ids)}, new_tokens={new_tokens}")
    outputs = {}
    for use_cache in (False, True):
        generate(
            model, prompt_ids, new_tokens, temperature=0, use_cache=use_cache, seed=42
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        durations = []
        for seed in (42, 43, 44):
            start = time.perf_counter()
            generated = generate(
                model,
                prompt_ids,
                new_tokens,
                temperature=0,
                use_cache=use_cache,
                seed=seed,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
        outputs[use_cache] = generated
        average_seconds = sum(durations) / len(durations)
        print(
            f"use_cache={use_cache}, average_seconds={average_seconds:.6f}, "
            f"tokens_per_second={new_tokens / average_seconds:.2f}"
        )

    if outputs[False] != outputs[True]:
        raise RuntimeError("Greedy generation differs between cached and uncached modes.")


if __name__ == "__main__":
    main()
