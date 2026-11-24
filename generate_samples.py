# hw1-build: caspian-7
import json

import torch

from generate import generate
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def main():
    with open("configs/L.json", "r", encoding="utf-8") as config_file:
        run_config = json.load(config_file)
    tokenizer = BPETokenizer.load("tokenizer.json")

    model_config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=run_config["block_size"],
        n_layer=run_config["n_layer"],
        n_head=run_config["n_head"],
        n_embd=run_config["n_embd"],
        dropout=run_config["dropout"],
    )
    model = GPT(model_config)
    state_dict = torch.load(
        "checkpoints/L_best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    prompt = "ROMEO:\n"
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) + 120 > model_config.block_size:
        raise ValueError("The chosen prompt leaves insufficient room for 120 new tokens.")

    samples = []
    for index, seed in enumerate((42, 43, 44), start=1):
        sample_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=120,
            temperature=0.8,
            use_cache=True,
            seed=seed,
        )
        if len(sample_ids) - len(prompt_ids) != 120:
            raise RuntimeError("Generation did not return exactly 120 new tokens.")
        samples.append(
            f"Sample {index} (run L, 120 new tokens):\n"
            f"{tokenizer.decode(sample_ids).rstrip()}"
        )

    with open("samples.txt", "w", encoding="utf-8") as output_file:
        output_file.write("\n\n".join(samples) + "\n")
    print("Wrote three seeded, 120-token samples to samples.txt.")


if __name__ == "__main__":
    main()
