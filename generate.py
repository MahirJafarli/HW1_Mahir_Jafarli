# hw1-build: caspian-7
import math
import torch


def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if len(prompt_ids) + max_new_tokens > model.config.block_size:
        raise ValueError("Prompt plus generated tokens cannot exceed block_size.")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")
    if max_new_tokens == 0:
        return list(prompt_ids)

    device = next(model.parameters()).device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    sequence = list(prompt_ids)
    input_ids = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    kv_caches = None
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                if use_cache:
                    if step == 0:
                        logits, _, kv_caches = model(
                            input_ids,
                            kv_caches=[None] * len(model.transformer.h),
                        )
                    else:
                        logits, _, kv_caches = model(
                            input_ids[:, -1:],
                            kv_caches=kv_caches,
                        )
                else:
                    context = torch.tensor(
                        sequence, dtype=torch.long, device=device
                    ).unsqueeze(0)
                    logits, _ = model(context)

                next_logits = logits[0, -1, :]
                if temperature == 0:
                    next_token = int(torch.argmax(next_logits).item())
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None:
                        k = min(top_k, next_logits.numel())
                        top_values, top_indices = torch.topk(next_logits, k)
                        filtered_logits = torch.full_like(next_logits, float("-inf"))
                        filtered_logits.scatter_(0, top_indices, top_values)
                        next_logits = filtered_logits

                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(
                            next_logits, descending=True
                        )
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
                        remove[0] = False
                        next_logits[sorted_indices[remove]] = float("-inf")

                    probabilities = torch.softmax(next_logits, dim=-1)
                    next_token = int(
                        torch.multinomial(
                            probabilities, 1, generator=generator
                        ).item()
                    )

                sequence.append(next_token)
                input_ids = torch.tensor(
                    [[next_token]], dtype=torch.long, device=device
                )
    finally:
        model.train(was_training)

    return sequence
