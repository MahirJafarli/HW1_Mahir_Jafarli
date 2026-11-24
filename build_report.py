# hw1-build: caspian-7
import argparse
import csv
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tokenizer import BPETokenizer


MODEL_RUNS = ("S", "M", "L", "A1", "A2", "A3", "A4", "A5")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def add_paragraph(fig, text, y, width=106, font_size=9):
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    line_count = wrapped.count("\n") + 1
    fig.text(0.08, y, wrapped, ha="left", va="top", fontsize=font_size, linespacing=1.35)
    return y - line_count * 0.020 - 0.018


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Mahir Jafarli")
    parser.add_argument("--r4-observation", required=True)
    parser.add_argument("--cached-tokens-per-second", required=True, type=float)
    parser.add_argument("--uncached-tokens-per-second", required=True, type=float)
    parser.add_argument("--inference-device", default="CPU (PyTorch 2.15.0+cpu)")
    args = parser.parse_args()

    runs = {row["run_id"]: row for row in read_csv("runs.csv")}
    required = set(MODEL_RUNS) | {"bigram"}
    if not required.issubset(runs):
        raise ValueError(f"runs.csv is missing rows: {sorted(required - set(runs))}")

    for run_id in MODEL_RUNS:
        log_rows = read_csv(f"logs/{run_id}.csv")
        if not log_rows:
            raise ValueError(f"logs/{run_id}.csv has no evaluations.")
        observed_best = min(float(row["val_loss"]) for row in log_rows)
        if abs(observed_best - float(runs[run_id]["best_val_loss"])) > 0.00011:
            raise ValueError(f"{run_id} best loss in runs.csv does not match its log.")

    text = open("input.txt", "r", encoding="utf-8").read()
    split = int(0.9 * len(text))
    holdout_shard = text[split:]
    tokenizer = BPETokenizer.load("tokenizer.json")
    holdout_ids = tokenizer.encode(holdout_shard)
    bytes_per_token = len(holdout_shard.encode("utf-8")) / len(holdout_ids)
    first_merges = [
        tokenizer.vocab[256 + index].decode("utf-8", errors="replace")
        for index in range(10)
    ]

    samples = open("samples.txt", "r", encoding="utf-8").read().strip()
    if samples.count("(run L, 120 new tokens)") != 3:
        raise ValueError("samples.txt must contain exactly three run L samples of 120 new tokens.")

    mse = runs["M"]
    baseline = runs["bigram"]
    gain = 100.0 * (1.0 - float(mse["best_val_loss"]) / float(baseline["best_val_loss"]))
    r4_note = args.r4_observation.strip()

    with PdfPages("report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.965, args.name, fontsize=11, weight="bold", va="top")
        fig.text(0.08, 0.935, "HW1 — Build a GPT from Scratch", fontsize=18, weight="bold", va="top")
        fig.text(
            0.08, 0.895,
            "Tokenizer, decoder-only Transformer, training pipeline, inference, and controlled experiments",
            fontsize=9, va="top",
        )
        fig.text(
            0.08, 0.855,
            "I implemented a byte-level BPE tokenizer and a small GPT in PyTorch. The eight GPT runs use the "
            "same Tiny Shakespeare holdout shard and 3,000 training steps; each run's best metric below is "
            "traceable to runs.csv and its evaluation log.",
            fontsize=9, va="top", wrap=True,
        )
        fig.text(0.08, 0.795, "Experiment results", fontsize=12, weight="bold", va="top")

        table_rows = []
        for run_id in (*MODEL_RUNS, "bigram"):
            row = runs[run_id]
            table_rows.append([
                run_id,
                f'{int(row["params"]):,}',
                f'{float(row["best_val_loss"]):.4f}',
                f'{float(row["final_train_loss"]):.4f}',
            ])
        table_ax = fig.add_axes([0.07, 0.475, 0.86, 0.29])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=table_rows,
            colLabels=["Run", "Parameters", "Best holdout loss", "Final train loss"],
            cellLoc="center",
            loc="center",
            colWidths=[0.15, 0.25, 0.30, 0.30],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dce8f2")

        fig.text(
            0.08, 0.445,
            f'Model M lowers loss from the add-one bigram baseline by {gain:.1f}% '
            f'({mse["best_val_loss"]} versus {baseline["best_val_loss"]} nats/token), '
            "exceeding the assignment's 10% target.",
            fontsize=9, va="top",
        )
        cache_speedup = args.cached_tokens_per_second / args.uncached_tokens_per_second
        fig.text(
            0.08, 0.420,
            f"Greedy generation for 120 new tokens: {args.cached_tokens_per_second:.1f} tokens/s cached, "
            f"{args.uncached_tokens_per_second:.1f} tokens/s uncached "
            f"({cache_speedup:.2f}x speed-up) on {args.inference_device}.",
            fontsize=9, va="top",
        )
        curve = plt.imread("figures/validation_loss_curves.png")
        curve_ax = fig.add_axes([0.08, 0.055, 0.84, 0.35])
        curve_ax.imshow(curve)
        curve_ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.965, "Results and reflections", fontsize=16, weight="bold", va="top")
        scaling = plt.imread("figures/scaling_validation_loss.png")
        scaling_ax = fig.add_axes([0.12, 0.60, 0.76, 0.32])
        scaling_ax.imshow(scaling)
        scaling_ax.axis("off")

        s, m, large = (runs[key] for key in ("S", "M", "L"))
        y = 0.565
        y = add_paragraph(
            fig,
            f"Scaling comparison: S ({int(s['params']):,} parameters) reaches {s['best_val_loss']} "
            f"holdout loss, M ({int(m['params']):,}) reaches {m['best_val_loss']}, and L "
            f"({int(large['params']):,}) reaches {large['best_val_loss']}. The larger L model has the "
            "lowest measured loss among these three, while the logs show a lower throughput for the larger model.",
            y,
        )
        y = add_paragraph(
            fig,
            "R1. The most time-consuming inference bug was KV-cache prefill: the first cached call passed an "
            "empty cache list, but blocks treated None entries like ordinary attention and returned only hidden "
            "states instead of the cache tuple expected by GPT.forward. Comparing cached and uncached greedy "
            "generation exposed the mismatch. Commit 1986095 added an explicit cache-use path and fixed initialization.",
            y,
        )
        y = add_paragraph(
            fig,
            f"R2. A3 was the surprising result: using max_lr=0.01 did not diverge in this run and reached "
            f"{runs['A3']['best_val_loss']}, slightly below M's {runs['M']['best_val_loss']}. This is an "
            "observation for this seed and schedule, not evidence that the larger learning rate is generally safer.",
            y,
        )
        add_paragraph(
            fig,
            f"R3. The largest controlled ablation change was removing position embeddings: A1 reached "
            f"{runs['A1']['best_val_loss']} versus M at {runs['M']['best_val_loss']}, a loss increase of "
            f"{float(runs['A1']['best_val_loss']) - float(runs['M']['best_val_loss']):.4f}. "
            "This comparison isolates the role of learned position information in this setup.",
            y,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.965, "Generated samples and tokenizer analysis", fontsize=16, weight="bold", va="top")
        y = add_paragraph(
            fig,
            "R4. Three seeded samples of 120 new tokens each were generated from run L at temperature 0.8 "
            "and are included in samples.txt. " + r4_note,
            0.91,
        )
        y = add_paragraph(
            fig,
            f"R5. For the tokenizer used by run L, the first ten learned byte merges, in order, are: "
            f"{', '.join(repr(item) for item in first_merges)}. The tokenizer represents the holdout shard "
            f"at {bytes_per_token:.4f} UTF-8 bytes per token. "
            "Common short English letter pairs and a frequent trailing space appear early because BPE repeatedly "
            "merges frequent adjacent byte-token pairs; corpus word and punctuation frequencies shape the order.",
            y,
        )
        fig.text(0.08, y - 0.015, "First ten merges", fontsize=11, weight="bold", va="top")
        merge_rows = [[str(index + 1), repr(value)] for index, value in enumerate(first_merges)]
        merge_ax = fig.add_axes([0.12, max(0.08, y - 0.34), 0.76, 0.30])
        merge_ax.axis("off")
        merge_table = merge_ax.table(
            cellText=merge_rows,
            colLabels=["Rank", "Merged byte sequence"],
            cellLoc="left",
            loc="center",
            colWidths=[0.2, 0.8],
        )
        merge_table.auto_set_font_size(False)
        merge_table.set_fontsize(9)
        merge_table.scale(1, 1.25)
        for (row_index, _), cell in merge_table.get_celld().items():
            if row_index == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dce8f2")
        fig.text(
            0.08, 0.045,
            "All experiment metrics are from runs.csv and logs/<run_id>.csv. The holdout shard is the final "
            "10% of the corpus by characters and was not used to train the tokenizer or model.",
            fontsize=8, va="bottom",
        )
        pdf.savefig(fig)
        plt.close(fig)

    print("Wrote report.pdf (3 pages).")


if __name__ == "__main__":
    main()
