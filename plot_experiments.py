# hw1-build: caspian-7
import csv
import os

import matplotlib.pyplot as plt


RUN_IDS = ("S", "M", "L", "A1", "A2", "A3", "A4", "A5")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def main():
    runs = {row["run_id"]: row for row in read_csv("runs.csv")}
    missing_runs = [run_id for run_id in RUN_IDS if run_id not in runs]
    missing_logs = [
        run_id for run_id in RUN_IDS
        if not os.path.exists(os.path.join("logs", f"{run_id}.csv"))
    ]
    if missing_runs or missing_logs:
        raise FileNotFoundError(
            f"Missing run rows: {missing_runs}; missing training logs: {missing_logs}"
        )

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for run_id in RUN_IDS:
        rows = read_csv(os.path.join("logs", f"{run_id}.csv"))
        if not rows:
            raise ValueError(f"Training log for {run_id} has no evaluation rows.")
        steps = [int(row["step"]) for row in rows]
        losses = [float(row["val_loss"]) for row in rows]
        plt.plot(steps, losses, label=run_id)
    plt.xlabel("Training step")
    plt.ylabel("Holdout-shard loss (nats/token)")
    plt.title("Holdout-shard loss across GPT experiments")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "validation_loss_curves.png"), dpi=200)
    plt.close()

    scaling_ids = ("S", "M", "L")
    params = [int(runs[run_id]["params"]) for run_id in scaling_ids]
    best_losses = [float(runs[run_id]["best_val_loss"]) for run_id in scaling_ids]
    plt.figure(figsize=(7, 5))
    plt.plot(params, best_losses, marker="o")
    for run_id, parameter_count, loss in zip(scaling_ids, params, best_losses):
        plt.annotate(run_id, (parameter_count, loss), xytext=(5, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Unique trainable parameters (log scale)")
    plt.ylabel("Best holdout-shard loss (nats/token)")
    plt.title("Scaling: model size versus best validation loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "scaling_validation_loss.png"), dpi=200)
    plt.close()
    print("Saved figures/validation_loss_curves.png and figures/scaling_validation_loss.png")


if __name__ == "__main__":
    main()
