import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_results():
    results = []
    for path in RESULTS_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def plot_n_mels_train_loss(results):
    nmels_results = sorted(
        [r for r in results if r["groups"] == 1],
        key=lambda x: x["n_mels"]
    )

    plt.figure(figsize=(8, 5))
    for r in nmels_results:
        epochs = list(range(1, len(r["history"]["train_loss"]) + 1))
        plt.plot(epochs, r["history"]["train_loss"], marker="o", label=f'n_mels={r["n_mels"]}')

    plt.title("Train Loss по эпохам для разных n_mels")
    plt.xlabel("Эпоха")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nmels_train_loss.png", dpi=200)
    plt.close()


def plot_n_mels_test_acc(results):
    nmels_results = sorted(
        [r for r in results if r["groups"] == 1],
        key=lambda x: x["n_mels"]
    )

    x = [r["n_mels"] for r in nmels_results]
    y = [r["test_acc"] for r in nmels_results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title("Зависимость test accuracy от n_mels")
    plt.xlabel("n_mels")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nmels_test_accuracy.png", dpi=200)
    plt.close()


def plot_groups_epoch_time(results):
    groups_results = sorted(
        [r for r in results if r["groups"] != 1],
        key=lambda x: x["groups"]
    )

    x = [r["groups"] for r in groups_results]
    y = [r["avg_epoch_time"] for r in groups_results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title("Зависимость среднего времени эпохи от groups")
    plt.xlabel("groups")
    plt.ylabel("Среднее время эпохи, сек")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "groups_epoch_time.png", dpi=200)
    plt.close()


def plot_groups_parameters(results):
    groups_results = sorted(
        [r for r in results if r["groups"] != 1],
        key=lambda x: x["groups"]
    )

    x = [r["groups"] for r in groups_results]
    y = [r["parameters"] for r in groups_results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title("Зависимость числа параметров от groups")
    plt.xlabel("groups")
    plt.ylabel("Число параметров")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "groups_parameters.png", dpi=200)
    plt.close()


def plot_groups_flops(results):
    groups_results = sorted(
        [r for r in results if r["groups"] != 1],
        key=lambda x: x["groups"]
    )

    x = [r["groups"] for r in groups_results]
    y = [r["flops"] for r in groups_results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title("Зависимость FLOPs от groups")
    plt.xlabel("groups")
    plt.ylabel("FLOPs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "groups_flops.png", dpi=200)
    plt.close()


def plot_groups_test_acc(results):
    groups_results = sorted(
        [r for r in results if r["groups"] != 1],
        key=lambda x: x["groups"]
    )

    x = [r["groups"] for r in groups_results]
    y = [r["test_acc"] for r in groups_results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title("Зависимость test accuracy от groups")
    plt.xlabel("groups")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "groups_test_accuracy.png", dpi=200)
    plt.close()


def plot_summary_table(results):
    rows = []

    for r in results:
        rows.append({
            "n_mels": r["n_mels"],
            "groups": r["groups"],
            "parameters": r["parameters"],
            "FLOPs": r.get("flops", None),
            "avg_epoch_time": round(r["avg_epoch_time"], 3),
            "test_accuracy": round(r["test_acc"], 4)
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(["n_mels", "groups"])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Сводная таблица результатов экспериментов")
    plt.savefig(FIGURES_DIR / "summary_table.png", bbox_inches="tight", dpi=200)
    plt.close()

def main():
    results = load_results()

    plot_n_mels_train_loss(results)
    plot_n_mels_test_acc(results)
    plot_groups_epoch_time(results)
    plot_groups_parameters(results)
    plot_groups_flops(results)
    plot_groups_test_acc(results)
    plot_summary_table(results)

    print("Графики сохранены в папку figures/")


if __name__ == "__main__":
    main()