import argparse

from src.audio_io import patch_torchaudio
from src.train import train_model

from experiments.run_experiments import run_n_mels_experiments, run_groups_experiments
from experiments.plot_results import main as plot_results


def run_baseline():
    print("\nRunning baseline training...\n")

    model, result = train_model(
        data_root="data/yesno",
        batch_size=32,
        epochs=10,
        lr=1e-3,
        n_mels=80,
        groups=1,
        save_results=True
    )

    print("\nBaseline finished.")
    print(f"Test accuracy: {result['test_acc']:.4f}")


def run_experiments():
    print("\nRunning n_mels experiments...\n")
    run_n_mels_experiments()

    print("\nRunning groups experiments...\n")
    run_groups_experiments()

    print("\nExperiments finished.")


def run_plots():
    print("\nGenerating plots...\n")
    plot_results()
    print("Plots saved to figures/")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "experiments", "plots"],
        help="What to run"
    )

    args = parser.parse_args()

    patch_torchaudio()

    if args.mode == "baseline":
        run_baseline()

    elif args.mode == "experiments":
        run_experiments()

    elif args.mode == "plots":
        run_plots()


if __name__ == "__main__":
    main()