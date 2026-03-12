from src.audio_io import patch_torchaudio
from src.train import train_model

patch_torchaudio()


def run_n_mels_experiments():
    for n_mels in [20, 40, 80]:
        train_model(
            data_root="data/yesno",
            batch_size=32,
            epochs=10,
            lr=1e-3,
            n_mels=n_mels,
            groups=1,
            save_results=True,
            save_dir="results"
        )


def run_groups_experiments():
    for groups in [2, 4, 8, 16]:
        train_model(
            data_root="data/yesno",
            batch_size=32,
            epochs=10,
            lr=1e-3,
            n_mels=80,
            groups=groups,
            save_results=True,
            save_dir="results"
        )


if __name__ == "__main__":
    run_n_mels_experiments()
    run_groups_experiments()