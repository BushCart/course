import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path

import soundfile as sf

from src.audio_io import patch_torchaudio ###
patch_torchaudio() ###

from src.melbanks import LogMelFilterBanks
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def main():

    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root="data/raw",
        subset="training",
        download=True
    )

    signal, sr, label, speaker, utt = dataset[0]

    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80
    )(signal)

    native_logmel = torch.log(melspec + 1e-6)

    custom_logmel = LogMelFilterBanks()(signal)

    print("shape equal:", native_logmel.shape == custom_logmel.shape)
    print("allclose:", torch.allclose(native_logmel, custom_logmel))

    fig, ax = plt.subplots(1, 2, figsize=(12,4))

    im1 = ax[0].imshow(native_logmel[0].numpy(), origin="lower", aspect="auto")
    ax[0].set_title("torchaudio Log-Mel")
    ax[0].set_xlabel("Time frames")
    ax[0].set_ylabel("Mel filter index")
    ax[0].set_xlim(50, 100)
    plt.colorbar(im1, ax=ax[0], label="Log amplitude")

    im2 = ax[1].imshow(custom_logmel[0].detach().numpy(), origin="lower", aspect="auto")
    ax[1].set_title("Custom LogMelFilterBanks")
    ax[1].set_xlabel("Time frames")
    ax[1].set_ylabel("Mel filter index")
    ax[1].set_xlim(50, 100)
    plt.colorbar(im2, ax=ax[1], label="Log amplitude")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "logmel_comparison.png", dpi=150, bbox_inches="tight")

    x = native_logmel.flatten().numpy()
    y = custom_logmel.flatten().detach().numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=2)

    min_v = min(x.min(), y.min())
    max_v = max(x.max(), y.max())

    plt.plot([min_v, max_v], [min_v, max_v], "r--")

    plt.xlabel("torchaudio + log (amplitude)")
    plt.ylabel("custom LogMelFilterBanks (amplitude)")
    plt.title("Element-wise comparison")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "logmel_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Figures saved to ./figures")


if __name__ == "__main__":
    main()