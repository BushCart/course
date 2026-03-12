from pathlib import Path
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
#from src.audio_io import patch_torchaudio ###
#patch_torchaudio() ###

LABEL_MAP = {
    "no": 0,
    "yes": 1
}


class YesNoDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.files = []
        self.labels = []

        for label_name in ["yes", "no"]:

            label_dir = self.root / label_name

            for wav in label_dir.glob("*.wav"):

                self.files.append(wav)
                self.labels.append(LABEL_MAP[label_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        waveform, sr = torchaudio.load(path)

        waveform = waveform.squeeze(0)

        length = waveform.shape[0]

        if length < 16000:
            pad = 16000 - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        elif length > 16000:
            waveform = waveform[:16000]

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return waveform, label


def get_dataloaders(data_root, batch_size=32):

    train_ds = YesNoDataset(Path(data_root) / "train")
    val_ds = YesNoDataset(Path(data_root) / "validation")
    test_ds = YesNoDataset(Path(data_root) / "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def check_dataset(data_root):

    data_root = Path(data_root)

    print("\nDATASET CHECK\n")

    for split in ["train", "validation", "test"]:

        yes_count = len(list((data_root / split / "yes").glob("*.wav")))
        no_count = len(list((data_root / split / "no").glob("*.wav")))

        total = yes_count + no_count

        print(f"{split.upper()}")
        print(f"yes: {yes_count}")
        print(f"no : {no_count}")
        print(f"total: {total}")
        print("-" * 20)

    print("Dataset structure OK\n")