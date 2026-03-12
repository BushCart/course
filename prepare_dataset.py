import shutil
from pathlib import Path
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS


TARGET_LABELS = {"yes", "no"}

RAW_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/yesno")


def copy_sample(dataset, split_name):
    for rel_path in dataset._walker:
        rel_path = Path(rel_path)

        if rel_path.exists():
            src_path = rel_path
        else:
            src_path = Path(dataset._path) / rel_path

        label = src_path.parent.name

        if label not in TARGET_LABELS:
            continue

        dst_dir = OUTPUT_PATH / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / src_path.name

        if not dst_path.exists():
            shutil.copy(src_path, dst_path)

def main():

    RAW_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("Downloading SpeechCommands dataset if needed...")

    train_ds = SPEECHCOMMANDS(root=RAW_PATH, subset="training", download=True)
    val_ds = SPEECHCOMMANDS(root=RAW_PATH, subset="validation")
    test_ds = SPEECHCOMMANDS(root=RAW_PATH, subset="testing")

    print("Preparing YES/NO subset...")

    copy_sample(train_ds, "train")
    copy_sample(val_ds, "validation")
    copy_sample(test_ds, "test")

    print("Done.")
    print("Dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()