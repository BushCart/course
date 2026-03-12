import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.dataset import get_dataloaders
from src.model import SpeechCommandCNN, count_parameters
from src.flops import compute_model_flops


def accuracy_fn(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_fn(logits, y) * batch_size
            total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


def save_experiment_result(result, save_dir="results"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"nmels_{result['n_mels']}_groups_{result['groups']}.json"
    save_path = save_dir / filename

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved results to: {save_path}")


def train_model(
    data_root="data/yesno",
    batch_size=32,
    epochs=10,
    lr=1e-3,
    n_mels=80,
    groups=1,
    device=None,
    save_results=True,
    save_dir="results"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size
    )

    model = SpeechCommandCNN(n_mels=n_mels, groups=groups).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    print(f"\n=== Experiment: n_mels={n_mels}, groups={groups} ===")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model)}")
    print("Start training...\n")

    for epoch in range(1, epochs + 1):
        model.train()

        start_time = time.time()
        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size_cur = x.size(0)
            running_loss += loss.item() * batch_size_cur
            total_samples += batch_size_cur

        epoch_time = time.time() - start_time
        train_loss = running_loss / total_samples

        _, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"time={epoch_time:.2f}s"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    flops_info = compute_model_flops(n_mels=n_mels, groups=groups)
    result = {
        "n_mels": n_mels,
        "groups": groups,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "parameters": count_parameters(model),
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "best_val_acc": max(history["val_acc"]),
        "final_train_loss": history["train_loss"][-1],
        "macs": flops_info["macs"],
        "flops": flops_info["flops"],
        "params_thop": flops_info["params_thop"],
    }

    print("\nTraining finished.")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    if save_results:
        save_experiment_result(result, save_dir=save_dir)

    return model, result