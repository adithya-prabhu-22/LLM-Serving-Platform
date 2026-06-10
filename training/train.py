import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel
from training.dataset import TextChunkDataset
from training.loss import GPTLoss
from training.evaluate import evaluate
from training.utils.save_safetensor import save_safetensor_checkpoint
from training.utils.checkpoint import save_config


DATA_DIR = "/content/drive/MyDrive/training_data"
OUTPUT_DIR = "/content/drive/MyDrive/GPT2_35M"
CONFIG_PATH = "training/configs/gpt_35m.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config_dict = json.load(file)
    return GPTConfig(**config_dict)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    config = load_config()

    output_dir = Path(OUTPUT_DIR)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, output_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TextChunkDataset(
        data_dir=DATA_DIR,
        block_size=config.block_size,
        stride=256,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = GPTModel(config).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = GPTLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    num_epochs = 3
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for step, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        val_loss, perplexity = evaluate(model, val_loader, criterion, device)

        print(
            f"\nEpoch {epoch+1}\n"
            f"Train Loss: {train_loss:.4f}\n"
            f"Validation Loss: {val_loss:.4f}\n"
            f"Perplexity: {perplexity:.4f}\n"
        )

        save_safetensor_checkpoint(model, checkpoint_dir / "latest.safetensors")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_safetensor_checkpoint(model, checkpoint_dir / "best.safetensors")
            print("New best model saved.")

    print("\nTraining Complete")


if __name__ == "__main__":
    main()