import argparse
import json
import math
import shutil
from functools import partial
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel
from training.datasets.streaming_dataset import StreamingDataset
from training.datasets.manifest_loader import load_manifest
from training.datasets.collate import collate_batch
from training.trainer.training_loop import train_one_epoch
from training.utils.tokenizer import TOKENIZER
from training.utils.seed import set_seed
from training.utils.checkpoint import load_checkpoint, latest_checkpoint, save_checkpoint
from training.utils.safetensor import save_safetensor


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_model(model_config):
    config = GPTConfig(
        vocab_size=TOKENIZER.n_vocab,
        block_size=model_config["block_size"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config.get("dropout", 0.1),
        ff_dim=model_config.get("ff_dim"),
        activation=model_config.get("activation", "gelu"),
        qkv_bias=model_config.get("qkv_bias", False),
        use_flash_attention=model_config.get("use_flash_attention", False),
        cache_type=model_config.get("cache_type", "ring"),
    )
    return GPTModel(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(config["model"])
    model.to(device)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config["dataset"]["manifest"])
    total_chunks = len(manifest["chunks"])
    print(f"Total Chunks: {total_chunks:,}")

    steps_per_epoch = 0
    for chunk_path in manifest["chunks"]:
        dataset = StreamingDataset(
            chunk_path=chunk_path,
            block_size=config["model"]["block_size"],
        )
        steps_per_epoch += math.ceil(
            len(dataset) / config["training"]["batch_size"]
        )

    total_steps = steps_per_epoch * config["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    latest_ckpt = latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    start_chunk = 0
    global_step = 0
    best_val_loss = float("inf")

    if latest_ckpt:
        print(f"Resuming from {latest_ckpt}")
        state = load_checkpoint(
            checkpoint_path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = state["epoch"]
        global_step = state["step"]
        best_val_loss = state["best_val_loss"]
        start_chunk = state["chunk_index"]

    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\nStarting Epoch {epoch + 1}")

        chunk_start = start_chunk if epoch == start_epoch else 0

        for chunk_idx in range(chunk_start, total_chunks):
            chunk_path = manifest["chunks"][chunk_idx]
            print(f"\nChunk {chunk_idx + 1}/{total_chunks}")

            dataset = StreamingDataset(
                chunk_path=chunk_path,
                block_size=config["model"]["block_size"],
            )

            dataloader = DataLoader(
                dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=True,
                collate_fn=partial(collate_batch, pad_token_id=0),
                num_workers=config["training"].get("num_workers", 4),
                pin_memory=(device == "cuda"),
            )

            global_step, best_val_loss, avg_loss = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch + 1,
                global_step=global_step,
                eval_dataloader=None,
                eval_interval=config["training"]["eval_interval"],
                save_interval=10**9,
                checkpoint_dir=checkpoint_dir,
                output_dir=str(output_dir),
                max_grad_norm=config["training"]["max_grad_norm"],
                best_val_loss=best_val_loss,
                gradient_accumulation_steps=config["training"].get(
                    "gradient_accumulation_steps", 1
                ),
                scaler=scaler,
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                loss=avg_loss,
                best_val_loss=best_val_loss,
                checkpoint_dir=checkpoint_dir,
                chunk_index=chunk_idx + 1,
            )
            print(f"Checkpoint saved after chunk {chunk_idx + 1}")

        start_chunk = 0

    final_model_path = f"{output_dir}/model.safetensors"
    save_safetensor(model, final_model_path)
    shutil.copy(args.config, f"{output_dir}/config.json")

    print("\nTraining complete.")
    print(f"Saved model: {final_model_path}")


if __name__ == "__main__":
    main()