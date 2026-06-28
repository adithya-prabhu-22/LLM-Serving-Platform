import time
import torch
from torch.amp import autocast
from training.trainer.loss import compute_loss
from training.trainer.evaluator import evaluate
from training.utils.checkpoint import save_checkpoint
from training.utils.safetensor import save_safetensor

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    global_step,
    eval_dataloader=None,
    eval_interval=500,
    save_interval=1000,
    checkpoint_dir=None,
    output_dir=None,
    max_grad_norm=1.0,
    best_val_loss=float("inf"),
    gradient_accumulation_steps=1,
    scaler=None,
    logging_steps=50,
):
    if scaler is None:
        raise ValueError("GradScaler must be provided.")

    model.train()
    running_loss = 0.0
    logged_microbatches = 0
    total_loss = 0.0
    total_microbatches = 0
    start_time = time.time()

    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    optimizer.zero_grad(set_to_none=True)
    total_batches = len(dataloader)

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_cuda:
            with autocast(device_type="cuda"):
                logits = model(input_ids)
                raw_loss = compute_loss(logits, targets)
                loss = raw_loss / gradient_accumulation_steps
        else:
            logits = model(input_ids)
            raw_loss = compute_loss(logits, targets)
            loss = raw_loss / gradient_accumulation_steps

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")

        scaler.scale(loss).backward()

        running_loss += raw_loss.item()
        total_loss += raw_loss.item()
        logged_microbatches += 1
        total_microbatches += 1

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1 == total_batches):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if scheduler is not None:
                scheduler.step()

            if global_step % logging_steps == 0:
                avg_loss = running_loss / logged_microbatches
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch}] "
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
                running_loss = 0.0
                logged_microbatches = 0
                start_time = time.time()

            if eval_dataloader is not None and global_step % eval_interval == 0:
                metrics = evaluate(
                    model=model,
                    dataloader=eval_dataloader,
                    device=device,
                )
                val_loss = metrics["loss"]
                print(
                    f"[Eval] "
                    f"Step {global_step} | "
                    f"Loss: {val_loss:.4f} | "
                    f"Perplexity: "
                    f"{metrics['perplexity']:.2f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if output_dir is not None:
                        save_safetensor(
                            model,
                            f"{output_dir}/best_model.safetensors",
                        )
                        print(f"New best model saved (val_loss={val_loss:.4f})")
                        if checkpoint_dir is not None:
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                epoch=epoch - 1,
                                step=global_step,
                                loss=val_loss,
                                best_val_loss=best_val_loss,
                                checkpoint_dir=checkpoint_dir,
                                checkpoint_name="best_checkpoint.pt",
                            )
                            print(f"Best checkpoint saved at step {global_step}")

            if checkpoint_dir is not None and global_step % save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch - 1,
                    step=global_step,
                    loss=raw_loss.item(),
                    best_val_loss=best_val_loss,
                    checkpoint_dir=checkpoint_dir,
                )
                print(f"Checkpoint saved at step {global_step}")

                if output_dir is not None:
                    save_safetensor(
                        model,
                        f"{output_dir}/model_step_{global_step}.safetensors",
                    )
                    print(f"Safetensor exported at step {global_step}")

    avg_loss_over_epoch = total_loss / total_microbatches if total_microbatches > 0 else 0.0
    return global_step, best_val_loss, avg_loss_over_epoch