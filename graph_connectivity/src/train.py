"""Training loop for graph connectivity (SFT and Auxiliary Loss modes)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import GraphConnectivityDataset, collate_fn
from .model import GraphGPT, DSUProbe, compute_losses
from .tokenizer import GraphTokenizer


def train_epoch(
    model: GraphGPT,
    probe: DSUProbe | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_state: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    if probe is not None:
        probe.train()

    total_loss = 0.0
    total_lm = 0.0
    total_state = 0.0
    correct = 0
    total = 0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        result = compute_losses(model, probe, batch, lambda_state, device)
        result["loss"].backward()

        # Gradient clipping
        params = list(model.parameters())
        if probe is not None:
            params += list(probe.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        optimizer.step()

        total_loss += result["loss"].item()
        total_lm += result["loss_lm"].item()
        total_state += result["loss_state"].item()

        preds = result["answer_logits"].argmax(dim=-1)
        correct += (preds == batch["targets"].to(device)).sum().item()
        total += batch["targets"].size(0)
        n_batches += 1

        pbar.set_postfix(loss=total_loss / n_batches, acc=correct / total)

    return {
        "loss": total_loss / n_batches,
        "loss_lm": total_lm / n_batches,
        "loss_state": total_state / n_batches,
        "accuracy": correct / total,
    }


@torch.no_grad()
def eval_epoch(
    model: GraphGPT,
    probe: DSUProbe | None,
    loader: DataLoader,
    lambda_state: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if probe is not None:
        probe.eval()

    total_loss = 0.0
    total_lm = 0.0
    total_state = 0.0
    correct = 0
    total = 0
    n_batches = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        result = compute_losses(model, probe, batch, lambda_state, device)
        total_loss += result["loss"].item()
        total_lm += result["loss_lm"].item()
        total_state += result["loss_state"].item()

        preds = result["answer_logits"].argmax(dim=-1)
        correct += (preds == batch["targets"].to(device)).sum().item()
        total += batch["targets"].size(0)
        n_batches += 1

        pbar.set_postfix(loss=total_loss / n_batches, acc=correct / total)

    return {
        "loss": total_loss / n_batches,
        "loss_lm": total_lm / n_batches,
        "loss_state": total_state / n_batches,
        "accuracy": correct / total,
    }


def save_checkpoint(
    model: GraphGPT,
    probe: DSUProbe | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if probe is not None:
        state["probe_state_dict"] = probe.state_dict()
    torch.save(state, save_dir / f"checkpoint_epoch{epoch}.pt")
    # Also save as "best" if needed
    torch.save(state, save_dir / "checkpoint_best.pt")


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = GraphTokenizer(max_n=args.max_n)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Data
    train_ds = GraphConnectivityDataset(
        args.train_path, tokenizer, max_n=args.max_n, fixed=False
    )
    val_ds = GraphConnectivityDataset(
        args.val_path, tokenizer, max_n=args.max_n, fixed=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} graphs, Val: {len(val_ds)} graphs")

    # Model
    model = GraphGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        max_n=args.max_n,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Probe (if using auxiliary loss)
    probe = None
    if args.lambda_state > 0:
        probe = DSUProbe(
            n_head=args.n_head,
            d_head=args.d_model // args.n_head,
            max_n=args.max_n,
        ).to(device)
        n_probe = sum(p.numel() for p in probe.parameters())
        print(f"Probe parameters: {n_probe:,}")

    # Optimizer
    params = list(model.parameters())
    if probe is not None:
        params += list(probe.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Training
    save_dir = Path(args.save_dir)
    best_val_loss = float("inf")
    patience_counter = 0
    log = []

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_epoch(
            model, probe, train_loader, optimizer,
            args.lambda_state, device, args.max_grad_norm,
        )
        val_metrics = eval_epoch(
            model, probe, val_loader, args.lambda_state, device,
        )

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": elapsed,
        }
        log.append(entry)

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping by val total loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(model, probe, optimizer, epoch, val_metrics, save_dir)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, probe, optimizer, epoch, val_metrics, save_dir / f"epoch_{epoch}")

    # Save training log
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return log


def main():
    parser = argparse.ArgumentParser(description="Train graph connectivity model")
    # Data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--max_n", type=int, default=30)
    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Training
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_state", type=float, default=0.0,
                        help="Weight for auxiliary DSU loss. 0 = SFT mode.")
    # Misc
    parser.add_argument("--save_dir", type=str, default="checkpoints/graph")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
