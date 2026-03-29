"""Evaluation pipeline for graph connectivity models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import GraphConnectivityDataset, collate_fn
from .model import GraphGPT, DSUProbe, compute_losses
from .tokenizer import GraphTokenizer


@torch.no_grad()
def evaluate_dataset(
    model: GraphGPT,
    probe: DSUProbe | None,
    loader: DataLoader,
    lambda_state: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataset, return metrics."""
    model.eval()
    if probe is not None:
        probe.eval()

    total_loss = 0.0
    total_lm = 0.0
    total_state = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in loader:
        result = compute_losses(model, probe, batch, lambda_state, device)
        total_loss += result["loss"].item()
        total_lm += result["loss_lm"].item()
        total_state += result["loss_state"].item()

        preds = result["answer_logits"].argmax(dim=-1)
        correct += (preds == batch["targets"].to(device)).sum().item()
        total += batch["targets"].size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_lm": total_lm / max(n_batches, 1),
        "loss_state": total_state / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
        "n_samples": total,
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    tokenizer: GraphTokenizer,
    device: torch.device,
    d_model: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    max_seq_len: int = 128,
    max_n: int = 30,
) -> tuple[GraphGPT, DSUProbe | None]:
    """Load model (and optionally probe) from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GraphGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
        max_n=max_n,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    probe = None
    if "probe_state_dict" in ckpt:
        probe = DSUProbe(
            n_head=n_head,
            d_head=d_model // n_head,
            max_n=max_n,
        ).to(device)
        probe.load_state_dict(ckpt["probe_state_dict"])

    return model, probe


def evaluate_all(
    checkpoint_path: str,
    data_dir: str,
    max_n: int = 30,
    batch_size: int = 64,
    lambda_state: float = 1.0,
    d_model: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
) -> dict[str, dict]:
    """Evaluate on all test sets (ID + OOD)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GraphTokenizer(max_n=max_n)

    model, probe = load_checkpoint(
        checkpoint_path, tokenizer, device,
        d_model=d_model, n_layer=n_layer, n_head=n_head, max_n=max_n,
    )

    data_dir = Path(data_dir)
    results = {}

    # Find all test JSON files
    test_files = sorted(data_dir.glob("test*.json"))
    for test_file in test_files:
        name = test_file.stem  # e.g., "test_id", "test_ood_large_n", etc.
        ds = GraphConnectivityDataset(test_file, tokenizer, max_n=max_n, fixed=True)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        metrics = evaluate_dataset(model, probe, loader, lambda_state, device)
        results[name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.4f} loss={metrics['loss']:.4f} (n={metrics['n_samples']})")

    return results
