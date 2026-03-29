"""Dataset and DataLoader for graph connectivity with on-the-fly augmentation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dsu import DSU, compute_dsu_states
from .tokenizer import GraphTokenizer, PAD_ID


class GraphConnectivityDataset(Dataset):
    """Dataset with on-the-fly augmentation (shuffle edges + random query).

    For train: random shuffle + random query each __getitem__ call.
    For val/test: fixed (precomputed) shuffle + query for reproducibility.
    """

    def __init__(
        self,
        graphs_path: str | Path,
        tokenizer: GraphTokenizer,
        max_n: int = 30,
        fixed: bool = False,
    ):
        """
        Args:
            graphs_path: path to JSON file with graphs
            tokenizer: compound tokenizer
            max_n: max number of vertices (for padding)
            fixed: if True, use precomputed shuffle/query (val/test mode)
        """
        self.tokenizer = tokenizer
        self.max_n = max_n
        self.fixed = fixed

        with open(graphs_path) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        n = item["n"]
        edges = [tuple(e) for e in item["edges"]]

        if self.fixed:
            # Val/test: use precomputed order, query, label
            shuffled = [tuple(e) for e in item["shuffled_edges"]]
            query = tuple(item["query"])
            label = item["label"]
        else:
            # Train: random shuffle + random query
            shuffled = edges[:]
            random.shuffle(shuffled)
            query, label = self._sample_query(edges, n)

        # Compute DSU states for the shuffled order
        comp_states = compute_dsu_states(shuffled, n)

        # Encode sequence
        input_ids = self.tokenizer.encode_sequence(shuffled, query, label)
        num_edges = len(shuffled)
        ans_pos = self.tokenizer.get_answer_position(num_edges)

        # Pad comp_states to [M, max_n] (M = num_edges)
        comp_padded = torch.zeros(num_edges, self.max_n, dtype=torch.long)
        for t, comp in enumerate(comp_states):
            comp_padded[t, :n] = torch.tensor(comp, dtype=torch.long)

        # Vertex mask: 1 for real vertices, 0 for padding
        vertex_mask = torch.zeros(self.max_n, dtype=torch.bool)
        vertex_mask[:n] = True

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target": torch.tensor(label, dtype=torch.long),
            "ans_pos": ans_pos,
            "num_edges": num_edges,
            "comp_states": comp_padded,       # [M, max_n]
            "vertex_mask": vertex_mask,        # [max_n]
            "num_vertices": n,
        }

    def _sample_query(
        self, edges: list[tuple[int, int]], n: int
    ) -> tuple[tuple[int, int], int]:
        """Sample a balanced query (50% reachable, 50% not)."""
        # Build connectivity via DSU
        dsu = DSU(n)
        for u, v in edges:
            dsu.union(u, v)

        # Split vertex pairs by reachability
        reachable = []
        unreachable = []
        for i in range(n):
            for j in range(i + 1, n):
                if dsu.find(i) == dsu.find(j):
                    reachable.append((i, j))
                else:
                    unreachable.append((i, j))

        # 50/50 balance
        if reachable and unreachable and random.random() < 0.5:
            u, v = random.choice(unreachable)
            label = 0
        elif reachable:
            u, v = random.choice(reachable)
            label = 1
        else:
            u, v = random.choice(unreachable)
            label = 0

        # Randomly swap order of query vertices
        if random.random() < 0.5:
            u, v = v, u
        return (u, v), label


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length sequences with padding."""
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    max_edges = max(item["num_edges"] for item in batch)

    B = len(batch)
    input_ids = torch.full((B, max_seq_len), PAD_ID, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)
    ans_pos = torch.zeros(B, dtype=torch.long)
    num_edges = torch.zeros(B, dtype=torch.long)
    comp_states = torch.zeros(B, max_edges, batch[0]["comp_states"].size(1), dtype=torch.long)
    vertex_mask = torch.zeros(B, batch[0]["vertex_mask"].size(0), dtype=torch.bool)
    edge_mask = torch.zeros(B, max_edges, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        m = item["num_edges"]
        input_ids[i, :seq_len] = item["input_ids"]
        targets[i] = item["target"]
        ans_pos[i] = item["ans_pos"]
        num_edges[i] = m
        comp_states[i, :m] = item["comp_states"]
        vertex_mask[i] = item["vertex_mask"]
        edge_mask[i, :m] = True

    return {
        "input_ids": input_ids,        # [B, max_seq_len]
        "targets": targets,            # [B]
        "ans_pos": ans_pos,            # [B]
        "num_edges": num_edges,        # [B]
        "comp_states": comp_states,    # [B, max_edges, max_n]
        "vertex_mask": vertex_mask,    # [B, max_n]
        "edge_mask": edge_mask,        # [B, max_edges]
    }
