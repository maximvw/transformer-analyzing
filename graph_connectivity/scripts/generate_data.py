"""Generate graph datasets for training and evaluation.

Graph types (train):
  - Erdos-Renyi G(N, p), p in {0.05, 0.10, 0.15, 0.20, 0.30}  — 50%
  - Random trees (Prufer sequence)                                — 20%
  - Sparse random (m = N*k edges, k in {1.0, 1.5, 2.0})         — 20%
  - Complete graph K_n, n in {3,4,5}                              — 5%
  - Path graph 0-1-2-..-(N-1)                                    — 5%

OOD test sets:
  - Large N (N in {25, 30})
  - Cyclic grids
  - Adversarial degree (hub + isolated)
  - Long diameter (long chains + noise)
  - Disconnected dense (two dense components)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from graph_connectivity.src.dsu import DSU


# --- Graph generators ---

def gen_erdos_renyi(n: int, p: float) -> list[tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges.append((i, j))
    return edges


def gen_random_tree(n: int) -> list[tuple[int, int]]:
    """Random tree via Prufer sequence."""
    if n <= 2:
        return [(0, 1)] if n == 2 else []
    prufer = [random.randint(0, n - 1) for _ in range(n - 2)]
    degree = [1] * n
    for v in prufer:
        degree[v] += 1
    edges = []
    for v in prufer:
        for u in range(n):
            if degree[u] == 1:
                edges.append((min(u, v), max(u, v)))
                degree[u] -= 1
                degree[v] -= 1
                break
    # Last edge: two remaining vertices with degree 1
    remaining = [u for u in range(n) if degree[u] == 1]
    if len(remaining) == 2:
        edges.append((min(remaining), max(remaining)))
    return edges


def gen_sparse_random(n: int, k: float) -> list[tuple[int, int]]:
    """Random graph with m = n*k edges."""
    m = int(n * k)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = min(m, len(all_edges))
    return random.sample(all_edges, m)


def gen_complete(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def gen_path(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def gen_cyclic_grid(rows: int, cols: int) -> tuple[int, list[tuple[int, int]]]:
    """Grid graph with optional wraparound."""
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    # Add cycle edges (wrap around)
    if cols > 2:
        for r in range(rows):
            edges.append((r * cols, r * cols + cols - 1))
    return n, edges


def gen_adversarial_degree(n: int) -> list[tuple[int, int]]:
    """Hub connected to half the vertices; other half forms isolated pairs.
    High-degree hub is NOT connected to the isolated component."""
    hub = 0
    half = n // 2
    edges = []
    # Hub connects to vertices 1..half-1
    for v in range(1, half):
        edges.append((0, v))
    # Isolated pairs in the second half
    for i in range(half, n - 1, 2):
        edges.append((i, i + 1))
    return edges


def gen_long_diameter(n: int) -> list[tuple[int, int]]:
    """Long chain + random noise edges that don't shortcut the chain."""
    # Main chain
    edges = [(i, i + 1) for i in range(n - 1)]
    # Add a few noise edges (not shortcutting much)
    for _ in range(n // 4):
        u = random.randint(0, n - 2)
        v = u + random.choice([1, 2]) if u + 2 < n else u + 1
        if v < n:
            e = (min(u, v), max(u, v))
            if e not in edges:
                edges.append(e)
    return edges


def gen_disconnected_dense(n: int) -> list[tuple[int, int]]:
    """Two dense components with no edges between them."""
    half = n // 2
    edges = []
    # Component 1: dense (p=0.7)
    for i in range(half):
        for j in range(i + 1, half):
            if random.random() < 0.7:
                edges.append((i, j))
    # Component 2: dense (p=0.7)
    for i in range(half, n):
        for j in range(i + 1, n):
            if random.random() < 0.7:
                edges.append((i, j))
    return edges


# --- Dataset assembly ---

def sample_query_fixed(edges: list[tuple[int, int]], n: int, balance: bool = True):
    """Sample a query with fixed seed for reproducibility."""
    dsu = DSU(n)
    for u, v in edges:
        dsu.union(u, v)

    reachable = []
    unreachable = []
    for i in range(n):
        for j in range(i + 1, n):
            if dsu.find(i) == dsu.find(j):
                reachable.append((i, j))
            else:
                unreachable.append((i, j))

    if balance and unreachable and reachable and random.random() < 0.5:
        u, v = random.choice(unreachable)
        label = 0
    elif reachable:
        u, v = random.choice(reachable)
        label = 1
    elif unreachable:
        u, v = random.choice(unreachable)
        label = 0
    else:
        u, v = 0, 1
        label = 1

    if random.random() < 0.5:
        u, v = v, u
    return (u, v), label


def generate_train_graphs(
    n_graphs: int,
    n_values: list[int],
    seed: int = 42,
) -> list[dict]:
    """Generate training graphs with specified type distribution."""
    random.seed(seed)
    graphs = []

    for _ in range(n_graphs):
        n = random.choice(n_values)
        r = random.random()

        if r < 0.50:
            p = random.choice([0.05, 0.10, 0.15, 0.20, 0.30])
            edges = gen_erdos_renyi(n, p)
            gtype = f"erdos_renyi_p{p}"
        elif r < 0.70:
            edges = gen_random_tree(n)
            gtype = "random_tree"
        elif r < 0.90:
            k = random.choice([1.0, 1.5, 2.0])
            edges = gen_sparse_random(n, k)
            gtype = f"sparse_k{k}"
        elif r < 0.95:
            n = random.choice([3, 4, 5])
            edges = gen_complete(n)
            gtype = "complete"
        else:
            edges = gen_path(n)
            gtype = "path"

        graphs.append({"n": n, "edges": [list(e) for e in edges], "type": gtype})

    return graphs


def generate_fixed_dataset(graphs: list[dict], seed: int = 0) -> list[dict]:
    """Add fixed shuffle + query to each graph for val/test."""
    random.seed(seed)
    fixed = []
    for g in graphs:
        n = g["n"]
        edges = [tuple(e) for e in g["edges"]]

        shuffled = edges[:]
        random.shuffle(shuffled)

        query, label = sample_query_fixed(edges, n)

        fixed.append({
            "n": n,
            "edges": [list(e) for e in edges],
            "shuffled_edges": [list(e) for e in shuffled],
            "query": list(query),
            "label": label,
            "type": g.get("type", "unknown"),
        })
    return fixed


def generate_ood_datasets(seed: int = 100) -> dict[str, list[dict]]:
    """Generate all OOD test sets."""
    random.seed(seed)
    ood = {}

    # OOD: Large N
    large_n_graphs = []
    for _ in range(1000):
        n = random.choice([25, 30])
        p = random.choice([0.05, 0.10, 0.15, 0.20])
        edges = gen_erdos_renyi(n, p)
        large_n_graphs.append({"n": n, "edges": [list(e) for e in edges], "type": "large_n"})
    ood["test_ood_large_n"] = generate_fixed_dataset(large_n_graphs, seed=101)

    # OOD: Cyclic grids
    grid_graphs = []
    for _ in range(1000):
        rows = random.randint(3, 6)
        cols = random.randint(3, 6)
        n, edges = gen_cyclic_grid(rows, cols)
        if n <= 30:
            grid_graphs.append({"n": n, "edges": [list(e) for e in edges], "type": "cyclic_grid"})
    ood["test_ood_cyclic_grid"] = generate_fixed_dataset(grid_graphs[:1000], seed=102)

    # OOD: Adversarial degree
    adv_graphs = []
    for _ in range(1000):
        n = random.choice([10, 15, 20, 25])
        edges = gen_adversarial_degree(n)
        adv_graphs.append({"n": n, "edges": [list(e) for e in edges], "type": "adversarial_degree"})
    ood["test_ood_adversarial"] = generate_fixed_dataset(adv_graphs, seed=103)

    # OOD: Long diameter
    long_graphs = []
    for _ in range(1000):
        n = random.choice([15, 20, 25, 30])
        edges = gen_long_diameter(n)
        long_graphs.append({"n": n, "edges": [list(e) for e in edges], "type": "long_diameter"})
    ood["test_ood_long_diameter"] = generate_fixed_dataset(long_graphs, seed=104)

    # OOD: Disconnected dense
    disc_graphs = []
    for _ in range(1000):
        n = random.choice([10, 14, 18, 20])
        edges = gen_disconnected_dense(n)
        disc_graphs.append({"n": n, "edges": [list(e) for e in edges], "type": "disconnected_dense"})
    ood["test_ood_disconnected"] = generate_fixed_dataset(disc_graphs, seed=105)

    return ood


def main():
    parser = argparse.ArgumentParser(description="Generate graph connectivity datasets")
    parser.add_argument("--output_dir", type=str, default="graph_connectivity/data")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_values_train = [8, 10, 12, 15, 20]
    n_total = args.n_train + args.n_val + args.n_test

    print(f"Generating {n_total} graphs (single pool, then split)...")
    all_graphs = generate_train_graphs(n_total, n_values_train, seed=args.seed)

    # Deterministic shuffle and split — guarantees no overlap
    random.seed(args.seed)
    random.shuffle(all_graphs)

    train_graphs = all_graphs[:args.n_train]
    val_graphs = all_graphs[args.n_train:args.n_train + args.n_val]
    test_graphs = all_graphs[args.n_train + args.n_val:]

    print(f"Saving train ({len(train_graphs)})...")
    with open(out / "train.json", "w") as f:
        json.dump(train_graphs, f)

    print(f"Saving val ({len(val_graphs)})...")
    val_fixed = generate_fixed_dataset(val_graphs, seed=args.seed + 100)
    with open(out / "val.json", "w") as f:
        json.dump(val_fixed, f)

    print(f"Saving test ID ({len(test_graphs)})...")
    test_fixed = generate_fixed_dataset(test_graphs, seed=args.seed + 200)
    with open(out / "test_id.json", "w") as f:
        json.dump(test_fixed, f)

    print("Generating OOD test sets...")
    ood = generate_ood_datasets(seed=args.seed + 1000)
    for name, data in ood.items():
        with open(out / f"{name}.json", "w") as f:
            json.dump(data, f)
        print(f"  Saved {len(data)} {name} examples")

    print("Done!")


if __name__ == "__main__":
    main()
