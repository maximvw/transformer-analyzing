"""Disjoint Set Union with union-by-min for deterministic component IDs."""

from __future__ import annotations


class DSU:
    """DSU with union-by-min: root of each component is always the minimum vertex."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, u: int, v: int) -> bool:
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False
        # union by min: smaller root becomes the root
        if ru > rv:
            ru, rv = rv, ru
        self.parent[rv] = ru
        return True

    def comp(self) -> list[int]:
        """Return comp[i] = Find(i) for all vertices."""
        return [self.find(i) for i in range(len(self.parent))]


def compute_dsu_states(edges: list[tuple[int, int]], n: int) -> list[list[int]]:
    """Compute DSU comp[] state after each edge.

    Returns list of length len(edges), where states[t] is comp[] after processing
    edges[0..t].
    """
    dsu = DSU(n)
    states = []
    for u, v in edges:
        dsu.union(u, v)
        states.append(dsu.comp())
    return states
