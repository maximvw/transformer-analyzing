"""Compound tokenizer for graph connectivity task.

Vocabulary:
  - Special: <PAD>=0, <START>=1, <SEP>=2, <ANS>=3, <END>=4
  - Edge tokens: E(i,j) for all i < j, i,j in [0, max_N)
  - Query tokens: Q(i,j) for all i != j, i,j in [0, max_N)
  - Answer tokens: 0, 1
"""

from __future__ import annotations


SPECIAL_TOKENS = {"<PAD>": 0, "<START>": 1, "<SEP>": 2, "<ANS>": 3, "<END>": 4}
PAD_ID = 0


class GraphTokenizer:
    def __init__(self, max_n: int = 30):
        self.max_n = max_n
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        self.token2id = dict(SPECIAL_TOKENS)

        idx = len(SPECIAL_TOKENS)

        # Answer tokens
        for ans in ["0", "1"]:
            self.token2id[ans] = idx
            idx += 1

        # Edge tokens E(i,j) with i < j
        for i in range(self.max_n):
            for j in range(i + 1, self.max_n):
                self.token2id[f"E({i},{j})"] = idx
                idx += 1

        # Query tokens Q(i,j) for all i != j
        for i in range(self.max_n):
            for j in range(self.max_n):
                if i != j:
                    self.token2id[f"Q({i},{j})"] = idx
                    idx += 1

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode_edge(self, u: int, v: int) -> int:
        """Encode edge with canonical ordering (min, max)."""
        a, b = min(u, v), max(u, v)
        return self.token2id[f"E({a},{b})"]

    def encode_query(self, u: int, v: int) -> int:
        return self.token2id[f"Q({u},{v})"]

    def encode_answer(self, label: int) -> int:
        return self.token2id[str(label)]

    def encode_sequence(
        self,
        edges: list[tuple[int, int]],
        query: tuple[int, int],
        label: int,
    ) -> list[int]:
        """Encode full sequence: <START> E... <SEP> Q <ANS> {0|1} <END>"""
        ids = [self.token2id["<START>"]]
        for u, v in edges:
            ids.append(self.encode_edge(u, v))
        ids.append(self.token2id["<SEP>"])
        ids.append(self.encode_query(*query))
        ids.append(self.token2id["<ANS>"])
        ids.append(self.encode_answer(label))
        ids.append(self.token2id["<END>"])
        return ids

    def get_edge_positions(self, seq_len_without_special: int) -> list[int]:
        """Edge positions are indices 1..M (after <START>)."""
        return list(range(1, seq_len_without_special + 1))

    def get_answer_position(self, num_edges: int) -> int:
        """Position of <ANS> token = 1 (START) + num_edges + 1 (SEP) + 1 (Q)."""
        return num_edges + 3

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2token.get(i, f"[UNK:{i}]") for i in ids)
