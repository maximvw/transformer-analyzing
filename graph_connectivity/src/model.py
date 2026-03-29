"""GPT-2 from scratch with classification head and DSU auxiliary probe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_ID


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, d_head]
        q = q.transpose(1, 2)  # [B, H, T, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]

        # Causal mask
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask
        if attn_mask is not None:
            # attn_mask: [B, T], True = valid, False = pad
            pad_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_weights = attn_weights.masked_fill(pad_mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)  # [B, H, T, d_head]
        # Keep per-head outputs for probe
        head_outputs = attn_out  # [B, H, T, d_head]

        out = attn_out.transpose(1, 2).reshape(B, T, -1)  # [B, T, C]
        out = self.out_proj(out)
        return out, head_outputs


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, head_outputs = self.attn(self.ln1(x), attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, head_outputs


class GraphGPT(nn.Module):
    """GPT-2-style model for graph connectivity.

    Architecture:
        - Token + position embeddings
        - N transformer blocks
        - Classification head on <ANS> position
        - Optional: DSU probe on last layer attention outputs
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.0,
        max_n: int = 30,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_n = max_n
        self.d_head = d_model // n_head

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Classification head: hidden[ans_pos] -> 2 classes
        self.classifier = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, T]
            attn_mask: [B, T], True = valid token

        Returns:
            hidden: [B, T, d_model] final hidden states
            last_head_outputs: [B, H, T, d_head] attention head outputs from last layer
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        last_head_outputs = None
        for block in self.blocks:
            x, head_outputs = block(x, attn_mask)
            last_head_outputs = head_outputs

        hidden = self.ln_f(x)
        return hidden, last_head_outputs


class DSUProbe(nn.Module):
    """Linear probe: attention head output -> comp[] prediction.

    For each head h:
        logits = W_h @ attn_h(t) + b_h  -> reshape [max_n, max_n]
        Row i = max_n logits for classifying comp[i]
    """

    def __init__(self, n_head: int, d_head: int, max_n: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.max_n = max_n

        # One linear layer per head: d_head -> max_n * max_n
        self.probes = nn.ModuleList(
            [nn.Linear(d_head, max_n * max_n) for _ in range(n_head)]
        )

    def forward(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_outputs: [B, H, M, d_head] — attention outputs at edge positions

        Returns:
            logits: [B, H, M, max_n, max_n] — comp[] predictions
        """
        B, H, M, _ = head_outputs.shape
        all_logits = []
        for h in range(H):
            logits_h = self.probes[h](head_outputs[:, h])  # [B, M, max_n*max_n]
            logits_h = logits_h.view(B, M, self.max_n, self.max_n)
            all_logits.append(logits_h)
        return torch.stack(all_logits, dim=1)  # [B, H, M, max_n, max_n]


def compute_losses(
    model: GraphGPT,
    probe: DSUProbe | None,
    batch: dict,
    lambda_state: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Compute L_LM and L_state for a batch.

    Returns dict with 'loss', 'loss_lm', 'loss_state' tensors.
    """
    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)
    ans_pos = batch["ans_pos"].to(device)
    attn_mask = input_ids != PAD_ID

    B = input_ids.size(0)

    hidden, last_head_outputs = model(input_ids, attn_mask)

    # L_LM: classification on <ANS> position
    ans_hidden = hidden[torch.arange(B, device=device), ans_pos]  # [B, d_model]
    answer_logits = model.classifier(ans_hidden)  # [B, 2]
    loss_lm = F.cross_entropy(answer_logits, targets)

    result = {
        "loss_lm": loss_lm,
        "answer_logits": answer_logits.detach(),
    }

    # L_state: DSU probe loss
    if probe is not None and lambda_state > 0:
        comp_states = batch["comp_states"].to(device)   # [B, max_edges, max_n]
        vertex_mask = batch["vertex_mask"].to(device)    # [B, max_n]
        edge_mask = batch["edge_mask"].to(device)        # [B, max_edges]

        max_edges = comp_states.size(1)

        # Gather attention outputs at edge positions (positions 1..M)
        edge_positions = torch.arange(1, max_edges + 1, device=device)  # [max_edges]
        edge_head_out = last_head_outputs[:, :, edge_positions, :]  # [B, H, max_edges, d_head]

        probe_logits = probe(edge_head_out)  # [B, H, max_edges, max_n, max_n]

        # Compute masked cross-entropy
        # probe_logits: [B, H, M, max_n (vertex), max_n (class)]
        # comp_states:  [B, M, max_n] — target class for each vertex at each edge step
        H = probe_logits.size(1)

        loss_state = torch.tensor(0.0, device=device)
        count = 0
        for h in range(H):
            logits_h = probe_logits[:, h]  # [B, M, max_n, max_n]
            # Reshape for cross_entropy: [B*M*max_n, max_n] vs [B*M*max_n]
            logits_flat = logits_h.reshape(-1, probe.max_n)
            targets_flat = comp_states.reshape(-1)

            # Mask: valid edge AND valid vertex
            # edge_mask: [B, M] -> [B, M, 1] -> [B, M, max_n]
            # vertex_mask: [B, max_n] -> [B, 1, max_n]
            combined_mask = edge_mask.unsqueeze(-1) & vertex_mask.unsqueeze(1)  # [B, M, max_n]
            mask_flat = combined_mask.reshape(-1)

            ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss_state = loss_state + (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            count += 1

        loss_state = loss_state / count
        result["loss_state"] = loss_state
        result["loss"] = loss_lm + lambda_state * loss_state
    else:
        result["loss_state"] = torch.tensor(0.0, device=device)
        result["loss"] = loss_lm

    return result
