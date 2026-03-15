import torch
from collections import defaultdict
from typing import Optional, Dict
from contextlib import contextmanager
from fancy_einsum import einsum
from src.ActivationCache import ActivationCache

# Registry for active hook handles: model_id -> { component_name: handle }
_HOOK_HANDLES: Dict[int, Dict[str, torch.utils.hooks.RemovableHandle]] = {}
PREFIX = "base_model.transformer.h."


def _find_module(model, component: str) -> torch.nn.Module:
    full_name = component if component.startswith(PREFIX) else PREFIX + component
    try:
        return model.get_submodule(full_name)
    except AttributeError:
        raise ValueError(f"Module named '{full_name}' not found")


def set_intervention(
    model,
    component: str,
    mode: str,
    records: dict[str, list[torch.Tensor]],
    vector: Optional[torch.Tensor] = None,
    proj_vec: Optional[torch.Tensor] = None,
    position: Optional[int] = None,
    head: Optional[int] = None,
    read_pos: Optional[int] = None,
) -> None:
    """
    Apply an intervention on `component` of `model`.

    mode:
      - 'off':     zero out outputs
      - 'replace': use `vector` instead
      - 'add':     add `vector`
      - 'knockout': prevent attention heads from moving info between two token positions
      - 'proj_add': project out a vector and add second vector

    `position`: index along the sequence axis (if provided).
    `head`: for 'hook_attn_output_per_head', index of the attention head (if provided).

    Sequence axis is 2 for 'hook_attn_pattern', else 1.
    Head axis for 'hook_attn_output_per_head' is 2.
    """
    # cleanup any existing hook
    clear_intervention(model, component, position, head)
    module = _find_module(model, component)

    # determine axes
    seq_axis = 2 if component.endswith("hook_attn_pattern") else 1
    head_axis = 2 if component.endswith("hook_attn_output_per_head") else None

    def transform(x: torch.Tensor) -> torch.Tensor:
        if mode == "noop":
            # No-op case, just return the original tensor
            # This is useful for recording activations without modification.
            return x

        if mode == "knockout":
            r, c = position, read_pos
            if r is None:
                raise ValueError("must supply both write_pos and read_pos for knockout")

            # y shape: [batch, heads, seq_q, seq_k]
            y = x.clone()

            if head is None:
                y[:, :, r, c] = float(0)
            else:
                y[:, head, r, c] = float(0)

            # renormalize each [batch, head, r, :] row to sum to 1
            row_sum = y.sum(dim=-1, keepdim=True)

            return y / (row_sum + 1e-12)

        # full-tensor cases
        if mode == "off" and position is None and head is None:
            return torch.zeros_like(x)
        if mode == "replace" and position is None and head is None:
            assert vector is not None, "Replace mode requires vector"
            return vector.expand_as(x)
        if mode == "add" and position is None and head is None:
            assert vector is not None, "Add mode requires vector"
            return x + vector
        if mode == "proj_add" and position is None and head is None:
            assert vector is not None, "Add mode requires vector"
            assert proj_vec is not None, "Add mode requires a projection vector"
            # normalize projection and patch directions
            unit_proj = proj_vec / proj_vec.norm()
            # scalar projection coefficient for each “row”
            # x: [batch, seq, d_model]
            coeffs = einsum(
                "batch seq d_model, d_model -> batch seq",
                x,
                unit_proj,
            )
            # build projection to remove and insertion of equal magnitude
            proj = coeffs.unsqueeze(-1) * unit_proj
            return x - proj + vector

        # slicing case: build slice indices
        sl = [slice(None)] * x.ndim
        if position is not None:
            sl[seq_axis] = position
        if head is not None:
            # only valid for hook_attn_output_per_head
            assert head_axis is not None, "Head index only applies to per-head hooks"
            sl[head_axis] = head
        y = x.clone()
        # apply transform on the selected slice
        target = x[tuple(sl)]

        if mode == "off":
            y[tuple(sl)] = torch.zeros_like(target)
        elif mode == "replace":
            assert vector is not None, "Replace mode requires vector"
            y[tuple(sl)] = vector.expand_as(target)
        elif mode == "add":
            assert vector is not None, "Add mode requires vector"
            y[tuple(sl)] = target + vector
        elif mode == "proj_add":
            assert vector is not None, "Proj_add mode requires vector"
            assert proj_vec is not None, "Proj_add mode requires a projection vector"
            unit_proj = proj_vec / proj_vec.norm()
            coeffs = einsum(
                "batch d_model, d_model -> batch",
                target,
                unit_proj,
            )
            proj = coeffs.unsqueeze(-1) * unit_proj
            y[tuple(sl)] = (target - proj) + vector
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        return y

    def _hook_fn(_mod, _inp, output):
        # only transform primary tensor in tuples
        if isinstance(output, tuple):
            first, *rest = output

            records[f"{component}_orig"].append(first.clone().detach())
            transformed = transform(first)
            records[f"{component}_interved"].append(transformed.clone().detach())
            return (transformed, *rest)

        records[f"{component}_orig"].append(output.clone().detach())
        transformed = transform(output)
        records[f"{component}_interved"].append(transformed.clone().detach())
        return transformed

    handle = module.register_forward_hook(_hook_fn)
    key = f"{component}@pos{position}@head{head}"
    _HOOK_HANDLES.setdefault(id(model), {})[key] = handle


def clear_intervention(
    model,
    component: str,
    position: Optional[int] = None,
    head: Optional[int] = None,
) -> None:
    """
    Remove any intervention on `component` (and optional `position`/`head`).
    """
    key = f"{component}@pos{position}@head{head}"
    mh = _HOOK_HANDLES.get(id(model), {})
    handle = mh.pop(key, None)
    if handle:
        handle.remove()
    if not mh:
        _HOOK_HANDLES.pop(id(model), None)


@contextmanager
def intervention(model, specs):
    """
    Temporarily applies one or more interventions to the given model during its forward pass,
    disabling gradients for the duration and ensuring interventions are cleared afterwards.

    Parameters:
        model: The model instance to intervene on.
        specs: A list of dicts, each specifying one intervention with keys:
            - component (str): Hook name, e.g. '3.attn.hook_attn_output_per_head'.
            - mode (str): Intervention mode, one of 'add', 'replace', 'off', etc.
            - vector (Tensor, optional): Tensor to add or replace with; required for 'add'/'replace'.
            - position (int or None): Sequence position to target, or None for all positions.
            - head (int or None): Attention head index, or None for all heads.

    Yields:
        None. Place model calls inside the `with` block to apply the interventions.
    """
    recorded_activations = defaultdict(list)
    # Apply each intervention spec
    for spec in specs:
        set_intervention(
            model,
            spec["component"],
            spec["mode"],
            recorded_activations,
            spec.get("vector"),
            spec.get("proj_vec"),
            spec.get("position"),
            spec.get("head"),
            spec.get("read_pos"),
        )

    try:
        with torch.no_grad():
            yield ActivationCache(recorded_activations)
    finally:
        # Clear each intervention
        for spec in specs:
            clear_intervention(
                model, spec["component"], spec.get("position"), spec.get("head")
            )


# =========================
# Example usage:
# =========================

# Single intervention
# spec = {
#     "component": "1.attn.hook_attn_output_per_head",
#     "mode": "add",
#     "vector": delta,
#     "position": -1,
#     "head": None
# }
# with intervention(model, [spec]):
#     outputs = model(input_ids)

# Multiple interventions
# specs = [
#     {
#         "component": "1.attn.hook_attn_pattern",
#         "mode": "off",
#         "vector": None,
#         "position": None,
#         "head": None
#     },
#     {
#         "component": "5.mlp.hook_mlp_mid",
#         "mode": "replace",
#         "vector": vec,
#         "position": None,
#         "head": None
#     }
# ]
# with intervention(model, specs):
#     logits = model(input_ids)
