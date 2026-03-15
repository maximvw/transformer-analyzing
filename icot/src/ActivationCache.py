"""
Utility functions for grabbing the residual stream.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Generator,
    cast,
    Iterable,
    Mapping,
    MutableMapping,
)
from collections import defaultdict
from contextlib import contextmanager
import torch
from torch import Tensor, nn

PREFIX = "base_model.transformer.h."


@dataclass
class ActivationCache(Mapping[str, Tensor]):
    """
    A simple cache for activations.

    Hookable modules and their shapes:
    *.attn.hook_value_states:           [batch, seq, head_idx, d_head]
    *.attn.hook_attn_pattern:           [batch, head_idx, seq (query), seq (key)]
    *.attn.hook_attn_output_per_head:   [batch, seq, head_idx, d_model]
    *.attn                              [batch, seq, d_model]
    *.hook_resid_mid                    [batch, seq, d_model]
    *.mlp.hook_mlp_mid                  [batch, seq, d_mlp]
    *.mlp                               [batch, seq, d_model]
    *.hook_resid_post                   [batch, seq, d_model]
    """

    _data: MutableMapping[str, list[Tensor]]
    _prefix = PREFIX

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._data:
            key = self._prefix + key

        if key not in self._data:
            raise KeyError(f"Key {key} not found in activation cache.")
        return self._data[key][0]

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def summary(self):
        """Quick summary of activation shapes."""
        formatted = {
            name.replace(self._prefix, ""): acts[0].shape
            for name, acts in self._data.items()
        }
        renames = [
            name for name in formatted.keys() if len(name) == 1 and name.isdigit()
        ]
        for name in renames:
            formatted[f"{name}.hook_resid_post"] = formatted.pop(name)
        return formatted

    def __repr__(self):
        return (
            f"ActivationCache:\n"
            + "    "
            + "\n    ".join(
                [
                    f"{name.replace(self._prefix, '')}: {acts[0].shape}"
                    for name, acts in self._data.items()
                ]
            )
        )


def _untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    # Exact or HF-style prefix match
    for n, m in model.named_modules():
        if n == name or PREFIX + name == n:
            return m
    # Suffix match for custom models (e.g. layers.0.attn..., base.layers.0.attn..., etc.)
    for n, m in model.named_modules():
        if n.endswith("." + name) or n.endswith(name):
            return m
    raise LookupError(name)


def _create_read_hook(layer_name: str, records: dict[str, list[Tensor]]) -> Any:
    """Create a hook function that records the model activation at :layer_name:"""

    def hook_fn(_module: Any, _inputs: Any, _outputs: Any) -> Any:
        # _inputs[0]: [batch, seq, d_model]
        # _outputs[0]: [batch, seq, d_model],
        activation = _untuple_tensor(_outputs)
        if not isinstance(cast(Any, activation), Tensor):
            raise ValueError(
                f"Expected a Tensor reading model activations, got {type(activation)}"
            )

        _activation = activation.clone().detach()
        records[layer_name].append(_activation)
        return _outputs

    return hook_fn


@contextmanager
def record_activations(
    model: nn.Module,
    module_names: list[str],
) -> Generator[dict[str, list[Tensor]], None, None]:
    """
    Record the model activations at each layer of type `layer_type`.
    This function will record every forward pass through the model
    at all layers of the given layer_type.

    Args:
        model: The model to record activations from.
        modules: List of modules to grab activations from.
    Example:
    """
    recorded_activations: dict[int, list[Tensor]] = defaultdict(list)
    hooks = []

    for _module_name in module_names:
        module = _get_module(model, _module_name)

        if _module_name.startswith(PREFIX):
            _module_name = _module_name.replace(PREFIX, "")

        # hook_fn: hook(module, input, output)
        hook_fn = _create_read_hook(
            _module_name,
            recorded_activations,
        )
        handle = module.register_forward_hook(hook_fn)
        hooks.append(handle)

    try:
        yield ActivationCache(recorded_activations)
    finally:
        for hook in hooks:
            hook.remove()