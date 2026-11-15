"""Resolve the GameWorld semantic binding payload at the local-action boundary.

This tiny primitive is also imported in the isolated Python worker, where the
vendor/GameWorld package is deliberately absent. Its behavior follows the
catalog's cell_bindings and parameter conventions.
"""

from collections.abc import Mapping
from typing import Any


def bind_semantic_action(
    binding: Mapping[str, Any], parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params = parameters or {}
    if not isinstance(params, Mapping):
        raise ValueError("Action parameters must be an object")
    replaced = set(params).intersection(binding)
    if replaced:
        raise ValueError(f"Cannot replace fixed action fields: {sorted(replaced)}")
    mapped = dict(binding)
    mapped.update({key: value for key, value in params.items() if value is not None})
    cells = mapped.get("cell_bindings")
    if isinstance(cells, Mapping):
        cell = str(params.get("cell") or "").strip().lower()
        coords = cells.get(cell)
        if not isinstance(coords, Mapping) or "x" not in coords or "y" not in coords:
            raise ValueError(f"Unknown or missing native cell binding: {cell!r}")
        mapped.update(x=coords["x"], y=coords["y"])
        mapped.pop("cell_bindings", None)
        mapped.pop("cell_param", None)
        mapped.pop("cell", None)
    return mapped
