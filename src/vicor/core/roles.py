"""GameWorld catalog adapters for one learning main that controls every role.

Derive actions, role ownership and descriptions from the existing catalog.
No game names, colors, mechanics or task outcomes are special-cased here.
"""

from dataclasses import dataclass
from typing import Any

from catalog.games import load_game
from catalog.games._base import RoleControls
from .semantic_binding import bind_semantic_action


@dataclass
class GameWorldRoleProfile:
    physical: RoleControls
    semantic: dict[str, dict[str, Any]]
    specs: dict[str, dict[str, Any]]
    role_names: tuple[str, ...]
    rules_text: str


def build_gameworld_role_profile(config) -> GameWorldRoleProfile:
    roles = load_game(config.game_id).game_roles
    if not roles or len(roles) != config.agent_count:
        raise ValueError("GameWorld roles do not match the runtime configuration")
    if not (len(config.role_controls_maps) == len(config.semantic_controls_maps)
            == len(config.semantic_controls_specs) == len(config.system_prompts)
            == len(roles)):
        raise ValueError("GameWorld role-specific control data is incomplete")

    multi = len(roles) > 1
    physical = config.role_controls_maps[0].copy()
    semantic: dict[str, dict[str, Any]] = {}
    specs: dict[str, dict[str, Any]] = {}
    names = []
    for index, role in enumerate(roles):
        name = role.name
        if not name or name in names or "." in name:
            raise ValueError(f"Invalid or duplicate GameWorld role name: {name!r}")
        names.append(name)
        controls = config.role_controls_maps[index]
        physical.allowed_keys.update(controls.allowed_keys)
        physical.allow_clicks = physical.allow_clicks or controls.allow_clicks
        physical.key_durations.update(controls.key_durations)
        for spec in config.semantic_controls_specs[index]:
            action_id = spec["id"]
            key = f"{name}.{action_id}" if multi else action_id
            if key in semantic:
                raise ValueError(f"Duplicate GameWorld action {key!r}")
            binding = config.semantic_controls_maps[index][action_id]
            semantic[key] = dict(binding)
            specs[key] = {**spec, "id": key, "role": name}

    if multi:
        # Official GameWorld role prompts retain task requirements and hazards.
        # All characters share the same rendered screen and keyboard.
        rules = (
            "You are the single MAIN decision-maker for every game role below. "
            "These characters share one environment. Control each role using the "
            "role-qualified semantic action IDs below, or the permitted physical "
            "keys (including combinations). Decide their cooperation yourself.\n\n"
            + "\n\n".join(
                f"=== GameWorld role: {role.name} ===\n"
                + config.system_prompts[i].split("# Output Format")[0]
                for i, role in enumerate(roles)
            )
        )
    else:
        rules = config.system_prompts[0].split("# Output Format")[0]
    rules += (
        "\nPermitted physical keys: " + str(sorted(physical.allowed_keys))
        + "\nMouse clicks permitted: " + str(physical.allow_clicks)
        + "\nRegistered semantic actions: " + str(list(specs.values()))
    )
    return GameWorldRoleProfile(
        physical=physical, semantic=semantic, specs=specs,
        role_names=tuple(names), rules_text=rules,
    )


def resolve_semantic_action(
    binding: dict[str, Any], spec: dict[str, Any], parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply only catalog-declared dynamic arguments, preserving fixed bindings."""
    params = {} if parameters is None else parameters
    if not isinstance(params, dict):
        raise ValueError("Semantic action parameters must be an object")
    declared = spec.get("parameters") or {}
    required = spec.get("required") or []
    unknown = set(params) - set(declared)
    if unknown:
        raise ValueError(f"Undeclared semantic parameters: {sorted(unknown)}")
    missing = set(required) - set(params)
    if missing:
        raise ValueError(f"Missing semantic parameters: {sorted(missing)}")
    for key, value in params.items():
        kind = declared[key].get("type") if isinstance(declared[key], dict) else None
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if kind in {"integer", "number"} and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{key} must be numeric")
        if key in binding:
            raise ValueError(f"Semantic parameter {key} overrides a fixed binding")
    return bind_semantic_action(binding, params)
