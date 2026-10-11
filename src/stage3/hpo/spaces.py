from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from stage2.home_config import home_recipe_from_dict
from stage3.config import stage3_config_from_dict


def aligned_ratio(value):
    return max(64, round(1024 * value / 64) * 64) / 1024


def compile_joint(base2, base3, parameters, *, root, split_seed=42, training_seed=42):
    """Expand shared multipliers into complete, validated Stage-specific recipes."""
    two, three = deepcopy(base2), deepcopy(base3)
    p = parameters
    for raw in (two, three):
        raw["data"]["seed"] = split_seed
        raw["training"]["seed"] = training_seed
        raw["training"]["protocol"] = "hpo_v4"
    two["data"].update(data_root=f"data/{split_seed}", task_catalog_path=f"data/{split_seed}/task_catalog.csv",
                       artifacts_dir=f"{root}/cache/split{split_seed}/stage2/artifacts")
    two["preparation"]["data_metadata_path"] = f"{root}/audit/data/split{split_seed}/stage2/metadata.json"
    three["data"].update(stage3_dir=f"data/{split_seed}/stage3", task_catalog=f"data/{split_seed}/task_catalog.csv",
                         artifacts_dir=f"{root}/cache/split{split_seed}/stage3/artifacts")
    three["preparation"].update(cache_dir=f"{root}/cache/split{split_seed}/stage3/entity_cache",
                               data_metadata_path=f"{root}/audit/data/split{split_seed}/stage3/metadata.json")
    three["initialization"]["simulation_artifacts_dir"] = two["data"]["artifacts_dir"]
    model = three["model"]
    model["global_experts"] = int(p.get("global_count", model["global_experts"]))
    model["private_experts"] = int(p.get("private_count", model["private_experts"]))
    model["expert_hidden_ratio"] = aligned_ratio(model["expert_hidden_ratio"] * p.get("global_width", 1))
    model["private_hidden_ratio"] = aligned_ratio(base3["model"]["expert_hidden_ratio"] * p.get("private_width", 1))
    model["global_dropout"] = p.get("global_dropout", model["dropout"])
    model["dropout"] = p.get("private_dropout", model["dropout"])
    for name in ("film_hidden_ratio", "tower_hidden_ratio"):
        model[name] = aligned_ratio(model[name] * p.get("private_width", 1))
    for group_id, group in three["groups"].items():
        group["experts"] = max(1, round(group["experts"] * p.get("group_count", 1)))
        group["expert_hidden_ratio"] = aligned_ratio(group["expert_hidden_ratio"] * p.get("group_width", 1) * p.get(f"group_width:{group_id}", 1))
        group["dropout"] = p.get("group_dropout", base3["model"]["dropout"])
        for phase in ("phase1", "phase2"):
            group[phase]["lr"] *= p.get("s3_group_lr", 1) * p.get(f"group_lr:{group_id}", 1)
            group[phase]["epochs"] = max(1, round(group[phase]["epochs"] * p.get("s3_group_epochs", 1)))
    phases = three["training"]["three_phase"]
    phases["global"]["lr"] = p.get("s3_global_lr", phases["global"]["lr"])
    phases["global"]["epochs"] = int(p.get("s3_global_epochs", phases["global"]["epochs"]))
    for class_id, recipe in phases["private_classes"].items():
        scale = p.get("private_width", 1) * p.get(f"private_width:{class_id}", 1)
        recipe["width_ratio"] = aligned_ratio(recipe["width_ratio"] * scale)
        recipe["phase1"]["lr"] *= p.get("s3_private_lr", 1)
        recipe["phase1"]["epochs"] = max(1, round(recipe["phase1"]["epochs"] * p.get("s3_private_epochs", 1)))
        for field in ("phase2_epochs", "phase3_epochs"):
            recipe[field] = round(recipe[field] * p.get("s3_private_epochs", 1))
    for task in three["tasks"].values():
        for field in ("phase1_private_epochs", "phase2_private_epochs", "phase3_private_epochs"):
            if field in task:
                task[field] = round(task[field] * p.get("s3_private_epochs", 1))
        if "phase1_private_lr" in task:
            task["phase1_private_lr"] *= p.get("s3_private_lr", 1)
        override = task.setdefault("model_overrides", {})
        for field in ("private_hidden_ratio", "tower_hidden_ratio", "film_hidden_ratio"):
            if field in override:
                override[field] = aligned_ratio(override[field] * p.get("private_width", 1)
                                                * p.get(f"private_width:{task['size_class']}", 1))
        override["private_dropout"] = p.get("private_dropout", override.get("private_dropout", base3["model"]["dropout"]))
        if "private_experts" in override:
            override["private_experts"] = model["private_experts"]
    three["training"]["owner_weight_decay"] = {scope: p.get(f"s3_{scope.lower()}_wd", three["training"]["weight_decay"])
                                                   for scope in ("GLOBAL", "GROUP", "PRIVATE")}
    # Stage2's task-local heads have a different base capacity from experimental heads.
    two["home"]["model"] = deepcopy(model)
    two["home"]["groups"] = {name: deepcopy(three["groups"][name]) for name in two["home"]["groups"]}
    for name, group in two["home"]["groups"].items():
        for phase in ("phase1", "phase2"):
            group[phase] = deepcopy(base2["home"]["groups"][name][phase])
    two["training"]["owner_training"] = {scope: {
        "lr": p.get(f"s2_{scope.lower()}_lr", base2["training"]["task_head_learning_rate"]),
        "epochs": int(p.get(f"s2_{scope.lower()}_epochs", base2["training"]["epochs"])),
        "weight_decay": p.get(f"s2_{scope.lower()}_wd", base2["training"]["weight_decay"])}
        for scope in ("GLOBAL", "GROUP", "PRIVATE")}
    two["training"]["epochs"] = max(value["epochs"] for value in two["training"]["owner_training"].values())
    home_recipe_from_dict(two)
    config = stage3_config_from_dict(three)
    for task, recipe in config.tasks.items():
        group_lr = config.groups[recipe.meta_group].phase1.lr
        if not config.training.three_phase.global_scope.lr > group_lr > config.resolved_private_recipe(task).phase1_lr:
            raise ValueError("HPO Phase1 requires GLOBAL LR > GROUP LR > PRIVATE LR")
    return two, three


def joint_controls(base2, base3, space, *, root, limit):
    controls = [("base", {})]
    for name, spec in space.items():
        for suffix, value in (("low", spec["low"]), ("high", spec["high"])):
            params = {name: value}
            # Seek the closest valid one-factor contrast without modifying other factors.
            for _ in range(16):
                try:
                    compile_joint(base2, base3, params, root=root)
                    break
                except ValueError:
                    baseline = spec.get("base", 1)
                    params[name] = (params[name] + baseline) / 2
                    if spec.get("kind") == "int":
                        params[name] = round(params[name])
            else:
                continue
            if params[name] != spec.get("base", 1):
                controls.append((f"{name}_{suffix}", params))
    interactions = ({"global_width": 2, "s3_global_lr": 0.0005},
                    {"s2_global_lr": 0.0003, "s2_global_epochs": 20},
                    {"s3_global_lr": 0.0005, "s3_global_epochs": 30})
    for index, params in enumerate(interactions):
        compile_joint(base2, base3, params, root=root)
        controls.append((f"interaction{index}", params))
    return controls[:limit]


def posttraining_controls(base, limit=18):
    controls = [("base", {}), ("linear", {"type": "linear"})]
    for kind in ("mlp", "residual_mlp"):
        for width, depth in ((1, 2), (0.5, 2), (2, 2), (1, 1), (1, 3)):
            controls.append((f"{kind}_w{width}_d{depth}", {"type": kind, "width": width, "depth": depth}))
    controls += [("lr_low", {"lr": 1e-5}), ("lr_high", {"lr": 1e-3}),
                 ("epoch_low", {"epochs": 5}), ("epoch_high", {"epochs": 50}),
                 ("dropout_low", {"dropout": 0}), ("dropout_high", {"dropout": 0.3})]
    return controls[:limit]


def compile_posttraining(base, params, target, cache):
    result = deepcopy(base)
    result["protocol"] = "hpo_v4"
    result["representations_cache"] = str(cache)
    result["predictor_overrides"] = {}
    original = base.get("predictor_overrides", {}).get(target, base["predictor"])
    kind = params.get("type", original["type"])
    if kind == "linear":
        result["predictor"] = {"type": "linear"}
    else:
        dims = original.get("hidden_dims", [256, 128] if target == "partial_atomic_charge" else [512, 256])
        first = dims[0]
        depth = int(params.get("depth", len(dims)))
        sizes = ([first] if depth == 1 else [first] * (depth - 1) + [max(64, first // 2)])
        result["predictor"] = {"type": kind, "activation": "gelu", "dropout": params.get("dropout", original.get("dropout", 0)),
                               "hidden_dims": [max(64, round(value * params.get("width", 1) / 64) * 64) for value in sizes]}
    for name, field in (("lr", "learning_rate"), ("epochs", "epochs"), ("seed", "seed")):
        if name in params:
            result[field] = params[name]
    return result


def write_yaml(path, data):
    path = Path(path)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Immutable HPO config changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)
