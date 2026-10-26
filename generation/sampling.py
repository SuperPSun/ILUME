"""Raw sampling, threshold filtering, Pareto layers and structural diversity."""
from __future__ import annotations

import json
import time

from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chemistry import State
from .training import rollout


def pareto_layers(records, condition):
    remaining = list(records)
    while remaining:
        vectors = [[(1 if o.direction == "maximize" else -1) * row["predictions"][o.task]["native"]
            for o in condition.objectives] for row in remaining]
        indices = [i for i, values in enumerate(vectors) if not any(
            all(a >= b for a, b in zip(other, values, strict=True)) and any(a > b for a, b in zip(other, values, strict=True))
            for j, other in enumerate(vectors) if i != j)]
        yield [remaining[i] for i in indices]
        selected = set(indices)
        remaining = [row for i, row in enumerate(remaining) if i not in selected]


def pair_distance(left, right, domain):
    cation = DataStructs.TanimotoSimilarity(domain.fingerprint(left["cation"]), domain.fingerprint(right["cation"]))
    anion = DataStructs.TanimotoSimilarity(domain.fingerprint(left["anion"]), domain.fingerprint(right["anion"]))
    return 1 - (cation * anion) ** .5


def select_candidates(records, condition, domain, count):
    unique = {(row["cation"], row["anion"]): row for row in records if row["passed"]}
    result = []
    for rank, layer in enumerate(pareto_layers(list(unique.values()), condition)):
        while layer and len(result) < count:
            selected = max(layer, key=lambda row: (
                min(pair_distance(row, prior, domain) for prior in result) if result else row["log_reward"],
                row["log_reward"], row["cation"], row["anion"]))
            result.append({**selected, "pareto_rank": rank})
            layer.remove(selected)
        if len(result) == count:
            break
    return result


def hypervolume(points):
    """Exact dominated volume above the fixed zero reference for up to three objectives."""
    points = [p for p in points if all(v > 0 for v in p)]
    if not points:
        return 0.0
    if len(points[0]) == 1:
        return max(p[0] for p in points)
    total, previous = 0.0, 0.0
    for bound in sorted({p[0] for p in points}):
        total += (bound - previous) * hypervolume([p[1:] for p in points if p[0] >= bound])
        previous = bound
    return total


def diversity_metrics(records, domain):
    unique = list({(r["cation"], r["anion"]): r for r in records}.values())
    scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(smiles=r["cation"]) or "acyclic" for r in unique}
    distances = [pair_distance(a, b, domain) for i, a in enumerate(unique) for b in unique[:i]]
    return {"count": len(records), "unique_pairs": len(unique), "cation_scaffolds": len(scaffolds),
        "anion_families": len({r["anion_family"] for r in unique}),
        "mean_pair_distance": sum(distances) / len(distances) if distances else None}


def sample_candidates(model, environment, oracle, condition, rng, *, attempts, count, raw_path,
                      random_policy=False, independent_pairing=False, max_property_queries=1000000):
    if attempts < 1 or count < 1 or max_property_queries < 1:
        raise ValueError("Sampling requires positive proposal and candidate budgets")
    records, failed, used, budget_exhausted = [], 0, 0, False
    started, initial_queries = time.monotonic(), oracle.new_queries
    model.eval()
    with raw_path.open("x", encoding="utf-8") as handle:
        for index in range(attempts):
            used += 1
            states = rollout(model, environment, condition, rng, random_policy=random_policy)
            if states is None:
                failed += 1
                handle.write(json.dumps({"attempt": index, "status": "no_legal_completion"}) + "\n")
                continue
            candidate = environment.describe(states[-1])
            original_anion = candidate["anion_id"]
            if independent_pairing:
                import torch
                with torch.no_grad():
                    children = environment.children(State(), condition.constraints, condition.fixed_anion)
                    probabilities = ([1.0] * len(children) if random_policy else
                        model.log_probabilities(State(), children).exp().cpu().tolist())
                    independent_anion = rng.choices(children, weights=probabilities, k=1)[0].anion
                candidate = environment.describe(State(independent_anion, states[-1].cation, True))
            needed = oracle.missing_queries(candidate["cation"], candidate["anion"], condition)
            if oracle.new_queries - initial_queries + needed > max_property_queries:
                budget_exhausted = True
                handle.write(json.dumps({"attempt": index, "status": "property_query_budget_exhausted"}) + "\n")
                break
            score = oracle.evaluate(candidate["cation"], candidate["anion"], condition)
            row = {**candidate, **score, "attempt": index,
                "trajectory": environment.trajectory_record(states), "request": condition.to_dict(),
                "predictor_identity": oracle.predictor.identity["hash"], "resource_identity": environment.library.identity,
                "pairing_mode": "independent" if independent_pairing else "conditioned",
                "generation_anion_id": original_anion}
            records.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    selected = select_candidates(records, condition, oracle.domain, count)
    capabilities = oracle.predictor.capabilities
    points = [[2 + (1 if o.direction == "maximize" else -1) *
        (row["predictions"][o.task]["native"] - capabilities[o.task].mean) / capabilities[o.task].scale
        for o in condition.objectives] for row in records]
    return selected, {"attempts": used, "proposal_budget": attempts, "failed_trajectories": failed,
        "property_query_budget": max_property_queries, "query_budget_exhausted": budget_exhausted,
        "threshold_pass_rate": sum(r["passed"] for r in records) / used,
        "requested": count, "returned": len(selected), "shortfall": max(0, count - len(selected)),
        "raw": diversity_metrics(records, oracle.domain), "selected": diversity_metrics(selected, oracle.domain),
        "new_property_queries": oracle.new_queries - initial_queries, "elapsed_seconds": time.monotonic() - started,
        "hypervolume": hypervolume(points) if len(condition.objectives) <= 3 else None,
        "hypervolume_reference": "signed_native_training_zscore_minus_two_v1",
        "hypervolume_status": "computed" if len(condition.objectives) <= 3 else "not_computed_above_three_objectives"}
