"""Stationary property rewards, cache isolation and applicability diagnostics."""
from __future__ import annotations

import math
import json
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from common.training import canonical_json_sha256
from stage3.inference import PredictionQuery
from .chemistry import canonical, charge


class ApplicabilityDomain:
    def __init__(self, references: dict, capabilities: dict):
        self.references, self.capabilities = references, capabilities
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fingerprints = {}
        self.records, self.ranges, self.charge_pairs = {}, {}, {}
        for task, rows in references.items():
            cap = capabilities[task]
            if not rows:
                raise ValueError(f"Empty applicability reference: {task}")
            self.records[task] = sorted({(canonical(row["cation"]), canonical(row["anion"]),
                canonical(row["solute"]) if cap.requires_solute else "") for row in rows})
            self.charge_pairs[task] = {(charge(c), charge(a)) for c, a, _ in self.records[task]}
            self.ranges[task] = {name: (min(float(row[name]) for row in rows),
                max(float(row[name]) for row in rows)) for name in cap.condition_columns}

    def fingerprint(self, smiles):
        if smiles not in self.fingerprints:
            self.fingerprints[smiles] = self.generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        return self.fingerprints[smiles]

    def distance(self, task: str, cation: str, anion: str, conditions: dict, solute: str = "") -> dict:
        records = self.records[task]
        similarities = []
        for index, smiles in enumerate((cation, anion, solute)):
            if smiles:
                similarities.append(DataStructs.BulkTanimotoSimilarity(self.fingerprint(smiles),
                    [self.fingerprint(record[index]) for record in records]))
        similarity = max(math.prod(values) ** (1 / len(values)) for values in zip(*similarities, strict=True))
        cap = self.capabilities[task]
        excursions = {name: max(lo - conditions[name], conditions[name] - hi, 0.0)
            / cap.condition_stats[name]["scale"] for name, (lo, hi) in self.ranges[task].items()}
        seen = (charge(cation), charge(anion)) in self.charge_pairs[task]
        distance = 1 - similarity + sum(excursions.values()) / max(1, len(excursions)) + float(not seen)
        return {"distance": distance, "pair_similarity": similarity, "condition_excursions": excursions,
            "charge_pair_seen": seen}


class RewardOracle:
    def __init__(self, predictor, domain: ApplicabilityDomain, *, beta: float = 4.0,
                 uncertainty_penalty: float = 1.0, domain_penalty: float = 1.0,
                 epsilon: float = 1e-8, cache_path: Path | None = None):
        if (not all(math.isfinite(v) for v in (beta, uncertainty_penalty, domain_penalty, epsilon))
                or beta <= 0 or uncertainty_penalty < 0 or domain_penalty < 0 or not 0 < epsilon < 1):
            raise ValueError("Invalid stationary reward parameters")
        self.predictor, self.domain = predictor, domain
        self.beta, self.uncertainty_penalty, self.domain_penalty, self.epsilon = (
            beta, uncertainty_penalty, domain_penalty, epsilon)
        self.cache, self.cache_path, self.new_queries = {}, cache_path, 0
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                item = json.loads(line)
                if (canonical_json_sha256(item["context"]) != item["key"]
                        or item["context"]["predictor"] != predictor.identity["hash"]
                        or item["context"]["execution"] != getattr(predictor, "execution_identity", predictor.identity)["hash"]
                        or canonical_json_sha256(item["prediction"]) != item["prediction_sha256"]):
                    raise ValueError("Prediction cache identity mismatch")
                self.cache[item["key"]] = item

    def missing_queries(self, cation, anion, condition):
        cation, anion = canonical(cation), canonical(anion)
        return sum(canonical_json_sha256({"predictor": self.predictor.identity["hash"],
            "execution": getattr(self.predictor, "execution_identity", self.predictor.identity)["hash"],
            "cation": cation, "anion": anion, "task": o.task, "conditions": condition.conditions[o.task],
            "solute": condition.solutes.get(o.task, "")}) not in self.cache for o in condition.objectives)

    def evaluate(self, cation: str, anion: str, condition) -> dict:
        cation, anion = canonical(cation), canonical(anion)
        predictions, reliability = {}, {}
        contexts = {}
        for objective in condition.objectives:
            task = objective.task
            context = {"predictor": self.predictor.identity["hash"], "cation": cation, "anion": anion,
                "execution": getattr(self.predictor, "execution_identity", self.predictor.identity)["hash"],
                "task": task, "conditions": condition.conditions[task], "solute": condition.solutes.get(task, "")}
            key = canonical_json_sha256(context)
            contexts[task] = (key, context)
        missing = {task: context for task, (key, context) in contexts.items() if key not in self.cache}
        if missing:
            batch = self.predictor.predict([PredictionQuery(cation, anion,
                {task: context["conditions"] for task, context in missing.items()},
                {task: context["solute"] for task, context in missing.items() if context["solute"]})])[0]
            for task, context in missing.items():
                key = contexts[task][0]
                prediction = batch[task]
                if len(prediction["folds_native"]) != 5 or not all(math.isfinite(v) for v in prediction["folds_native"]):
                    raise ValueError("Reward oracle requires five finite native predictions")
                item = {"key": key, "context": context, "prediction": prediction,
                    "prediction_sha256": canonical_json_sha256(prediction)}
                self.cache[key] = item
                self.new_queries += 1
                if self.cache_path:
                    with self.cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        for task, (key, context) in contexts.items():
            predictions[task] = self.cache[key]["prediction"]
            reliability[task] = self.domain.distance(task, cation, anion, context["conditions"], context["solute"])
        weight_sum = sum(o.weight for o in condition.objectives)
        satisfaction, uncertainty, distance, passed = 0.0, 0.0, 0.0, True
        deviations = {}
        for o in condition.objectives:
            cap, pred = self.predictor.capabilities[o.task], predictions[o.task]
            threshold = cap.mean if o.threshold is None else cap.to_native(o.threshold, o.unit)
            sign = 1 if o.direction == "maximize" else -1
            delta = sign * (pred["native"] - threshold) / cap.scale
            log_desirability = -max(-delta, 0) - math.log1p(math.exp(-abs(delta)))
            weight = o.weight / weight_sum
            satisfaction += weight * log_desirability
            uncertainty += weight * pred["std_native"] / cap.scale
            distance += weight * reliability[o.task]["distance"]
            passed = passed and (o.threshold is None or delta >= 0)
            deviations[o.task] = None if o.threshold is None else pred["value"] - o.threshold
        score = self.beta * satisfaction - self.uncertainty_penalty * uncertainty - self.domain_penalty * distance
        log_floor = math.log(self.epsilon)
        log_reward = max(log_floor, score) + math.log1p(math.exp(-abs(log_floor - score)))
        return {"log_reward": log_reward, "predictions": predictions, "reliability": reliability,
            "uncertainty": uncertainty, "domain_distance": distance, "passed": passed, "deviations": deviations}
