# ADR-0091: Stage1-v4 loss weights and read-only gradient audit

## Status

Accepted (2026-10-07). Revises only the RDKit/Uni-Mol coefficients in ADR-0089 and adds execution-only diagnostics. ADR-0090 capacity and all other numerical contracts remain unchanged; no new capacity profile.

## Decision

The active v4 Base coefficients are SMILES/atom/bond=`1/1/1`, alignment=`0.1`, RDKit=`0.5`, Uni-Mol=`0.25`, electronic=`0.1`. Shared schema defaults and historical YAML remain unchanged. Loss normalization remains valid-element means within molecules, then role2/2/1 weighted molecule means. No sampler, encoder, modality dropout, optimizer, scheduler or fixed-final-epoch change.

`training.gradient_audit_interval_steps` defaults to0 (off); `gradient_audit_batch_size` defaults to32. Default values are omitted on serialization. Enabled audits require dual-view v4. Active Base explicitly enables every5,000 completed optimizer updates, using32 validation molecules. Both fields are execution-only: recorded in run/checkpoint configuration, excluded from scientific config hash, training identity and encoder identity. Audit cadence or probe size may change on epoch-boundary resume; loss changes may not.

## Probe and gradient contract

- Select `min(32, validation_size)` molecules without replacement using an independent local RNG seeded by `data.seed+400000`; no role/property/HF stratification. Reuse the existing packer, cached auxiliary targets and evaluation masker with fixed seed `data.seed+400001`. Sample order and masks remain fixed across steps, epochs, attempts and world sizes. Record IDs, corpus/feature identity, mask hash and probe hash.
- Run after optimizer/scheduler update and gradient clearing, only at positive multiples of the interval; no additional epoch-final audit. Use eval mode and evaluation masking (ordinary and fusion modality dropout off), with the training AMP setting and FP32 squared-norm accumulation.
- A single eager forward supplies all seven objectives. For each, use `autograd.grad` only over distinct SMILES/Graph/Fusion parameters; auxiliary heads are excluded, but their chain-rule contribution to encoder gradients remains. Do not call backward, populate `.grad`, clip or update optimizer/scheduler. Release each gradient set before the next objective.
- Raw norm is `||∇encoder L_i||₂`: role normalization is retained; only the outer coefficient is excluded. Weighted norm is `abs(lambda_i) * raw_norm`, computed without another backward. These are objective-specific magnitudes, not the norm of their vector sum or a measurement of alignment/conflict between objectives.
- DDP rank0 evaluates the complete probe without DDP forward or training loss collectives; other ranks wait via synchronized error/status broadcast. Preserve all module mode flags and Python/NumPy/Torch CPU/current-rank CUDA RNG; do not initialize or inspect other CUDA devices. Normal errors propagate to all ranks; no OOM resize or silent skip.

## Output and recovery

Append `gradient_audit.jsonl`, separate from unchanged training metrics. Each row contains epoch, global step, attempt ID, training identity, precision, probe metadata, current coefficients, and:

```text
smiles_grad_norm
atom_grad_norm
bond_grad_norm
alignment_grad_norm
rdkit_grad_norm
unimol_grad_norm
electronic_grad_norm
weighted_grad_norms.{smiles,atom,bond,alignment,rdkit,unimol,electronic}
coverage.<objective>.{valid_molecules,valid_role_weight_sum,status}
```

Missing supervision produces `null` with `no_valid_targets`, not a misleading zero. A valid objective with a zero derivative records0. Non-finite norms are an explicit audit error; they never silently change weights or budgets. In particular, a small natural probe may have no electronic labels and cannot then diagnose electronic gradient strength.

Recovery retains complete-epoch/attempt behavior. Failed-attempt audit rows are not truncated or used to choose a checkpoint; replaying an incomplete epoch can append another observation of the same step. Audit settings do not require a checkpoint format upgrade. Validation remains reporting-only: its derivatives are never applied to parameters or used for automatic decisions.

## Artifact boundary and verification

Corpus, train-only statistics and the completed teacher cache remain reusable, including all existing failure masks. New loss weights change training identity, so new Base trains Stage1 from scratch in a fresh output directory, then regenerates/trains Stage2 and prepares/trains/evaluates Stage3. Historical outputs are read-only; no formal runs are launched during implementation.

Tests compare objective gradients with independent references, weighted norms and missing targets, audit-on/off model/gradient/optimizer/scheduler/RNG equality, fixed probe/masks, rank0-only DDP/error propagation, epoch resume and identity separation. Gradient audit is human-facing evidence only, not a new training objective or model-selection mechanism.
