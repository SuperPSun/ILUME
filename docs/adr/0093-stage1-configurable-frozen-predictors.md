# ADR-0093: Configurable predictors on frozen Stage1 representations

## Status

Accepted (2026-10-07). Extends only the independent regression command in [ADR-0092](0092-stage1-atom-charge-and-frozen-regression-heads.md). Stage1 pretraining and Stage2/3 remain unchanged.

## Context

Independent regressors previously always reconstructed a scalar Linear. Users need to specify nonlinear capacity without changing the frozen representation, data contract, optimization budget or final-epoch selection.

## Decision

- Regression YAML supports a shared `predictor` and complete per-target `predictor_overrides`. Default is `linear`; an override replaces, rather than merges, the shared recipe. Unknown fields, targets, types, invalid widths/dropout or nonlinear heads without hidden layers fail.
- Supported types are `linear`, `mlp` and `residual_mlp`. Input width is derived from the frozen source (1024 entity / 512 atom at Base), output is one scalar. No configurable input/output width or dynamic Python imports.
- MLP hidden stages are Linear, activation, Dropout; the output Linear has no activation. Each residual hidden stage uses `shortcut(x) + Dropout(Linear(Dropout(activation(Linear(x)))))`, with input→width→width projections and an Identity shortcut if widths agree, otherwise a Linear. No normalization or post-add activation. Activations are GELU/ReLU/SiLU; default GELU and dropout0.
- Linear retains pretrained-row/atom-head initialization and the equivalent normalization-coordinate conversion. MLP and residual MLP are random predictors, not pretrained Linear plus calibration residuals; initial predictions need not match pretraining. Source eligibility and missing-atom-head/scaler checks remain unchanged.
- Nonlinear initialization and training dropout each use a separate stream seeded by `seed + REGRESSION_TASKS.index(task)`. CPU and the active CUDA device RNG are restored afterwards. Independent task execution order cannot change head parameters. Linear's existing RNG and numerical path remain unchanged.
- One resolver and builder serve initialization, training and loading. Encoder/Fusion stay eval/frozen, with clean cached representations; only the current predictor enters its optimizer. Preserve role/molecule normalization, raw epochs, AdamW, constant LR, clipping, BF16 and validation reporting-only. Test is never read.
- New head artifacts use format2 with resolved predictor, input dimension, parameter count, initialization mode, seed, source and tensor hashes. Rebuild from the artifact alone, verify source width, identity/structure/manifest/state and load strictly. Existing format1 Linear artifacts remain readable.
- Default predictor fields are omitted from serialization and scientific identity. Nondefault jobs bind the resolved recipes for the selected targets; unselected overrides do not change that job's identity. Stage1 encoder/training and downstream identities are untouched.

## Current regression YAML recipe

`configs/v4/stage1/regression_heads.yaml` explicitly uses independent MLPs: each of the thirteen electronic targets has entity1024→512→256→1 (656,385 parameters); partial atomic charge has atom512→256→128→1 (164,353 parameters). Both use GELU, dropout0 and existing task-local random initialization. This is a configuration choice, not a shared schema default change: omitted predictor still resolves to Linear. Encoder/Fusion, source/data/normalization, AdamW/LR, ten epochs and fixed final selection remain unchanged. Use a fresh regression output; no Stage1/2/3 retraining is required.

## Consequences

Users can compare independently configured scalar and atom predictors on the same representation. Nonlinear heads add trainable parameters/dropout and cannot inherit the original Linear's raw predictions. Structure changes require a fresh regression output, not new Stage1/2/3 training. No artifact overwrite, HPO, scheduler changes, early stopping or new CLI flags.

## Alternatives

Rejected automatic weight migration into nonlinear heads and zero-initialized calibration residuals: those would change the scientific initialization experiment. Rejected arbitrary module imports and unrestricted network graphs: the three explicit builders cover the requested scope.

## Verification

Cover configuration/default/override errors, Linear affine and numerical compatibility, nonlinear seed/dropout isolation, residual shortcuts, scalar/atom widths and gradients, reversed branch execution, format1 compatibility, format2 prediction round-trip and corrupted source/structure rejection. Encoder/source artifacts remain bitwise unchanged; reporting cannot select the final head. Implementation uses temporary fixtures only; missing local catalog/data is reported without relaxing contracts.
