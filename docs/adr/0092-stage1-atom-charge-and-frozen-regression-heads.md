# ADR-0092: Stage1 atom-charge supervision and independent frozen regression heads

## Status

Accepted (2026-10-07). Extends v4 pretraining with partial atomic charge, revises the active ADR-0091 audit interval to1,000 updates and adds an opt-in post-training command. All encoder architecture, input/output, natural shuffle and downstream freeze decisions remain unchanged.

## Pretraining and audit

Active Base adds `lambda_partial_charge=0.1`, independent of electronic0.1; reconstruction1/1/1, alignment0.1, RDKit0.5 and Uni-Mol0.25 remain unchanged. A disposable `Linear512→1` predicts normalized charges from current graph atom states, using the same masked forward. SmoothL1 first averages atoms within each molecule, then uses role2/2/1 weighted valid-molecule means. Only the graph encoder receives this objective's representation gradient. Missing molecules return differentiable zero. No descriptor/role input, extra forward or sampler change.

The shared coefficient default is0, omitted on serialization; disabled supervision creates no atom head or extra initialization RNG calls. Historical config/state/numerical identities remain unchanged. Enabled supervision is v4-only and adds513 auxiliary parameters, not encoder capacity. Existing v4 kind/format4 remains; loss and label-sidecar identity reject incompatible resumes. Full epoch checkpoints retain all heads; encoder-only export still contains only SMILES encoder, graph encoder and Fusion.

Base audit interval is1,000; `--gradient-audit-interval-steps` optionally overrides YAML, including0 to disable. Audit settings remain execution-only. Enabled atom supervision adds `partial_charge_grad_norm` and corresponding coefficient, weighted norm and coverage. The previous seven fields, fixed32 validation probe, mask, mode/RNG restoration, rank0-only derivatives and synchronized errors remain unchanged. No valid atom targets produces null; valid zero derivatives produce0.

## Train-only atom-label sidecar

Only partial-charge train rows and their referenced verified MOL2 structures supply pretraining labels. Use the existing typed-isomorphism/connectivity fallback, hydrogen policy and deterministic mapping contract unchanged, now shared by Stage1/2 through `common.atom_targets`; preserve the old Stage2 public imports and mapping identity namespace.

Join exact canonical structures already in the corpus, not seed/augmentation ancestry. Retain every source CSV row, including conflicting canonical duplicates, as an independent charge observation with its source ID; never average labels or select a first/last row. Fit charge mean/population std on all observations of matched Stage1 training structures (zero variance→1); retain valid-split matched labels only for reporting. A corpus molecule is encoded once; each observation separately averages its atom loss and receives its role weight in the charge objective. Other objectives and corpus sampling are unchanged. Sidecar format2 and the explicit observation policy enter identity, rejecting old sidecars/checkpoints. If no matched training atoms exist, all charge supervision is masked and coverage reports it.

Independent partial-charge head training likewise retains source rows as separate samples, reusing the same frozen atom representation. The conflicting-label rejection below now applies only to scalar electronic targets; canonical train/valid overlap still fails for all targets. This revision does not resolve mapping ambiguity or add conformer inputs.

A separate sidecar binds corpus/manifest, train CSV, structure manifest and referenced MOL2 SHA/size, mapping, statistics and target state hash. It stores normalized targets and mapping audit; corrupt, incomplete or changed source/corpus fails. It never rewrites corpus/statistics/teacher cache. Normal CLI prepare also prepares a separate managed sidecar run; `--partial-charge-only` prepares it from existing corpus without rebuilding five-million-molecule features. The active cache path points to that run's `artifacts/` directory. Training requires a valid sidecar when enabled. Dataset/packer concatenate ragged labels and masks using actual atom counts.

## Independent frozen-head training

[ADR-0093](0093-stage1-configurable-frozen-predictors.md) extends this section with configurable MLP/residual MLP predictors and format2 artifacts; the Linear initialization and all data/freeze/optimization boundaries below remain valid.

`scripts/stage1/regression.py` is explicit opt-in, not automatically called after pretraining. It accepts a complete final v4 pretraining checkpoint, self-contained regression YAML, new output, optional targets and execution-only device. It does not accept encoder-only exports or incomplete epochs. Default targets are HOMO/LUMO plus eleven HF scalar targets and partial atomic charge; `q_max/min/std/pos_frac` are scalar summaries, not atom charge.

Freeze/eval the encoder and Fusion. Encode unmasked full simulation train/valid structures once, without modality dropout or descriptor input; cache learned entity/atom states with source and tensor hashes. No teacher inference is needed. Each target independently starts from the original electronic-head row or atom head and has its own linear-head optimizer. Old v4 checkpoints may select existing scalar tasks; missing atom head is an error, never randomly substituted.

Fit each task's post-training normalization using full train only. Convert initialized head weights/bias by the exact affine change of target coordinates, preserving original-unit predictions within floating-point tolerance. Defaults are ten raw epochs, batch128, AdamW1e-4, betas0.9/0.999, eps1e-8, WD0.01, clipping1, BF16 and constant LR. Role2/2/1 molecule means remain; atom loss/metrics average within molecules. Only train updates heads; source valid reports MAE/RMSE at initialization and each epoch; test is never read. Canonical train/valid overlap or conflicting scalar electronic labels fails; partial-charge observations are retained separately. Missing scalar labels are skipped, empty train fails and empty valid metrics are null. Always publish the fixed last epoch, irrespective of validation. Failed head runs restart in a new output; no head mid-job resume or overwrite.

Outputs contain frozen representation bank, per-task metrics and `regression_head.pt/json`, and final summary. Independent kind/identity binds base checkpoint SHA, base training identity, encoder hash, source train/valid hashes, feature identity, normalization, task/budget/seed and final head hash. Loading checks source SHA, self-hash and artifact/state/manifest agreement. Original checkpoints, encoder-only artifact, other heads and downstream Stage2/3 are never updated; regression artifacts are not Stage1 encoder replacements or official Stage3 reporting models.

## Verification and artifact boundary

Tests cover optional audit CLI, disabled historical schema/state/initialization, molecule/role weighting, missing labels, atom gradients, CPU/CUDA BF16/DDP audit isolation, source/mapping/hash rejection, unchanged corpus/teacher identities, final-epoch recovery and encoder-only export. Post-training tests verify affine initialization, frozen/non-target states, train/valid/test boundaries, constant LR, source binding and that deliberately worsened validation cannot change final head state.

New charge pretraining needs fresh Stage1 and downstream Stage2/3 training outputs; existing corpus and teacher cache remain reusable after preparing the sidecar. Independent frozen-head training alone does not require rerunning downstream stages. No formal prepare, teacher generation, training or post-training runs occur during implementation. Local missing electronic/charge sources remain an explicit data issue, not a fallback or relaxed validation.
