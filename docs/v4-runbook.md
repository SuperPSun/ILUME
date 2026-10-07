# v4 execution

Run from repository root, using fresh `outputs/v4/` paths. Do not overwrite historical v3 outputs. Science contract: [ADR-0089](adr/0089-v4-frozen-dual-view-stage1.md). CUDA examples use four devices.

Current Base uses twelve SMILES Transformer layers and eight independent residual graph blocks: encoder-only approximately60.73M parameters for a2,048-token vocabulary ([ADR-0090](adr/0090-stage1-v4-residual-encoder-capacity.md)). Compared with the earlier approximately30M v4 Base, **existing Stage1 prepared corpus, statistics and completed Uni-Mol cache are reusable**; skip their generation steps below if already complete. Train the new Stage1 from scratch, then regenerate Stage2 representations/train Stage2 and prepare/train/evaluate Stage3. Do not resume the old Stage1 checkpoint into the new architecture. Output defaults have not changed: if a training/downstream directory is occupied, select fresh paths and update downstream references consistently, or explicitly arrange archival before running; these commands do not authorize replacing historical results.

## Stage1 prepare and offline teacher

```bash
python scripts/stage1/prepare.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/prepare
```

Teacher generation must run in a **separate environment**. The commands below target Linux x86-64 with a CUDA 12.8-compatible driver and use Python 3.11 to keep the teacher package's older pandas/NumPy constraints isolated from the normal training environment. For another CUDA runtime, install the matching PyTorch wheel from the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/) instead of the `cu128` wheel. Uni-Mol's installation notes also require NumPy below 2 for its RDKit stack ([official installation guide](https://github.com/deepmodeling/Uni-Mol/blob/main/docs/source/installation.md)).

Create the environment and install the project plus the pinned offline teacher package. Installing `unimol-tools` with `--no-deps` prevents its resolver from replacing the selected PyTorch, NumPy or RDKit versions; the required runtime dependencies are installed explicitly:

```bash
conda create -n ilume-unimol2 python=3.11 pip -y
conda activate ilume-unimol2
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==1.5.3 rdkit==2025.9.5
python -m pip install -e .
python -m pip install scipy joblib addict scikit-learn numba
python -m pip install --no-deps unimol-tools==0.1.3.post1
```

Download the official [Uni-Mol2 84M checkpoint](https://huggingface.co/dptech/Uni-Mol2/blob/main/modelzoo/84M/checkpoint.pt) from the repository root. The published file is about 337 MB; its SHA-256 is checked below. The project deliberately disables automatic model downloads, so the file must exist at the exact configured path:

```bash
mkdir -p assets/unimol2/modelzoo/84M
curl --fail --location --retry 3 --continue-at - \
  https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt \
  --output assets/unimol2/modelzoo/84M/checkpoint.pt
echo '5b9241630f1cf0b173fb06d1e76096e5daf5f91c3e51b066ba69524dafe60e35  assets/unimol2/modelzoo/84M/checkpoint.pt' | sha256sum --check -
```

If the checksum fails, do not run cache generation; obtain a verified copy before continuing. Then verify the environment and local file:

```bash
python - <<'PY'
from pathlib import Path
import importlib.metadata
import json
import numpy
import torch
from rdkit import rdBase
from unimol_tools.models import unimolv2
from stage1.identity import validate_feature_generation_runtime

checkpoint = Path("assets/unimol2/modelzoo/84M/checkpoint.pt")
assert checkpoint.is_file(), f"Missing local checkpoint: {checkpoint}"
assert importlib.metadata.version("unimol_tools") == "0.1.3.post1"
assert torch.__version__.split("+", 1)[0] == "2.9.0", torch.__version__
assert torch.cuda.is_available(), "CUDA is unavailable in the teacher environment"
assert numpy.__version__ == "1.26.4", numpy.__version__
assert str(unimolv2.MODEL_CONFIG_V2["weight"]["84m"]) == "modelzoo/84M/checkpoint.pt"
metadata = json.loads(Path("outputs/v4/stage1/base/prepare/artifacts/metadata.json").read_text())
validate_feature_generation_runtime(metadata)
print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}; rdkit={rdBase.rdkitVersion}")
print(f"unimol_tools={importlib.metadata.version('unimol_tools')}; checkpoint={checkpoint}")
PY
```

The validator checks the prepared corpus's `feature_generation_contract`, including RDKit runtime and tokenizer versions. Match `rdBase.rdkitVersion`, not only `pip show rdkit`: package metadata and the imported runtime can differ. If the runtime differs from the prepared contract, fix the isolated environment before continuing; do not edit artifact metadata or rebuild the prepared corpus merely to bypass the check. The installation example targets runtime `2025.09.5`; if your corpus records another version, install that exact runtime instead.

First generate an independent small audit and inspect `manifest.json` (attempted/valid/failed, throughput and embedding bytes); this audit is not a training cache:

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 32 --limit 100 --output outputs/v4/stage1/base/teacher_audit
```

After approving success rate, throughput and storage, generate the full cache. Rerunning the identical command resumes immutable shards, not arbitrary partial rows. OOM is a failed run, not a masked molecule. Batch size/device and `--workers` are execution settings; canonical structure order and per-molecule conformer seed are fixed.

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 32
```

If GPU utilization is low, use `--workers` to prepare conformers and Uni-Mol features in parallel CPU processes. A bounded ordered queue prefetches inputs while the parent performs GPU inference; no worker loads the teacher model or uses CUDA. Features are computed once rather than repeated before inference. Default `--workers 1` retains serial execution. Worker processes consume additional RAM; do not exceed the job's CPU allocation.

For a Slurm job, start with `#SBATCH --cpus-per-task=8`, one GPU, and the following commands inside the allocated job. Keep the existing isolated teacher environment. First inspect this **new** parallel audit before running the full command:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 64 --workers 8 --limit 100 --output outputs/v4/stage1/base/teacher_audit_parallel
python -u scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 64 --workers 8
```

Stop the old job before restarting against the same cache root; concurrent writers are not supported. Completed shards are reused even if worker count changes. Redirected/Slurm output emits flushed `teacher_progress` JSON at startup, each committed shard and approximately every 30 seconds between inference batches. `processed/total` includes in-memory work; `committed` counts safely published molecules. A single slow conformer/batch can delay the next log. Monitor with `tail -f slurm-<job-id>.out`.

After the full cache completes, return to the `ilume` training environment. Uni-Mol is required only for cache generation; Stage1 training reads the cache. Stage1 refuses partial/unbound teacher caches; losses weight roles2/2/1 while the loader remains the original natural shuffle.

```bash
conda deactivate
conda activate ilume
python scripts/stage1/train.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/train
```

Output includes resume checkpoints with auxiliary heads and encoder-only `stage1_encoder.pt`. Deployment/Stage2 do not load teacher or auxiliary heads.

### Stage1 loss and gradient audit

Current Base uses reconstruction1/1/1, alignment0.1, RDKit0.5, Uni-Mol0.25, electronic0.1 and independent partial-charge0.1 ([ADR-0091](adr/0091-stage1-v4-loss-weights-gradient-audit.md), [ADR-0092](adr/0092-stage1-atom-charge-and-frozen-regression-heads.md)). These loss changes require a new Stage1 run, not resume from earlier coefficients; existing corpus/statistics/teacher cache remain reusable. Downstream Stage2/3 must use the newly trained encoder and fresh outputs.

Training automatically appends `gradient_audit.jsonl` every1,000 completed optimizer updates. Rank0 uses a fixed32-molecule validation probe and evaluation masks with dropout off; other ranks wait. It reports the original seven raw encoder norms plus `partial_charge_grad_norm` when enabled, `weighted_grad_norms` (absolute loss coefficient times norm), effective target coverage, probe IDs/hash, step and attempt. The existing metrics and checkpoint selection remain unchanged.

Set `training.gradient_audit_interval_steps: 0` or pass `--gradient-audit-interval-steps 0` to disable; a positive CLI value overrides YAML, and an omitted value follows YAML. `gradient_audit_batch_size` controls probe size (default32). These diagnostics settings do not change scientific identity. Missing-label objectives are `null` with `no_valid_targets`; absent electronic/charge labels in this small natural probe are not evidence of weak gradients. Norms are from separate objectives, not the vector sum: do not automatically adjust coefficients based solely on their magnitude. Audits use one additional forward and up to eight gradient calculations; OOM is an explicit failure, not an automatic probe resize. Monitor with:

```bash
tail -f outputs/v4/stage1/base/train/gradient_audit.jsonl
```

Epoch-boundary resume appends attempt-tagged observations without deleting failed-attempt rows. Audit never uses validation derivatives to update parameters; test is not read.

### Partial-charge sidecar on an existing corpus

The parser accepts `@<TRIPOS>MOLM` as a known typo for `@<TRIPOS>ATOM`; do not edit the original MOL2 or manifest SHA. All other parse/integrity checks remain strict. This compatibility policy enters Stage1 charge-source identity: retain earlier runs and use a fresh sidecar output/cache path when their identity differs.

Normal Stage1 prepare also prepares the separate atom-label sidecar. If corpus and teacher cache already exist, run only the sidecar command instead; it does not change either. Active Base reads `data/stage1/properties/partial_atomic_charge/train.csv` through `auxiliary.simulation_dir`, and verifies the MOL2 resources referenced by `auxiliary.partial_charge_manifest`. Existing electronic sources are still required for formal v4 training. The following command matches the current YAML cache path; use it only when the directory is unused or contains a compatible format2 sidecar:

```bash
python scripts/stage1/prepare.py --config configs/v4/stage1/base.yaml --partial-charge-only --output outputs/v4/stage1/base/partial_charge
```

Only exact canonical matches are labeled; charge statistics include all source observations of matched Stage1 training structures. Same-structure charge vectors remain separate observations, never averaged or selected by first/last row. Pretraining reuses one molecular forward but evaluates each observation separately. Corpus sampling, other losses and teacher cache remain unchanged; no seed propagation or test/valid label reading occurs. Corrupt/missing resources and parsing errors still fail; no-isomorphism records follow the audited skip policy below. The format2 sidecar and observation policy change charge-supervised training identity, so old sidecars/checkpoints cannot be resumed under this recipe. Full checkpoints retain `partial_charge_head`; encoder-only export does not. To opt out scientifically, set `loss.lambda_partial_charge: 0` in a separate self-contained YAML (not an execution-only switch).

If the default output already contains an old or failed run, retain it and prepare into a fresh directory, for example `--output outputs/v4/stage1/base/partial_charge_observations`. Before training, set Base `auxiliary.partial_charge_cache` and regression `partial_charge_cache` to that directory's `artifacts/` subdirectory. Do not relabel hashes or delete old outputs to bypass compatibility checks.

No-isomorphism records are skipped and recorded in `artifacts/mapping_audit.json` (`status: skipped`, `reason: no_graph_isomorphism`, split/CSV line/mol_id/SMILES/resource filename/SHA). `artifacts/metadata.json` includes `attempted_observations`, `source_observations` (successfully mapped) and `skipped_observations`. This masks only unavailable charge supervision, not corpus molecules or other losses. Missing/corrupt resources and parsing errors still stop preparation. The skip policy changes charge identity; prepare into a fresh directory rather than reusing artifacts from the previous fail-on-mapping recipe. Independent regression writes corresponding train/valid mapping audits in its output root.

### Independent frozen regression-head training

Partial-charge head training retains each source row as a separate sample, reusing the frozen atom representation for duplicate structures; see the [sidecar contract](#partial-charge-sidecar-on-an-existing-corpus) for pretraining and compatibility. Scalar electronic conflict and train/valid overlap checks remain unchanged. Before using the command below, verify the regression YAML's `simulation_dir`: it currently specifies `data/stage2`, while migrated local electronic/charge sources are under `data/stage1/properties`. Point a self-contained run YAML at the actual authority directory; this does not authorize moving or overwriting data.

Run explicitly after the final Stage1 epoch; nothing is automatically appended to pretraining. The command needs complete final `last.pt` or the final epoch checkpoint, not `stage1_encoder.pt`. Configure data/artifact paths and budgets in `configs/v4/stage1/regression_heads.yaml`. It encodes clean structures once, freezes the encoder, and independently trains all13 scalar heads plus atom charge. The current YAML explicitly uses MLPs: entity1024→512→256→1 for each electronic target, atom512→256→128→1 for partial charge, GELU and dropout0. Each is independently initialized using its existing task-local seed (656,385 / 164,353 parameters respectively), with no weight sharing. Shared schema default remains Linear, which starts from its trained row/head when predictor is omitted. Defaults:10 epochs, LR1e-4, constant LR, batch128, AdamW/WD0.01, BF16, clip1. Validation only reports original-unit MAE/RMSE; final means epoch10, never best. Test is not read. Use a new output for this MLP recipe; existing Linear outputs remain read-only, and Stage1/2/3 need not be rerun.

```bash
python scripts/stage1/regression.py --config configs/v4/stage1/regression_heads.yaml --checkpoint outputs/v4/stage1/base/train/last.pt --output outputs/v4/stage1/base/regression_mlp --device cuda:0
```

Optional subset, including use of old complete v4 checkpoints without an atom head:

```bash
python scripts/stage1/regression.py --config configs/v4/stage1/regression_heads.yaml --checkpoint outputs/v4/stage1/base/train/last.pt --tasks HOMO_eV LUMO_eV --output outputs/v4/stage1/base/regression_mlp_homo_lumo --device cuda:0
```

Default tasks are HOMO_eV/LUMO_eV, ESP_max/min/std/pos_frac, Dipole, Quadrupole, q_max/min/std/pos_frac, gap_eV and partial_atomic_charge. The q_* summaries are not atom-charge prediction. Train/valid canonical overlap and conflicting scalar electronic labels fail; partial-charge observations are retained separately. New task normalization fits full train only; Linear's initialized weights/bias are converted to preserve original-unit predictions. Regression output has a source-bound `representations.pt`, per-task `metrics.jsonl` and `regression_head.pt/json`, plus summary. It is not a replacement Stage1 encoder or a Stage3 reporting artifact; no Stage2/3 rerun is needed for this head-only experiment. Source checkpoint, encoder and other heads are read-only. Existing output cannot be overwritten; a failed head run restarts in a fresh directory. No teacher execution or teacher cache is needed for scalar-only post-training; atom-head initialization additionally validates its original sidecar/scaler identity.

Predictor configuration is YAML-only; CLI arguments above are unchanged. Copy the self-contained regression YAML to a new file and change only the desired recipe. For example:

```yaml
predictor:
  type: linear
predictor_overrides:
  HOMO_eV:
    type: mlp
    hidden_dims: [512, 256]
    activation: gelu
    dropout: 0.1
  partial_atomic_charge:
    type: residual_mlp
    hidden_dims: [256, 256]
    activation: gelu
    dropout: 0.1
```

Overrides replace the complete shared predictor recipe; unlisted targets use the shared default. To use one MLP recipe for all targets, set `predictor.type: mlp` and its hidden dimensions/activation/dropout instead. Types are `linear/mlp/residual_mlp`; activations are `gelu/relu/silu`, default GELU, dropout0. Electronic input is learned1024, atom input atom512, and output1 is fixed. MLP uses Linear→activation→Dropout hidden layers. Each residual block uses input→width→width with two dropouts and an Identity/projected shortcut; no normalization or post-add activation. Each hidden width is positive; nonlinear lists cannot be empty. No arbitrary module imports or input/output width overrides.

Linear retains pretrained initialization, numerical behavior and identity. Nonlinear heads are randomly initialized with stable task-local seeds, not pretrained-linear residual calibration; their initial raw predictions need not match the source. Structure enters the selected-target regression identity and requires a new output, but does not change any Stage1/2/3 identity. All predictors keep the same training/data/frozen-encoder rules, including rejection of partial-charge requests when the source lacks its atom head. Format2 stores the resolved structure, input width, parameter count, initialization, seed and hashes; `stage1.regression.load_regression_head(task_root, checkpoint_path)` reconstructs from the artifact without YAML and still accepts old format1 Linear heads. See [ADR-0093](adr/0093-stage1-configurable-frozen-predictors.md).

### Copy an existing teacher checkpoint to another server

The checkpoint can be copied without downloading it again. From the source repository root, replace `SERVER` with your SSH alias (`h100` or `szx`) and `/path/to/ILUME` with that server's repository root. Check any existing destination file first; if its hash matches, skip the copy. If it differs, retain it for inspection and use a fresh destination rather than overwriting it.

```bash
sha256sum assets/unimol2/modelzoo/84M/checkpoint.pt
ssh SERVER 'mkdir -p /path/to/ILUME/assets/unimol2/modelzoo/84M'
rsync -avP --partial --append-verify \
  assets/unimol2/modelzoo/84M/checkpoint.pt \
  SERVER:/path/to/ILUME/assets/unimol2/modelzoo/84M/checkpoint.pt
ssh SERVER 'sha256sum /path/to/ILUME/assets/unimol2/modelzoo/84M/checkpoint.pt'
```

Both hashes must equal the published SHA-256 above. Repeat for the second server. Create the isolated teacher environment on each server separately, then run the environment check and small audit there before full cache generation.

## Stage2 and Stage3 Base

```bash
python scripts/stage2/prepare.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/base/prepare
python scripts/stage2/train.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/base/train
python scripts/stage3/prepare.py --config configs/v4/stage3/base.yaml --output outputs/v4/stage3/base/prepare
python scripts/stage3/train.py --config configs/v4/stage3/base.yaml --fold 1 2 3 4 5 --output outputs/v4/stage3/base/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/base/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/base/valid
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/base/train --split test --ensemble-folds --output outputs/v4/stage3/base/test
```

Stop on any failure; downstream stages require final artifacts and intact manifests. Stage1 is permanently frozen, not merely LR=0; ObjectEncoder owns the1241→1024 input projection. Stage3 final file name, prediction CSV and gate diagnostics remain unchanged, but kinds/identities are v4. Hydration has no test. Simulation scalar evaluation uses the existing `--domain simulation --ensemble-folds` interface with v4 config and separate output.

## Core ablations

Complete Base first for paired authority/source checks. w/o Stage1 uses the same feature statistics, random frozen structure encoders and explicit descriptors:

```bash
python scripts/stage2/prepare.py --config configs/v4/ablations/no_stage1_stage2.yaml --output outputs/v4/ablations/no_stage1/stage2/prepare
python scripts/stage2/train.py --config configs/v4/ablations/no_stage1_stage2.yaml --output outputs/v4/ablations/no_stage1/stage2/train
python scripts/stage3/prepare.py --config configs/v4/ablations/no_stage1_stage3.yaml --output outputs/v4/ablations/no_stage1/stage3/prepare
python scripts/stage3/train.py --config configs/v4/ablations/no_stage1_stage3.yaml --fold 1 2 3 4 5 --output outputs/v4/ablations/no_stage1/stage3/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/evaluate.py --config configs/v4/ablations/no_stage1_stage3.yaml --checkpoint-dir outputs/v4/ablations/no_stage1/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/v4/ablations/no_stage1/stage3/valid
python scripts/stage3/evaluate.py --config configs/v4/ablations/no_stage1_stage3.yaml --checkpoint-dir outputs/v4/ablations/no_stage1/stage3/train --split test --ensemble-folds --output outputs/v4/ablations/no_stage1/stage3/test
python scripts/stage2/zero_update.py --config configs/v4/stage2/base.yaml --trained-encoder outputs/v4/stage2/base/train/stage2_encoder.pt --output outputs/v4/ablations/no_stage2/stage2_zero_update
python scripts/stage3/prepare.py --config configs/v4/ablations/no_stage2_stage3.yaml --output outputs/v4/ablations/no_stage2/stage3/prepare
python scripts/stage3/train.py --config configs/v4/ablations/no_stage2_stage3.yaml --fold 1 2 3 4 5 --output outputs/v4/ablations/no_stage2/stage3/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/evaluate.py --config configs/v4/ablations/no_stage2_stage3.yaml --checkpoint-dir outputs/v4/ablations/no_stage2/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/v4/ablations/no_stage2/stage3/valid
python scripts/stage3/evaluate.py --config configs/v4/ablations/no_stage2_stage3.yaml --checkpoint-dir outputs/v4/ablations/no_stage2/stage3/train --split test --ensemble-folds --output outputs/v4/ablations/no_stage2/stage3/test
python scripts/benchmarks/sweep.py --config configs/v4/ablations/no_stage3_home.yaml --output outputs/v4/ablations/no_stage3_home --max-workers 1
```

The last two controls retain their twenty-experimental-task boundary and no simulation auxiliary training. Compare five-fold task-equal validation first; test reports never select epochs/configuration. Existing baseline results need comparison-identity checks, not automatic reruns/relabeling.
