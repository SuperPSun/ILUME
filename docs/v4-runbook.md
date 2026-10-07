# v4 execution

Run from repository root, using fresh `outputs/v4/` paths. Do not overwrite historical v3 outputs. Science contract: [ADR-0089](adr/0089-v4-frozen-dual-view-stage1.md). CUDA examples use four devices.

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
