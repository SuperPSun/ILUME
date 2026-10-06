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
python -m pip install numpy==1.26.4 pandas==1.5.3 rdkit==2025.9.3
python -m pip install -e .
python -m pip install scipy joblib addict scikit-learn numba
python -m pip install --no-deps unimol-tools==0.1.3.post1
```

Place the official Uni-Mol2 84M checkpoint at `assets/unimol2/modelzoo/84M/checkpoint.pt` yourself. The project deliberately disables automatic model downloads. Verify the environment and local file before starting the cache audit:

```bash
python - <<'PY'
from pathlib import Path
import importlib.metadata
import numpy
import torch
from rdkit import rdBase
from unimol_tools.models import unimolv2

checkpoint = Path("assets/unimol2/modelzoo/84M/checkpoint.pt")
assert checkpoint.is_file(), f"Missing local checkpoint: {checkpoint}"
assert importlib.metadata.version("unimol_tools") == "0.1.3.post1"
assert torch.__version__ == "2.9.0+cu128", torch.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert str(unimolv2.MODEL_CONFIG_V2["weight"]["84m"]) == "modelzoo/84M/checkpoint.pt"
print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}; rdkit={rdBase.rdkitVersion}")
print(f"unimol_tools={importlib.metadata.version('unimol_tools')}; checkpoint={checkpoint}")
PY
```

The printed RDKit runtime version must exactly match `rdkit_version` in `outputs/v4/stage1/base/prepare/artifacts/metadata.json`; if it does not, stop and rebuild neither cache nor training data until the environment matches. Once the full teacher cache is complete, leave the isolated environment and return to the regular ILUME environment (replace `ilagent2` below if yours has a different name):

```bash
conda deactivate
conda activate ilagent2
```

Do not install Uni-Mol into the regular training environment. The teacher import is only needed for cache generation, never for Stage1 training or deployment.

First generate an independent small audit and inspect `manifest.json` (attempted/valid/failed, throughput and embedding bytes); this audit is not a training cache:

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 32 --limit 100 --output outputs/v4/stage1/base/teacher_audit
```

After approving success rate, throughput and storage, generate the full cache. Rerunning the identical command resumes immutable shards, not arbitrary partial rows. OOM is a failed run, not a masked molecule. Batch size/device are execution settings; canonical structure order and per-molecule conformer seed are fixed.

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 32
```

Return to the original training environment. Stage1 refuses partial/unbound teacher caches; losses weight roles2/2/1 while the loader remains the original natural shuffle.

```bash
python scripts/stage1/train.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/train
```

Output includes resume checkpoints with auxiliary heads and encoder-only `stage1_encoder.pt`. Deployment/Stage2 do not load teacher or auxiliary heads.

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
