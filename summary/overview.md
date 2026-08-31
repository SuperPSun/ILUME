# ILUME result overview

## Stage 3 TEST (5-fold ensemble)

1. ILUME — macro normalized MAE 0.186134
2. MLP — macro normalized MAE 0.199194
3. D-MPNN — macro normalized MAE 0.244526
4. ECFP+XGBoost — macro normalized MAE 0.269395
5. MoLFormer-XL-both-10pct — macro normalized MAE 0.270395
Coverage: 11 test tasks / 21 enabled Stage 3 tasks.

## Stage 3 VALIDATION (5-fold mean)

1. ILUME — macro normalized MAE 0.215594
2. MLP — macro normalized MAE 0.220096
3. D-MPNN — macro normalized MAE 0.248322
4. ECFP+XGBoost — macro normalized MAE 0.302096
5. MoLFormer-XL-both-10pct — macro normalized MAE 0.303961

## Stage 2 CORE

1. D-MPNN — macro normalized MAE 0.0570515
2. MLP — macro normalized MAE 0.0676515
3. ILUME — macro normalized MAE 0.0883141
4. MoLFormer-XL-both-10pct — macro normalized MAE 0.0936612
5. ECFP+XGBoost — macro normalized MAE 0.134703

## Partial Charge

1. D-MPNN — macro normalized MAE 0.122218
2. ILUME — macro normalized MAE 0.141587
Not evaluated: ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost, mlp@outputs/benchmarks/v1/mlp, molformer@outputs/benchmarks/v1/molformer
Not eligible: None

## Stage 2 FULL

1. D-MPNN — macro normalized MAE 0.073343
2. ILUME — macro normalized MAE 0.101632
Not evaluated: ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost, mlp@outputs/benchmarks/v1/mlp, molformer@outputs/benchmarks/v1/molformer
Not eligible: None

## Core task wins

- dmpnn@outputs/benchmarks/v1/dmpnn: 3

## Experiment health

- ✓ dmpnn@outputs/benchmarks/v1/dmpnn: complete
- ✓ ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost: complete
- ✓ ilume@outputs/v1/stage2/base/evaluate: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold1: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold2: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold3: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold4: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold5: complete
- ✓ ilume@outputs/v1/stage3/base/test: complete
- ✓ mlp@outputs/benchmarks/v1/mlp: complete
- ✓ molformer@outputs/benchmarks/v1/molformer: complete
