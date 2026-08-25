# ILUME result overview

## Stage 3 TEST (5-fold ensemble)

1. ILUME — macro normalized MAE 0.186944
2. MLP — macro normalized MAE 0.195299
3. ECFP+XGBoost — macro normalized MAE 0.269395
Coverage: 11 test tasks / 21 enabled Stage 3 tasks.

## Stage 3 VALIDATION (5-fold mean)

1. MLP — macro normalized MAE 0.214285
2. ILUME — macro normalized MAE 0.214773
3. ECFP+XGBoost — macro normalized MAE 0.302096

## Stage 2 CORE

1. MLP — macro normalized MAE 0.0676567
2. ECFP+XGBoost — macro normalized MAE 0.134703

## Partial Charge

No eligible run.
Not evaluated: ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost, mlp@outputs/benchmarks/v1/mlp
Not eligible: None

## Stage 2 FULL

No eligible run.
Not evaluated: ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost, mlp@outputs/benchmarks/v1/mlp
Not eligible: None

## Core task wins

- mlp@outputs/benchmarks/v1/mlp: 3

## Experiment health

- ✓ ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold1: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold2: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold3: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold4: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold5: complete
- ✓ ilume@outputs/v1/stage3/base/test: complete
- ✓ mlp@outputs/benchmarks/v1/mlp: complete
