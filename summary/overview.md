# ILUME result overview

## Stage 3 TEST (5-fold ensemble)

1. MLP — macro normalized MAE 0.22534
2. ECFP+XGBoost — macro normalized MAE 0.269395
Coverage: 11 test tasks / 21 enabled Stage 3 tasks.

## Stage 3 VALIDATION (5-fold mean)

1. ECFP+XGBoost — macro normalized MAE 0.302096
2. MLP — macro normalized MAE 5.56352e+14

## Stage 2 CORE

1. MLP — macro normalized MAE 0.18921
2. ECFP+XGBoost — macro normalized MAE 0.236286
3. ILUME — macro normalized MAE 0.275687

## Partial Charge

1. ILUME — macro normalized MAE 0.143063
Not evaluated: ecfp_xgboost@outputs/benchmarks/ecfp_xgboost/sweep, mlp@outputs/benchmarks/mlp/sweep
Not eligible: None

## Stage 2 FULL

1. ILUME — macro normalized MAE 0.253583
Not evaluated: ecfp_xgboost@outputs/benchmarks/ecfp_xgboost/sweep, mlp@outputs/benchmarks/mlp/sweep
Not eligible: None

## Core target wins

- ecfp_xgboost@outputs/benchmarks/ecfp_xgboost/sweep: 1
- mlp@outputs/benchmarks/mlp/sweep: 4

## Experiment health

- ✓ ecfp_xgboost@outputs/benchmarks/ecfp_xgboost/sweep: complete
- ✓ ilume@outputs/v1/stage2/base/evaluate/test: complete
- ⚠ ilume@outputs/v1/stage3/base/evaluate/test: legacy (reporting_schema_missing)
- ⚠ ilume@outputs/v1/stage3/base/evaluate/valid_fold1: legacy (reporting_schema_missing)
- ⚠ ilume@outputs/v1/stage3/base/evaluate/valid_fold2: legacy (reporting_schema_missing)
- ⚠ ilume@outputs/v1/stage3/base/evaluate/valid_fold3: legacy (reporting_schema_missing)
- ⚠ ilume@outputs/v1/stage3/base/evaluate/valid_fold4: legacy (reporting_schema_missing)
- ⚠ ilume@outputs/v1/stage3/base/evaluate/valid_fold5: legacy (reporting_schema_missing)
- ✓ mlp@outputs/benchmarks/mlp/sweep: complete
