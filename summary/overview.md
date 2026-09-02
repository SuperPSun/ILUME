# ILUME result overview

## Stage 3 TEST (5-fold ensemble)

1. RDKit 2D MLP + Stage2 + HoME — macro normalized MAE 0.182611
2. RDKit 2D + HoME — macro normalized MAE 0.184132
3. ILUME — macro normalized MAE 0.186134
4. MLP — macro normalized MAE 0.199194
5. ILUME Stage3 Single-task MLP — macro normalized MAE 0.217962
6. SPMM — macro normalized MAE 0.218458
7. D-MPNN — macro normalized MAE 0.244526
8. ECFP+XGBoost — macro normalized MAE 0.269395
9. ILBERT — macro normalized MAE 0.269967
10. MoLFormer-XL-both-10pct — macro normalized MAE 0.270395
Coverage: 11 test tasks / 21 enabled Stage 3 tasks.

## Stage 3 VALIDATION (5-fold mean)

1. ILUME — macro normalized MAE 0.215594
2. MLP — macro normalized MAE 0.220096
3. RDKit 2D MLP + Stage2 + HoME — macro normalized MAE 0.220941
4. ILUME Stage3 Single-task MLP — macro normalized MAE 0.223073
5. RDKit 2D + HoME — macro normalized MAE 0.223459
6. D-MPNN — macro normalized MAE 0.248322
7. ILBERT — macro normalized MAE 0.268402
8. SPMM — macro normalized MAE 0.289178
9. ECFP+XGBoost — macro normalized MAE 0.302096
10. MoLFormer-XL-both-10pct — macro normalized MAE 0.303961

## Stage 2 CORE

1. D-MPNN — macro normalized MAE 0.0570515
2. SPMM — macro normalized MAE 0.058257
3. ILBERT — macro normalized MAE 0.0610342
4. MLP — macro normalized MAE 0.0676515
5. ILUME — macro normalized MAE 0.0883141
6. RDKit 2D MLP + Stage2 — macro normalized MAE 0.0917394
7. MoLFormer-XL-both-10pct — macro normalized MAE 0.0936612
8. ECFP+XGBoost — macro normalized MAE 0.134703

## Partial Charge

1. D-MPNN — macro normalized MAE 0.122218
2. ILUME — macro normalized MAE 0.141587
Not evaluated: ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost, ilbert@outputs/benchmarks/v1/ilbert, mlp@outputs/benchmarks/v1/mlp, molformer@outputs/benchmarks/v1/molformer, rdkit_2d_stage2@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/evaluate/test, spmm@outputs/benchmarks/v1/spmm
Not eligible: None

## Stage 2 FULL

1. D-MPNN — macro normalized MAE 0.073343
2. ILUME — macro normalized MAE 0.101632
Not evaluated: ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost, ilbert@outputs/benchmarks/v1/ilbert, mlp@outputs/benchmarks/v1/mlp, molformer@outputs/benchmarks/v1/molformer, rdkit_2d_stage2@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/evaluate/test, spmm@outputs/benchmarks/v1/spmm
Not eligible: None

## Core task wins

- dmpnn@outputs/benchmarks/v1/dmpnn: 2
- ilbert@outputs/benchmarks/v1/ilbert: 1

## Experiment health

- ✓ dmpnn@outputs/benchmarks/v1/dmpnn: complete
- ✓ ecfp_xgboost@outputs/benchmarks/v1/ecfp_xgboost: complete
- ✓ ilbert@outputs/benchmarks/v1/ilbert: complete
- ✓ ilume@outputs/v1/stage2/base/evaluate: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold1: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold2: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold3: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold4: complete
- ✓ ilume@outputs/v1/stage3/base/evaluate/fold5: complete
- ✓ ilume@outputs/v1/stage3/base/test: complete
- ✓ ilume_stage3_single_task_mlp@outputs/benchmarks/v1/ilume_stage3_single_task_mlp: complete
- ✓ mlp@outputs/benchmarks/v1/mlp: complete
- ✓ molformer@outputs/benchmarks/v1/molformer: complete
- ✓ rdkit_2d_home@outputs/ablations/stage1_stage2_rdkit_home/evaluate/test: complete
- ✓ rdkit_2d_home@outputs/ablations/stage1_stage2_rdkit_home/evaluate/valid/fold1: complete
- ✓ rdkit_2d_home@outputs/ablations/stage1_stage2_rdkit_home/evaluate/valid/fold2: complete
- ✓ rdkit_2d_home@outputs/ablations/stage1_stage2_rdkit_home/evaluate/valid/fold3: complete
- ✓ rdkit_2d_home@outputs/ablations/stage1_stage2_rdkit_home/evaluate/valid/fold4: complete
- ✓ rdkit_2d_home@outputs/ablations/stage1_stage2_rdkit_home/evaluate/valid/fold5: complete
- ✓ rdkit_2d_stage2@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/evaluate/test: complete
- ✓ rdkit_2d_stage2_home@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/test: complete
- ✓ rdkit_2d_stage2_home@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/valid/fold1: complete
- ✓ rdkit_2d_stage2_home@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/valid/fold2: complete
- ✓ rdkit_2d_stage2_home@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/valid/fold3: complete
- ✓ rdkit_2d_stage2_home@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/valid/fold4: complete
- ✓ rdkit_2d_stage2_home@outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/valid/fold5: complete
- ✓ spmm@outputs/benchmarks/v1/spmm: complete
