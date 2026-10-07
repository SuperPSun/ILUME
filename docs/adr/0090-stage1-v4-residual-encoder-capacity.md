# ADR-0090: Stage1-v4 Base residual encoder capacity

## Status

Accepted (2026-10-07). Revises only the encoder capacity in ADR-0089. There is one active v4 Base; historical v3, legacy and Capacity contracts are unchanged.

## Decision

- SMILES Transformer uses 12 layers instead of 8, with width512, 8 attention heads, FFN2048 and dropout0.10 unchanged.
- `model.graph_message_mode` accepts `shared` or `residual_blocks`. The default is `shared`, omitted from serialized configuration and encoder identity; historical parameter names, initialization order and forward operations remain unchanged. The residual mode is allowed only for `dual_view_v4`.
- In residual mode, `graph_depth=8` means eight complete independently parameterized directed message-passing blocks. In shared mode the original initial projection plus `depth−1` shared updates remains intact.
- Each residual block computes, using directed source/destination and reverse-edge indices:

```text
m = sum(incoming edges at source) − reverse_edge_state
h = h + Dropout(GELU(W_message(LayerNorm(m))))
h = h + Dropout(W₂(Dropout(GELU(W₁(LayerNorm(h))))))
```

`W_message` is a bias-free512→512 projection; the FFN is512→2048→512. Each block has its own two LayerNorms and projections. Features, masks, input projection, atom/bond readout and atom-mean pooling do not change. No attention, virtual node or other graph mechanism is added.
- The lightweight1024D residual fusion, learned1024, atom512, all five supervision families, coefficients, missing-label masks, role2/2/1 loss weighting, natural shuffle and fusion-only80/10/10 dropout remain unchanged. Training still uses ten epochs, global batch128, AdamW1e-4/WD0.01, existing warmup/cosine, BF16, clipping1 and complete-epoch resume. No automatic batch adjustment or gradient checkpointing is enabled.

## Parameter audit

With a fixed2,048-token vocabulary and the configured maximum sequence length:

| Component | Historical shared Base | Residual Base |
|---|---:|---:|
| SMILES encoder | 26,400,768 | 39,010,304 |
| Graph encoder | 964,779 | 19,613,867 |
| Fusion | 2,101,248 | 2,101,248 |
| Encoder-only | 29,466,795 | 60,725,419 |
| Disposable auxiliary heads | 4,400,528 | 4,400,528 |
| Training total | 33,867,323 | 65,125,947 |

Actual vocabulary size can change the embedding and masked-SMILES head counts. Deployment still exports only the two encoders and fusion.

## Identity and artifact boundary

The non-default graph mode enters training and encoder identity. Existing v4 artifact kinds/formats remain unchanged; complete configuration/identity and strict state loading reject cross-structure resume. Historical approximately30M artifacts load using their embedded shared-mode configuration. No warm-start, partial loading or parameter copying is supported.

The prepared corpus, train-only normalization statistics, feature-generation contract and offline Uni-Mol teacher cache identities do not include this architecture change and are reusable without regeneration or changes to failure masks. New Base must train Stage1 from scratch, then regenerate Stage2 representation caches and train Stage2, then prepare/train/evaluate Stage3. The learned1024+RDKit217→1241 ObjectEncoder input and permanent Stage1 freeze in Stage2/3 remain unchanged. w/o Stage1 constructs a random frozen encoder from the current Base configuration.

Default paths remain unchanged. Existing outputs are never moved, overwritten or deleted automatically; users must select fresh output paths or explicitly arrange archival and consistent downstream references.

## Verification and interpretation

Tests cover independent block parameters and counts, bonded/bondless/mixed graphs and differentiable zero gradients, legacy shared-mode configuration/state/prediction compatibility, cache identity reuse, strict epoch resume/export and downstream freezing. Implementation does not run formal prepare, cache generation or training. A future result compares the combined SMILES-depth and graph-block capacity change, not either component in isolation.
