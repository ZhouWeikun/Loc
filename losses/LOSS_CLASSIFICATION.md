# Loss Classification

## Scope

This note classifies the metric-learning losses in `losses/` using two primary dimensions.

For the current cleanup, we temporarily exclude the Dirichlet-energy style loss:

- `WeightedDirichletEnergyLoss`

Reason:

- it optimizes positive-edge smoothness directly
- it is not a ranking loss over positive-vs-negative edges

## Unified Input Assumption

For later refactoring, we assume the mainline losses will gradually unify to the `feat_mat` interface:

- `feat_mat[i, j]` means the feature distance between row-anchor `i` and candidate edge `j`

Under this view, most current losses are not truly "anchor-free".
They use an implicit row anchor:

- each row is one anchor
- each column entry is one candidate positive or negative edge

## Primary Axis 1: Edge Aggregation Pattern

### 1. `single-hard-pair`

Per row, only one hardest positive edge and one hardest negative edge are used.

Typical form:

- `d_pos_hard = max(pos edges)`
- `d_neg_hard = min(neg edges)`
- optimize `d_pos_hard - d_neg_hard`

This family is structurally triplet-like.

### 2. `mined-set + logsumexp`

Per row, first mine a set of hard positive and hard negative edges, then aggregate them with `logsumexp`.

Typical form:

- mine `hard_pos_mask` and `hard_neg_mask`
- aggregate all hard edges instead of keeping only one pair

This family is structurally MS-like.

## Primary Axis 2: Whether Aggregation Uses Continuous Weights

### 1. `mask-only`

Positive and negative regions are defined by a binary mask, or by a threshold such as:

- `pos_weight > neg_weight`

But the final aggregation itself does not multiply by continuous edge weights.

### 2. `weighted`

Continuous weights participate in aggregation.
Typical cases:

- weight multiplies the selected hard-pair violation
- weight is used in the row-wise selection score
- weight multiplies all mined hard edges before `logsumexp`

## Main 2D Taxonomy

### A. `single-hard-pair + mask-only`

Representative idea:

- one hardest positive edge
- one hardest negative edge
- no continuous weighting in the final aggregation

Current classes:

- `HardTripleLoss_fm_mask`
- `HardTripleLoss_fm_weight`
- `SWTLoss_fm_mat`
- `WeightedSoftTripletLoss_v0`
- `WeightedSoftTripletLoss_v1`

Notes:

- `HardTripleLoss_fm_weight` takes `pos_weight/neg_weight`, but in the current implementation they are only used to generate the positive/negative partition. Its aggregation is still mask-only.
- `HardTripleLoss_fm_mask` and `SWTLoss_fm_mat` are close to the older soft-hard-triplet style.

### B. `single-hard-pair + weighted`

Representative idea:

- still keep only one hard positive edge and one hard negative edge per row
- but the chosen pair or the violation term is continuously weighted

Current classes:

- `SoftMultiSimLoss_Max`
- `SoftMultiSimLoss_WeightedMax`
- `WeightedSoftTripletLoss_v2`
- `WeightedSoftTripletLoss_v3`

Notes:

- despite the `MultiSim` naming, `SoftMultiSimLoss_Max` and `SoftMultiSimLoss_WeightedMax` are structurally still hard-pair losses
- the difference from the mask-only family is that weights affect either selection or penalty strength

### C. `mined-set + logsumexp + mask-only`

Representative idea:

- mine a hard positive set and a hard negative set
- aggregate the full hard set with `logsumexp`
- no continuous weight in aggregation

Current classes:

- `MSLoss_fm_mat`
- `MSLossComputer.compute_ms_loss`

Notes:

- this is the cleanest MS-style family in the current codebase
- it does not reduce to triplet unless the mined set degenerates to one pair or `logsumexp` becomes a max approximation

### D. `mined-set + logsumexp + weighted`

Representative idea:

- mine a hard set
- weight the violations
- aggregate with `logsumexp`

Current classes:

- `SoftMultiSimLoss_LogSum`
- `SoftWeightedRelativeMSLoss` with `mining_mode='all'`
- `MSLossComputer.compute_scl_loss`

Notes:

- this is the weighted MS-style family

## Hybrid Case

### `SoftWeightedRelativeMSLoss`

This class is a hybrid:

- `mining_mode='all'` -> `mined-set + logsumexp + weighted`
- `mining_mode='max'` -> `single-hard-pair + weighted`

So it should not be treated as belonging to only one fixed cell.

## Important Conceptual Notes

### 1. `MSLoss` is not automatically equal to `TripletLoss`

Even with hard mining, MS-style loss remains different from hard-pair triplet-style loss:

- triplet-style keeps one hardest pair
- MS-style keeps a mined hard set and uses `logsumexp`

MS-style only becomes close to triplet-style in a limiting sense:

- the hard set collapses to one pair
- or `logsumexp` behaves almost like `max`

### 2. "Anchor" should not be the primary classification axis here

Under the `feat_mat` view:

- each row is already an implicit anchor

So for this codebase, the more stable classification is:

- how edges are aggregated
- whether aggregation uses continuous weights

## Current Cleanup Recommendation

For this repository, the cleanest organization is:

1. `single-hard-pair + mask-only`
2. `single-hard-pair + weighted`
3. `mined-set + logsumexp + mask-only`
4. `mined-set + logsumexp + weighted`
5. `energy / smoothness regularizers`

If a future refactor unifies all mainline losses to `feat_mat`, this taxonomy should remain valid without depending on old file boundaries.
