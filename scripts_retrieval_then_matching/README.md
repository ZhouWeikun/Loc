# Retrieval Then Matching

This directory contains the post-hoc retrieval -> GIM matching -> rerank/refine
analysis tools.

## Main Entrypoints

Run GIM matching/refine from saved Stage1 retrieval bundles:

```bash
conda run -n gim python /home/data/zwk/pyproj_neuloc_v0/scripts_retrieval_then_matching/retrieval_then_gim_refine.py
```

Recompute recall metrics from saved matching/refine details:

```bash
conda run -n gim python /home/data/zwk/pyproj_neuloc_v0/scripts_retrieval_then_matching/export_matching_refine_recall_from_details.py
```

## Output Root

Matching/refine outputs are written under:

```text
/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/mathing_refine
```

Each result records `refine_target_retrieval_dir`, which points back to the
Stage1 retrieval result directory used as the refine target.

## Notes

- Use the nested `scripts_retrieval_then_matching/*.py` paths for new
  commands.
- The old top-level wrapper has been removed to avoid path ambiguity.
- The recall recompute step reads existing `gim_refine_details.pt` files and
  does not rerun GIM matching.
