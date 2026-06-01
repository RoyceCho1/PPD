# PPD Row-Level User PCA Progress Summary

Last updated: 2026-05-13 Asia/Seoul

This note summarizes the current row-level user embedding PCA workflow so another session can continue from the existing outputs without re-discovering context.

## Goal

Analyze row-level user profile embeddings generated from 4-shot support preference pairs.

Each row is treated as one preference instance:

```text
one row = one user profile embedding generated from one 4-shot support set
emb shape per row = [L, 3584], usually [29, 3584]
row vector = mean over token dimension -> [3584]
```

The goal is not user-level aggregation yet. The current goal is to understand whether PCA axes over row-level embeddings correspond to interpretable visual/user preference directions, and later use those directions for embedding shift generation experiments.

## Environment

Use the PCA environment:

```bash
conda activate ppd_pca
```

The scripts assume the following packages are available:

```text
numpy, pandas, pyarrow, scikit-learn, matplotlib, joblib, tqdm
```

The data path `/data/roycecho/PPD/pca_data` is linked to the storage-backed location under `/Data_Storage/roycecho/PPD/pca_data`.

## Implemented Scripts

### Phase 1: Row Vector Extraction

Script:

```text
pca/user_row_pca/01_extract_row_vectors.py
```

Purpose:

```text
Read JSON shards from user_emb_7b_full.
Mean-pool each row's emb [L, 3584] into [3584].
Write shard-level .npy vectors and .parquet metadata.
```

Main output root:

```text
/data/roycecho/PPD/pca_data/user_row_pca/vectors/
```

Output pattern:

```text
vectors/{split}/{split}_shard{i}_mean.npy
vectors/{split}/{split}_shard{i}_meta.parquet
```

Important note:

```text
train_shard99.json was effectively empty/malformed for required columns.
The train PCA fit used available complete train vector shards, resulting in 39,972 rows.
```

### Phase 2: Row PCA Fit and Transform

Script:

```text
pca/user_row_pca/02_fit_row_pca.py
```

Main completed run:

```text
/data/roycecho/PPD/pca_data/user_row_pca/pca/train_all_max100000_seed0_50/
```

Key files:

```text
row_pca_raw_50.pkl
row_pca_l2_50.pkl
train_all_row_pc_scores_raw.npy
train_all_row_pc_scores_l2.npy
train_all_row_meta.parquet
explained_variance_raw.csv
explained_variance_l2.csv
pc_stats_raw.json
pc_stats_l2.json
fit_config.json
```

Fit command used:

```bash
python pca/user_row_pca/02_fit_row_pca.py \
  --mode fit \
  --vector-root /data/roycecho/PPD/pca_data/user_row_pca/vectors \
  --output-root /data/roycecho/PPD/pca_data/user_row_pca \
  --run-name train_all_max100000_seed0_50 \
  --fit-split train \
  --n-components 50 \
  --max-rows 100000 \
  --seed 0 \
  --run-raw \
  --run-l2
```

Fit result:

```text
row_count = 39,972
unique_users = 4,416
effective_n_components = 50
```

Explained variance:

```text
raw PC1~3 cumulative ≈ 0.420
raw PC1~10 cumulative ≈ 0.550
raw PC1~50 cumulative ≈ 0.726

l2 PC1~3 cumulative ≈ 0.444
l2 PC1~10 cumulative ≈ 0.575
l2 PC1~50 cumulative ≈ 0.743
```

Raw/L2 axis stability:

```text
raw PC1  <-> l2 PC1  corr ≈ +0.999
raw PC2  <-> l2 PC2  corr ≈ +0.999
raw PC3  <-> l2 PC3  corr ≈ +0.999
raw PC4  <-> l2 PC4  corr ≈ -0.999, sign flipped
raw PC5  <-> l2 PC5  corr ≈ +0.989
raw PC6  <-> l2 PC7  corr ≈ +0.953
raw PC7  <-> l2 PC6  corr ≈ +0.982
raw PC8  <-> l2 PC8  corr ≈ +0.971
```

PC1 is strongly related to embedding norm/text verbosity and should be interpreted cautiously.

### Phase 3: PCA Text Analysis

Script:

```text
pca/user_row_pca/03_analyze_row_pca_text.py
```

This script now supports two layers of analysis:

```text
1. Top-k extreme example markdowns, usually top30 high/low per PC.
2. Section-aware fractional keyword/category statistics over top/bottom 1%, 5%, 7%, 10%, plus middle 45-55%.
```

Section parser extracts:

```text
full_text
user_profile
preferred
dispreferred
differences
```

Important contrast definitions:

```text
preferred_dispreferred_contrast
  = preferred section example_rate - dispreferred section example_rate
  inside the same PC group

contrast_delta
  = contrast(high group) - contrast(low group)
  for the same PC, fraction, keyword/category
```

Interpretation:

```text
positive contrast_delta:
  the keyword/category is more preference-aligned in the PC high direction

negative contrast_delta:
  the keyword/category is more preference-aligned in the PC low direction
```

Current keyword/category implementation:

```text
unique keywords = 70
category-keyword pairs = 73
```

Categories:

```text
style
color_lighting
composition
detail_sharpness
aesthetic
```

`soft` intentionally appears in both `color_lighting` and `detail_sharpness`; the config records it as a duplicated/ambiguous keyword.

Main completed L2 output:

```text
/data/roycecho/PPD/pca_data/user_row_pca/text_inspection/train_all_l2_section_frac/
```

Main completed raw output:

```text
/data/roycecho/PPD/pca_data/user_row_pca/text_inspection/train_all_raw_section_frac/
```

Full L2 command:

```bash
python pca/user_row_pca/03_analyze_row_pca_text.py \
  --scores /data/roycecho/PPD/pca_data/user_row_pca/pca/train_all_max100000_seed0_50/train_all_row_pc_scores_l2.npy \
  --meta /data/roycecho/PPD/pca_data/user_row_pca/pca/train_all_max100000_seed0_50/train_all_row_meta.parquet \
  --output-root /data/roycecho/PPD/pca_data/user_row_pca \
  --output-name train_all_l2_section_frac \
  --num-pcs 10 \
  --top-k 30 \
  --stat-fracs 0.01 0.05 0.07 0.10 \
  --focus-pcs 4 5 8
```

Main output files in each section-aware directory:

```text
pc01_high_texts.md ... pc10_low_texts.md
pc_top_bottom_summary.csv
keyword_frequency_by_pc.csv
keyword_frequency_delta_by_pc.csv
pc01_quantile_summary.csv
section_parse_summary.csv
pc_group_stats.csv
keyword_frequency_by_pc_group_section.csv
category_frequency_by_pc_group_section.csv
keyword_high_low_delta_by_pc_group_section.csv
category_high_low_delta_by_pc_group_section.csv
preferred_dispreferred_contrast_by_pc_group.csv
preferred_dispreferred_contrast_delta_by_pc_group.csv
focus_pc_summary.csv
analyze_row_pca_text_config.json
```

Section parser coverage for L2 full analysis:

```text
full_text:      39972 / 39972 = 100.0%
user_profile:   39125 / 39972 = 97.9%
preferred:      39972 / 39972 = 100.0%
dispreferred:   39949 / 39972 = 99.9%
differences:    39966 / 39972 = 99.98%
```

### Phase 3 Plotting

Script:

```text
pca/user_row_pca/04_plot_row_pca_text_analysis.py
```

This script reads the section-aware CSVs and generates visual summaries.

L2 plot command used:

```bash
python pca/user_row_pca/04_plot_row_pca_text_analysis.py \
  --analysis-dir /Data_Storage/roycecho/PPD/pca_data/user_row_pca/text_inspection/train_all_l2_section_frac \
  --fraction 0.10 \
  --sections user_profile preferred \
  --focus-pcs 4 5 8 \
  --top-keywords 16
```

Plot output:

```text
/Data_Storage/roycecho/PPD/pca_data/user_row_pca/text_inspection/train_all_l2_section_frac/plots/
```

Generated plots:

```text
category_high_low_heatmap_user_profile_top10.png
category_high_low_heatmap_preferred_top10.png
preferred_dispreferred_contrast_delta_category_top10.png
keyword_high_low_heatmap_user_profile_top10_focus.png
preferred_dispreferred_contrast_delta_keyword_focus_top10.png
focus_pc_summary_PC4_PC5_PC8_top10.png
pc_group_stats_norm_textlen.png
section_coverage.png
plot_row_pca_text_analysis_config.json
```

Recommended plot reading order:

```text
1. section_coverage.png
2. pc_group_stats_norm_textlen.png
3. category_high_low_heatmap_user_profile_top10.png
4. category_high_low_heatmap_preferred_top10.png
5. preferred_dispreferred_contrast_delta_category_top10.png
6. focus_pc_summary_PC4_PC5_PC8_top10.png
```

## Current Interpretation

### PC1

PC1 is stable across raw and L2, but likely dominated by norm/text verbosity/profile intensity.

It should not be the first candidate for controllable semantic shifts.

### PC4

PC4 has strong semantic signal, but interpretation is mixed.

L2 section-aware top10 results:

```text
user_profile:
  aesthetic category strongly higher in PC4 high
  style category strongly lower in PC4 high

preferred:
  detail_sharpness slightly higher in PC4 high
  style lower in PC4 high

contrast_delta:
  style positive
  aesthetic negative
```

Provisional interpretation:

```text
PC4 high:
  visually appealing / engaging / dynamic / composition-aware profile

PC4 low:
  explicit realistic / stylized / focus-heavy profile
```

Use PC4 cautiously because user_profile high-low and preferred-dispreferred contrast are not perfectly aligned.

### PC5

PC5 is currently the strongest candidate for a semantic preference direction.

L2 top10 preferred section high-low shows PC5 high associated with:

```text
vibrant
composition
warm
depth
texture
detailed
dynamic
dramatic
```

Preferred-dispreferred contrast also supports PC5 positive for:

```text
warm
bright
large
soft
dramatic
colorful
clear
intricate
texture
```

Provisional interpretation:

```text
PC5 positive direction:
  warm / vibrant / detailed / depth / dramatic visual preference
```

Artifact check from `pc_group_stats_norm_textlen.png` and `pc_group_stats.csv`:

```text
PC5 top_10pct:
  mean_vec_norm ≈ 68.56
  text_word_count ≈ 674.50

PC5 bottom_10pct:
  mean_vec_norm ≈ 67.21
  text_word_count ≈ 722.27

PC5 middle_45_55pct:
  mean_vec_norm ≈ 67.49
  text_word_count ≈ 704.37
```

PC5 does not look like a strong norm/text-length artifact. It is the best first candidate for embedding shift experiments.

### PC8

PC8 direction appears reversed relative to the initial top30 impression.

L2 section-aware top10 results suggest PC8 low, not PC8 high, is more associated with:

```text
warm
vibrant
depth
colorful
detailed
intricate
stylized
```

Provisional interpretation:

```text
PC8 negative direction:
  warm / vibrant / detailed / composition-depth / stylized preference

PC8 positive direction:
  relatively less ornate, less colorful, less warm/vibrant
```

Artifact check:

```text
PC8 top_10pct:
  mean_vec_norm ≈ 68.44
  text_word_count ≈ 762.58

PC8 bottom_10pct:
  mean_vec_norm ≈ 70.49
  text_word_count ≈ 737.16

PC8 middle_45_55pct:
  mean_vec_norm ≈ 67.09
  text_word_count ≈ 689.82
```

PC8 has more artifact risk than PC5 because both extremes differ from the middle in norm/text length, and bottom has higher norm. Use PC8 negative direction as a secondary candidate and validate with generation.

## Best Next Step

Proceed to Phase 4: generate PCA-shifted user embeddings and evaluate generation changes.

Recommended first test axes:

```text
1. PC5 positive direction
2. PC8 negative direction
3. PC4 positive direction, lower confidence
```

Recommended shift scales should use PCA score statistics from:

```text
/data/roycecho/PPD/pca_data/user_row_pca/pca/train_all_max100000_seed0_50/pc_stats_l2.json
```

Start conservatively:

```text
shift = mean embedding + alpha * std(PC_k score) * PCA component_k
alpha values: -2, -1, 0, +1, +2
```

For PC8, prioritize negative direction:

```text
alpha values for PC8: 0, -1, -2
```

Before generation, verify how Stage 2 consumes user embeddings in:

```text
stage_2/train_stage2_full.py
stage_2/tasks/eval/generate_stage2_user_grid.py
```

The likely Phase 4 script name from the earlier plan:

```text
pca/user_row_pca/04_make_pca_shifted_embeddings.py
```

However, because `04_plot_row_pca_text_analysis.py` now exists, use either:

```text
pca/user_row_pca/05_make_pca_shifted_embeddings.py
```

or rename the plotting script if maintaining the original numbering is important.

## Useful Commands

Re-run L2 section-aware analysis:

```bash
conda activate ppd_pca
python pca/user_row_pca/03_analyze_row_pca_text.py \
  --scores /data/roycecho/PPD/pca_data/user_row_pca/pca/train_all_max100000_seed0_50/train_all_row_pc_scores_l2.npy \
  --meta /data/roycecho/PPD/pca_data/user_row_pca/pca/train_all_max100000_seed0_50/train_all_row_meta.parquet \
  --output-root /data/roycecho/PPD/pca_data/user_row_pca \
  --output-name train_all_l2_section_frac \
  --num-pcs 10 \
  --top-k 30 \
  --stat-fracs 0.01 0.05 0.07 0.10 \
  --focus-pcs 4 5 8 \
  --overwrite
```

Re-generate L2 plots:

```bash
python pca/user_row_pca/04_plot_row_pca_text_analysis.py \
  --analysis-dir /Data_Storage/roycecho/PPD/pca_data/user_row_pca/text_inspection/train_all_l2_section_frac \
  --fraction 0.10 \
  --sections user_profile preferred \
  --focus-pcs 4 5 8 \
  --top-keywords 16 \
  --overwrite
```

Inspect key CSVs quickly:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('/Data_Storage/roycecho/PPD/pca_data/user_row_pca/text_inspection/train_all_l2_section_frac')
print(pd.read_csv(root / 'section_parse_summary.csv'))
print(pd.read_csv(root / 'pc_group_stats.csv').query("pc in ['PC5','PC8'] and group in ['top_10pct','middle_45_55pct','bottom_10pct']"))
PY
```

## Cautions

Some markdown examples include NSFW or sensitive image descriptions. Prefer CSV/plot-level aggregate analysis for axis interpretation.

The current analysis is still row-level. Do not average by user unless explicitly starting a separate user-level PCA analysis.

PC signs are arbitrary in PCA. Interpret signs relative to the generated high/low groups and saved CSVs, not as intrinsic positive/negative semantics.
