# AGENTS.md

## Repo shape

- This is a script-oriented Python research repo, not an installable package: no `pyproject.toml`, `setup.py`, `requirements.txt`, or `__init__.py` files are present. Run scripts directly from the repo root unless a script documents otherwise.
- The root README describes the public release as the VLM/Stage 1 component, but this working tree also contains active Stage 2 Stable Cascade experiment code under `stage_2/`.
- Large/generated data are intentionally outside git: `data`, `artifacts`, `latents`, and `pca_data` are symlinks or ignored storage-backed paths. Do not treat missing files there as source deletions.

## Environments and tooling

- Prefer Conda envs over ad-hoc pip installs. `environment.yaml` defines `ppd`; `environments_backend.yaml` defines `ppd_clean`. PCA scripts expect `conda activate ppd_pca` and print install hints for that env.
- README install commands mention `pip install -e ".[train]"` and `requirements.txt`, but the corresponding packaging files are not in this tree. Trust the Conda YAMLs and script-local imports over README setup prose.
- There is no committed CI, lint, typecheck, pytest, or formatter config. Verification here is usually targeted `python -m py_compile ...` plus the relevant smoke script, not a repo-wide `pytest`/`ruff`/`mypy` command.
- `rg` may be unavailable in this environment; use `find`, AST-grep, or direct file reads when searching.

## Stage 1 entrypoints

- User classifier: `scripts/run_user_classify.sh` activates `rlhf`, assigns GPUs from `gpus=(0..7)`, and runs `python user_classification/user_classifier.py`. The Python script uses Abseil flags and requires a real `--dataset_name`, `--wandb_project`, and `--output_dir`; defaults contain TODO placeholders.
- LLaVA user embeddings: `scripts/gen_emb.sh` activates `ppd`, sets `CUDA_VISIBLE_DEVICES=1`, then calls `python llava_embeddings/pick_a_pick_user_emb_7b.py --device cuda:0 --device_map none`. The visible GPU is remapped, so `cuda:0` means the first GPU inside `CUDA_VISIBLE_DEVICES`.
- Multi-GPU embedding resume scripts are stateful: `scripts/gen_emb_multi_7b.sh` is hardcoded for chunks `67..99` with `CUDA_VISIBLE_DEVICES=0,1,2,3`; `scripts/gen_emb_resume.sh` is hardcoded for chunks `44..79`.
- GPT-4o evals require `OPENAI_API_KEY` in the environment. `scripts/eval_winrate_gpt4o.sh` currently has `datasets=()`, so it does nothing until that array is populated.

## Stage 2 mental model

- Read `stage_2/README.md` first. Its intended flow is: build UID manifests -> build pair assignments -> extract needed UIDs -> generate raw Stage C latents -> build latent manifests -> load `Stage2PreferenceDataset` -> patch Stage C -> smoke/train/infer.
- Core runtime files are top-level `stage_2/*.py`: `stage2_dataset.py`, `user_adapter.py`, `patch_stage_c.py`, `forward_only_stage2.py`, `train_step_smoke_stage2.py`, `train_smoke_stage2.py`, `train_stage2_full.py`, and `infer_stage2.py`.
- Offline prep lives in `stage_2/tasks/`; diagnostics live in `stage_2/analysis/`. `stage_2/tasks/latent/archive_legacy_diagnostics/` is historical debugging code, not the canonical latent path.
- Stage 2 scripts use sibling-import fallback patterns so they can run both from repo root and as `stage_2.*` imports. Preserve that style when moving shared utilities.

## Stage 2 data contracts

- `Stage2PreferenceDataset` is the schema authority. It flattens user rows into pair samples, accepts embedding JSON in list or columnar dict forms, and expects `emb` tensors shaped `[L, 3584]` by default.
- Assignment-backed samples use `preferred_image_uid_{i}`, `dispreferred_image_uid_{i}`, and `caption_{i}` fields, plus assignment JSONL `support_pairs`/`query_pairs` when provided.
- Canonical Stage C latents are raw Stability-AI EfficientNet features with shape `[16, 24, 24]`, `latent_semantics == "stability_train_c_raw_effnet"`, and `scaled == false`. Do not apply the diffusers/Wuerstchen `add(1).div(42)` scaling for the current Stage 2 training path.
- The default forward/training paths point at `data/user_emb_7b_full/*`, `artifacts/pair_assignments/*`, `latents/latents_24x24_stability_raw/*`, and split UID maps like `data/train_uid_to_path.json`.

## Stage 2 verification commands

- Dataset/model integration smoke:
  ```bash
  python stage_2/forward_only_stage2.py --local-files-only --summary-only
  ```
- One backward-pass DPO smoke, without optimizer step:
  ```bash
  python stage_2/train_step_smoke_stage2.py --local-files-only --summary-only
  ```
- Short optimizer-update smoke writes under `artifacts/stage2_train_smoke` by default:
  ```bash
  python stage_2/train_smoke_stage2.py --local-files-only --max-steps 2 --summary-only
  ```
- For syntax-only checks after edits, compile just the touched scripts, for example:
  ```bash
  python -m py_compile stage_2/stage2_dataset.py stage_2/patch_stage_c.py
  ```

## Stable Cascade / GPU gotchas

- `stage_2/tasks/latent/stability_raw_image_to_latents.py` imports `/data/roycecho/StableCascade/modules/effnet.py` and defaults checkpoints/cache to `/Data_Storage/roycecho/PPD/...`; portability fixes must account for those absolute paths.
- `--reference-device cuda` in Stage 2 smoke/full training means “same CUDA device as train” unless an explicit device like `cuda:1` is passed. Use explicit `--device cuda:0 --reference-device cuda:1` when splitting train/reference priors across two GPUs.
- Default patch paths are small and conservative: `down_blocks.0.2` and `down_blocks.0.5`. `--patch-all-attention-blocks` is available but changes memory and drift behavior substantially.
- User-conditioning trainable scope is intentionally narrow: markers include `.user_projection.`, `.user_adapter.k_proj.`, `.user_adapter.v_proj.`, optional `.user_adapter.out_proj.`, and optional `.user_scale`.

## PCA workflow

- Row-level PCA scripts live in `pca/user_row_pca/` and assume `ppd_pca` plus `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `joblib`, and `tqdm`.
- PCA treats one Stage 1 embedding row as one sample: mean-pool `emb [L,3584]` to `[3584]`. Current notes in `PPD_user_row_pca_progress_summary.md` say the train fit used 39,972 rows because `train_shard99.json` was effectively empty/malformed for required columns.
- Main PCA output root is `/data/roycecho/PPD/pca_data/user_row_pca`, with the storage-backed symlink also visible under `/Data_Storage/roycecho/PPD/pca_data/user_row_pca`.
