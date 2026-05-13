# scripts/

## Core pipeline scripts

- `extract_embeddings.py`: extracts XLSR layer-12 embeddings with VAD filtering.
- `learn_k-means.py`: trains MiniBatchKMeans on extracted embeddings.
- `infer_k-means.py`: assigns cluster IDs to embedding frames.
- `run_get_multiple_data_df.py`: builds donor clip groups (CSV) using `--num-hours` and `--num-sets`.
- `atds_token.py`: computes per-group raw CATDS scores and token counts (fully CLI-based).
- `corr_atds_tokens.py`: fits quadratic correction coefficients from raw scores vs token counts.
- `sort_by_atds_token.py`: applies coefficient-based normalization and exports top-N manifest rows.
- `convert-checkpoint.py`: converts custom-task checkpoints for standard wav2vec fine-tuning.

## Data preparation helpers

- `m4a_to_wav.py` (CLI): M4A -> 16kHz mono WAV
- `resample_16000Hz.py` (CLI): recursive WAV resampling
- `wav_len_checker.py` (CLI): list/move long clips by threshold (single unified tool)
- `wav_len_checker_move.py`: compatibility wrapper; use `wav_len_checker.py` instead

## Notes

- Prefer documenting a reproducible command in `README.md` when adding new scripts.
