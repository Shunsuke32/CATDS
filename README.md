# CATDS: Clip-Level Donor Selection for SSL model Continued Pre-Training

This repository is maintained as a CATDS execution repository.
The focus is clip-level donor selection, continued pre-training, fine-tuning, and ASR evaluation.

Paper: [Interspeech 2025 (PDF)](https://www.isca-archive.org/interspeech_2025/mitsumori25_interspeech.pdf)

## What is used

- CATDS scripts (`scripts/`)
- fairseq training configs (`configs/`)
- fairseq manifests (`data/manifests/`)
- custom fairseq task (`custom_task/`) for continued pre-training config `_9`

Raw audio and large checkpoints are intentionally not versioned.

## Quick Start (Docker)

Commands below assume Docker with GPU support and that you run inside `/workspace` (see `docker-compose.yaml`).

```bash
git clone <your-repo-url> CATDS
cd CATDS
docker compose run --rm w2v2-cpt-transfer
```

Inside the container:

```bash
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -P checkpoints/
```

For CTC beam decoding support, also install:

```bash
git clone https://github.com/flashlight/sequence && cd sequence && pip install . && cd ..
pip install flashlight-text
```

## Data

Manifests are expected under `data/manifests/pretrain` and `data/manifests/finetune`.
Depending on your branch/release, they may be generated locally or provided as part of a release artifact.

Audio should be placed exactly as referenced by the first line of each `.tsv` manifest.
Example (`data/manifests/pretrain/punjabi_train-10h.tsv`):

```tsv
/workspace/data/IndicSUPERB/punjabi/audio
844424933625601-500-m.wav       133747
844424931325353-1163-f.wav      100682
```

Dataset used in this CATDS workflow:

- IndicSUPERB: https://ai4bharat.iitm.ac.in/indicsuperb/

## CATDS Pipeline

### 0) Audio preprocessing (16kHz mono)

M4A -> WAV 16kHz mono:

```bash
python scripts/m4a_to_wav.py \
  --input-dir /work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio \
  --workers 8
```

Resample WAV recursively to 16kHz (mono load):

```bash
python scripts/resample_16000Hz.py \
  --input-dir /work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio \
  --target-sr 16000
```

Optional: find or move clips longer than 21.6s (single script only):

```bash
python scripts/wav_len_checker.py \
  --input-dir /work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio \
  --min-seconds 21.6 \
  --mode list \
  --output-csv /work/result/punjabi_longer_than_21_6s.csv
```

```bash
python scripts/wav_len_checker.py \
  --input-dir /work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio \
  --min-seconds 21.6 \
  --mode move \
  --move-dir /work/data/IndicSUPERB/kb_data_clean_m4a/tmp_long_files
```

### 1) Build donor clip groups (~21.6s)

```bash
python scripts/run_get_multiple_data_df.py \
  --wav-dir /work/data/IndicSUPERB/kb_data_clean_m4a/hindi/train/audio \
  --num-hours 0.006 \
  --num-sets 20000 \
  --output-csv /work/result/hindi_21_20000_full.csv
```

### 2) Prepare target representation space

Extract target embeddings (5h example):

```bash
python scripts/extract_embeddings.py \
  --checkpoint-path /work/checkpoints/xlsr2_300m.pt \
  --wav-dir /work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio \
  --num-hours 5 \
  --num-sets 1 \
  --output-parquet /work/tmp/punjabi.parquet
```

Train and apply target k-means:

```bash
python scripts/learn_k-means.py \
  /work/tmp/punjabi.parquet \
  /work/tmp/k-means_punjabi.joblib
```

```bash
python scripts/infer_k-means.py \
  /work/tmp/k-means_punjabi.joblib \
  /work/tmp/punjabi.parquet \
  /work/tmp/punjabi_clustered.parquet
```

### 3) Compute raw per-group CATDS score

`scripts/atds_token.py` now performs all of the following in one command:

1. donor-group embedding extraction
2. target k-means application
3. target sentencepiece training/loading
4. cosine similarity computation
5. per-group raw score + token count export

```bash
python scripts/atds_token.py \
  --groups-csv /work/result/hindi_21_20000_full.csv \
  --donor-wav-dir /work/data/IndicSUPERB/kb_data_clean_m4a/hindi/train/audio \
  --target-clustered-parquet /work/tmp/punjabi_clustered.parquet \
  --kmeans-model /work/tmp/k-means_punjabi.joblib \
  --checkpoint-path /work/checkpoints/xlsr2_300m.pt \
  --target-lang punjabi \
  --donor-lang hindi \
  --output-atds-csv /work/result/ATDS_hindi_21_20000_full.csv \
  --output-piece-counts-csv /work/result/piece_counts_sums_hindi_21_20000_full.csv \
  --tmp-dir /work/tmp
```

Note about `tgt_utts.txt`:
`atds_token.py` automatically creates target utterance text under `--tmp-dir` when `--spm-model` is not provided, so no separate manual workflow is required.

### 4) Fit quadratic correction equation

```bash
python scripts/corr_atds_tokens.py \
  --atds-csv /work/result/ATDS_hindi_21_20000_full.csv \
  --counts-csv /work/result/piece_counts_sums_hindi_21_20000_full.csv \
  --output-plot /work/result/hindi_bias_fit.png \
  --output-coef-csv /work/result/hindi_bias_coef.csv
```

The script prints:

`y = a*x^2 + b*x + c` where `x = piece_counts_sum`.

### 5) Normalize and select top-N groups

Put the fitted coefficients directly in CLI (`--coef-a --coef-b --coef-c`):
The example below uses `--top-n 4000`, which is 20% of the original 20,000 donor groups.

```bash
python scripts/sort_by_atds_token.py \
  --atds-csv /work/result/ATDS_hindi_21_20000_full.csv \
  --counts-csv /work/result/piece_counts_sums_hindi_21_20000_full.csv \
  --groups-csv /work/result/hindi_21_20000_full.csv \
  --coef-a -0.0000004010 \
  --coef-b 0.00083133 \
  --coef-c 0.26619643 \
  --top-n 4000 \
  --manifest-root /work/data/IndicSUPERB/kb_data_clean_m4a/hindi/train/audio \
  --output-manifest /work/data/manifests/pretrain/hindi_21_20000to4000_CATDS.tsv \
  --output-ranking-csv /work/result/hindi_21_20000_ranking.csv
```

## Training and Evaluation

### Configs used in this project

- continued pre-training: `configs/w2v2-large-cpt_indic-70h_9.yaml`
- fine-tuning: `configs/w2v2-large-finetune_punjabi-1h.yaml`

### Continued pre-training

`w2v2-large-cpt_indic-70h_9.yaml` uses:

- `common.user_dir: /work/custom_task`
- `task._name: temp_sampled_audio_pretraining`

So `custom_task/` is important for this CPT config.

```bash
fairseq-hydra-train \
  --config-dir /work/configs \
  --config-name w2v2-large-cpt_indic-70h_9 \
  dataset.train_subset='punjabi_10h_pretrain,hindi_21_20000to4000_CATDS'
```

### Convert checkpoint before fine-tuning (custom task case)

```bash
python scripts/convert-checkpoint.py /work/outputs/<cpt-run>/checkpoints/checkpoint_last.pt
```

### Fine-tuning

```bash
fairseq-hydra-train \
  --config-dir /work/configs \
  --config-name w2v2-large-finetune_punjabi-1h \
  model.w2v_path=/work/outputs/<cpt-run>/checkpoints/<converted-checkpoint>.pt
```

### ASR evaluation

```bash
python /fairseq/examples/speech_recognition/infer.py \
  /work/data/manifests/finetune/punjabi \
  --gen-subset test-2h_known_2 \
  --path /work/outputs/<finetune-run>/checkpoints/<checkpoint>.pt \
  --results-path /work/data/artefacts/asr-results/punjabi \
  --task audio_finetuning \
  --nbest 1 \
  --w2l-decoder viterbi \
  --criterion ctc \
  --labels ltr \
  --max-tokens 5000000 \
  --post-process letter
```

## Additional Documentation

- Python dependencies: `requirements.txt`
- `custom_task` provenance: `custom_task/README.md`
- Script notes: `scripts/README.md`

## Acknowledgement

This repository was built with reference to:

- [fauxneticien/w2v2-cpt-transfer](https://github.com/fauxneticien/w2v2-cpt-transfer.git)

Thank you to the original authors for open implementation details.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{mitsumori25_interspeech,
  title     = {{Cross-lingual Data Selection Using Clip-level Acoustic Similarity for Enhancing Low-resource Automatic Speech Recognition}},
  author    = {Shunsuke Mitsumori and Sara Kashiwagi and Keitaro Tanaka and Shigeo Morishima},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {3314--3318},
  doi       = {10.21437/Interspeech.2025-1399},
  issn      = {2958-1796},
}
```
