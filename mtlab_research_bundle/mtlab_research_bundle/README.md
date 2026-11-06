# MTLab research add-on: IBM alignment, phrase-based SMT and NMT

This bundle is designed to be copied into the existing project `D:\\machine translation evaluation`. It does **not** replace the existing `src/mtlab/experiments/tracking.py`, dataset preparation, MGIZA++ installation, or folder structure.

## What is included

1. **IBM alignment family:** a Python experiment runner that calls the already-installed MGIZA++ binaries. It supports IBM1, IBM2, HMM, IBM3, IBM4 and IBM5 in both directions. This is the practical full-corpus backend. It prepares vocabularies, SNT files, word classes and co-occurrence files, merges A3 output, and logs each run in MLflow. IBM4 and IBM5 are the primary research comparisons.
2. **Pure-Python IBM4/IBM5 reference:** an optional NLTK-backed implementation in `train_ibm_nltk.py`. It uses frequency-rank word classes and is for small subsets only; it is not a replacement for MGIZA++ on the full Samanantar corpus.
3. **Phrase-based SMT:** Python phrase extraction from bidirectional alignments, grow-diag-final-and symmetrization, phrase translation probabilities, approximate lexical weights, interpolated Kneser-Ney target LM, and a coverage-stack phrase decoder with a distortion penalty.
4. **NMT:** PyTorch GRU + Bahdanau attention, LSTM + Bahdanau attention, and encoder-decoder Transformer. It uses a shared SentencePiece BPE vocabulary trained from Samanantar and logs epoch losses/checkpoints to MLflow.
5. **Translation evaluation:** BLEU, chrF++, TER, optional FLORES spBLEU using the official FLORES SentencePiece model, and optional COMET. Results are logged to MLflow and exported as JSON/CSV/plots.

## Important scope and honesty

- MGIZA++ is the algorithm implementation for the full-corpus IBM4/IBM5 runs. Python is the orchestration, tracking, data preparation and reporting layer. The optional NLTK script is a genuinely Python-level reference implementation, but it is slow and should only be used on a few hundred pairs.
- The Python phrase decoder is a research baseline, not a drop-in reproduction of Moses. It implements phrase consistency, phrase translation probabilities, a Kneser-Ney n-gram LM, beam/coverage decoding and distance-based distortion. It does **not** implement Moses' full lexicalized MSD reordering, OSM, Hiero/SCFG, factored SMT, or syntax-based SMT. Those are separate model families and should not be claimed as completed by this bundle.
- A full Samanantar IBM4/IBM5 run and full phrase-table extraction are not guaranteed to finish in a few hours on a 16 GB laptop. Start with a clearly-labelled subset (default 50,000 pairs), then use `--max-pairs 0` for full corpus. Never report subset scores as full-corpus scores.
- Your `data/alignment_gold/en-hi/` directory is empty. No AER is calculated or required here. Translation evaluation uses FLORES reference translations and BLEU-family metrics.

## 1. Copy the files

Copy the bundle's `src/` and `scripts/` directories into the project root, merging them with the existing directories. Do not overwrite `src/mtlab/experiments/tracking.py`.

Files added:

```text
src/mtlab/smt/phrase_based.py
src/mtlab/smt/nltk_alignment.py
src/mtlab/nmt/models.py
scripts/train_ibm.py
scripts/train_ibm_nltk.py
scripts/train_phrase_smt.py
scripts/decode_phrase_smt.py
scripts/train_nmt.py
scripts/decode_nmt.py
scripts/evaluate_translation.py
scripts/plot_translation_results.py
```

## 2. Dependencies

Run from PowerShell at the project root:

```powershell
uv add --group experiment mlflow psutil nltk sentencepiece sacrebleu sacremoses matplotlib numpy
```

PyTorch is expected to already be installed in the existing `.venv`; do not reinstall it if it is already working with your CUDA setup. `nvidia-ml-py` is optional for NVIDIA system telemetry. COMET is optional and can be installed separately:

```powershell
uv add --group experiment unbabel-comet
```

The official FLORES SentencePiece model is separate from the model trained for NMT. Pass the official FLORES model to `evaluate_translation.py` for spBLEU.

## 3. Keep MLflow running

In terminal 1:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts/mlflow
```

Open `http://127.0.0.1:5000`. Run scripts from another terminal. The existing `tracking.py` sets the tracking URI to this server.

## 4. Train IBM4 and IBM5 using MGIZA++

The script expects your existing binaries at `D:/mt-tools/mgiza/install` and MSYS2 Perl at `C:/msys64/usr/bin/perl.exe`.

Start with 50,000 aligned pairs:

```powershell
uv run python scripts/train_ibm.py --max-pairs 50000 --threads 4 --models ibm4,ibm5
```

Run only IBM4:

```powershell
uv run python scripts/train_ibm.py --max-pairs 50000 --threads 4 --models ibm4
```

Run additional statistical alignment baselines:

```powershell
uv run python scripts/train_ibm.py --max-pairs 50000 --threads 4 --models ibm1,ibm2,hmm,ibm3
```

Train on the entire prepared Samanantar split (may take much longer):

```powershell
uv run python scripts/train_ibm.py --max-pairs 0 --threads 4 --models ibm4,ibm5
```

Default input files:

```text
data/processed/samanantar/en-hi/word/train.en
data/processed/samanantar/en-hi/word/train.hi
```

Default MGIZA workspace:

```text
D:/mtlab_work/python_pipeline/
```

The script writes `alignment_manifest.json`, which stores the paths and MLflow run IDs for both directions of each model. It does not need `train.ids` or `alignment_gold`.

MLflow experiment: `01-word-alignment`.

## 5. Optional pure-Python NLTK IBM4/IBM5 reference

This is useful for explaining/validating the model algorithms on a tiny corpus, not full-scale training:

```powershell
uv run python scripts/train_ibm_nltk.py --model ibm4 --direction en-hi --max-pairs 300 --iterations 1
uv run python scripts/train_ibm_nltk.py --model ibm5 --direction en-hi --max-pairs 300 --iterations 1
```

The NLTK implementation's IBM5 initializer trains IBM4 first. The class assignment here is frequency-rank-based, not Brown `mkcls`; for the main full corpus experiment use MGIZA++.

## 6. Train phrase-based SMT from IBM4 alignments

```powershell
uv run python scripts/train_phrase_smt.py --manifest D:/mtlab_work/python_pipeline/alignment_manifest.json --model ibm4
```

Train the IBM5-based phrase model:

```powershell
uv run python scripts/train_phrase_smt.py --manifest D:/mtlab_work/python_pipeline/alignment_manifest.json --model ibm5
```

Each command extracts consistent phrase pairs using forward/reverse alignments, computes phrase and lexical features, and trains a 3-gram interpolated Kneser-Ney Hindi LM.

Output example:

```text
D:/mtlab_work/python_pipeline/phrase_smt/ibm4/en-hi/
    phrase_table.sqlite
    target_lm.pkl
    model_info.json
```

MLflow experiment: `02-phrase-based-smt`.

## 7. Decode FLORES development or test sentences

Use the **prepared tokenized source file** as decoder input. Use raw FLORES references for evaluation after the decoder detokenizes hypotheses.

Example paths (adjust only if your actual filenames differ):

```powershell
uv run python scripts/decode_phrase_smt.py `
  --model-info D:/mtlab_work/python_pipeline/phrase_smt/ibm4/en-hi/model_info.json `
  --source data/evaluation/flores200/en-hi/word/dev.en `
  --output artifacts/translations/ibm4_dev.hi `
  --run-name ibm4-pbsmt-dev-distance
```

For an additional distance-based reordering comparison, change `--distortion-weight` and `--max-distortion`, for example:

```powershell
uv run python scripts/decode_phrase_smt.py `
  --model-info D:/mtlab_work/python_pipeline/phrase_smt/ibm4/en-hi/model_info.json `
  --source data/evaluation/flores200/en-hi/word/dev.en `
  --output artifacts/translations/ibm4_dev_monotone.hi `
  --distortion-weight 0 `
  --max-distortion 0 `
  --run-name ibm4-pbsmt-dev-monotone
```

Tune decoder weights on `dev`, not `devtest`.

## 8. Train NMT systems

Train a Transformer baseline:

```powershell
uv run python scripts/train_nmt.py --architecture transformer --max-train-pairs 50000 --epochs 5 --batch-size 16 --d-model 128 --layers 2
```

Train GRU + Bahdanau attention:

```powershell
uv run python scripts/train_nmt.py --architecture gru --max-train-pairs 50000 --epochs 5 --batch-size 16 --hidden-size 128
```

Train LSTM + Bahdanau attention:

```powershell
uv run python scripts/train_nmt.py --architecture lstm --max-train-pairs 50000 --epochs 5 --batch-size 16 --hidden-size 128
```

Switch recurrent attention to Luong dot-product attention:

```powershell
uv run python scripts/train_nmt.py --architecture gru --attention luong --max-train-pairs 50000 --epochs 5 --batch-size 16
```

Train the reverse direction by swapping the source/target files:

```powershell
uv run python scripts/train_nmt.py --architecture transformer --direction hi-en --train-en data/processed/samanantar/en-hi/word/train.hi --train-hi data/processed/samanantar/en-hi/word/train.en --valid-en data/processed/samanantar/en-hi/word/validation.hi --valid-hi data/processed/samanantar/en-hi/word/validation.en --max-train-pairs 50000 --epochs 5 --batch-size 16
```

Use `--max-train-pairs 0` for the full training split. The dataset streams from disk with a shuffle buffer rather than loading the entire corpus into RAM. On the 4 GB RTX 3050, reduce batch size to 8 if CUDA runs out of memory. One shared SentencePiece tokenizer is stored under `artifacts/nmt/tokenizer/`; each architecture/direction has its own MLflow run and best validation-loss checkpoint.

## 9. Decode NMT on FLORES

Example Transformer checkpoint:

```powershell
uv run python scripts/decode_nmt.py `
  --checkpoint artifacts/nmt/transformer/en-hi/best_model.pt `
  --source data/evaluation/flores200/en-hi/word/dev.en `
  --output artifacts/translations/transformer_dev.hi
```

For Hindi → English, use the `hi-en` checkpoint directory and `word/dev.hi` as the source. The decoder writes detokenized SentencePiece text.

## 10. Evaluate translation quality

Use raw FLORES references. Example:

```powershell
uv run python scripts/evaluate_translation.py `
  --source data/evaluation/flores200/en-hi/raw/dev.en `
  --reference data/evaluation/flores200/en-hi/raw/dev.hi `
  --hypothesis artifacts/translations/ibm4_dev.hi `
  --system ibm4-pbsmt `
  --direction en-hi `
  --dataset-split dev `
  --run-name ibm4-pbsmt-flores-dev
```

For official FLORES spBLEU, add `--spm-model PATH_TO_OFFICIAL_FLORES_SPM.model`.

For optional COMET, add `--comet`. On this laptop the script defaults to CPU for COMET to avoid exhausting the 4 GB GPU. Install COMET separately first.

Metrics:
- BLEU: SacreBLEU corpus BLEU.
- chrF++: SacreBLEU chrF with word order 2.
- TER: SacreBLEU TER.
- spBLEU: only when the official FLORES SentencePiece model is supplied.
- COMET: only when `--comet` is supplied and the COMET package/model is available.

Run the final version of the selected systems on FLORES `devtest` only after model/decoder decisions are frozen. The translation evaluation run is tracked in MLflow experiment `08-final-evaluation`.

## 11. Produce comparison plots

After evaluating several systems:

```powershell
uv run python scripts/plot_translation_results.py
```

Figures and a comparison CSV are written to:

```text
reports/figures/translation/
```

## 12. Smoke tests

The bundle was smoke-tested in an isolated environment on toy parallel data for:

- NLTK IBM4 and IBM5 training and alignment extraction.
- A3 parsing, forward/reverse symmetrization, phrase extraction, SQLite phrase-table creation, Kneser-Ney LM scoring and phrase decoding.
- GRU-attention, LSTM-attention and Transformer forward/backward passes and greedy decoding.
- SentencePiece training plus a miniature NMT train/validation epoch.

The Windows MGIZA++ executable, your local MLflow service, your actual Samanantar files and FLORES evaluation files cannot be executed in this isolated build environment. The MGIZA runner uses the binary paths and invocation conventions already established in your project; run it first with a subset before full corpus training.
