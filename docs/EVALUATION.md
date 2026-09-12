# Evaluation

An honest account of what is measured today, what the code guards against, and what a production evaluation harness for this pipeline should look like.

## What exists

**There are no automated tests.** No `tests/` directory, no pytest/unittest suites, no CI configuration. What the repository does have is an executable *metrics* pipeline:

| Asset | How to run | What it covers |
|-------|-----------|----------------|
| `run_evaluation.py` | `python run_evaluation.py` | Baseline (untuned `Helsinki-NLP/opus-mt-en-nl`) on the 84-sample software-domain test set; computes corpus BLEU with n-gram precisions and brevity penalty, chrF, and TER via sacrebleu; writes JSON + XLSX artifacts |
| `evaluation.py` (`TranslationEvaluator`) | via `python main.py --mode baseline` (or after any training mode) | The same metric suite plus optional COMET (`Unbabel/wmt22-comet-da`), on both FLORES devtest (general domain) and the software-domain test set; writes per-model JSON metrics, XLSX translation dumps, and a text report with 10 sample translations |
| `compare_models()` in `evaluation.py` | invoked by `main.py` after training modes | Cross-model comparison table (BLEU/chrF/TER/COMET/sample count) saved to `outputs/evaluation/model_comparison.xlsx` |
| `notebooks/01_domain_finetuning_demo.ipynb` | Jupyter | Interactive re-run of the baseline evaluation and model-initialization sanity checks |

### Recorded metrics

The only metrics checked into the repository are the baseline numbers in `outputs/evaluation/baseline_metrics.json` (84 samples, `Helsinki-NLP/opus-mt-en-nl`, software-domain test set):

| Metric | Score |
|--------|-------|
| BLEU | 24.45 |
| BLEU-1 / 2 / 3 / 4 | 42.86 / 33.33 / 20.00 / 12.50 |
| chrF | 76.35 |
| TER | 34.32 |
| Brevity penalty | 1.0 |

The corresponding per-sentence outputs are in `outputs/evaluation/baseline_translations.xlsx`. No fine-tuned-model metrics are committed; producing them requires running the training modes.

### Metric choices (as implemented)

- **BLEU** (sacrebleu corpus BLEU): n-gram precision standard; the code also records the brevity penalty and length ratio, with a division-by-zero guard on the ratio.
- **chrF**: character-level F-score — relevant for Dutch compound words, where word-level n-grams under-credit near-misses.
- **TER**: edit-rate view of post-editing effort.
- **COMET** (`wmt22-comet-da`): neural, source-aware quality estimate. Strictly optional at runtime: the import, the checkpoint download, and the prediction call are each wrapped so failure degrades to "COMET: N/A" rather than aborting the run.

## Edge cases the code visibly handles

Enumerated from the source, not from intent:

- **Training-corpus fallback chain** (`data_loader.py:load_wmt16_data`): `wmt16` `nl-en` → `opus100` `en-nl` → `_create_fallback_training_data()` (20 hard-coded software-domain pairs + up to 10k `opus_books` sentences). Each hop is a caught exception with a logged warning.
- **FLORES fallback chain** (`data_loader.py:load_flores_devtest`): `facebook/flores` → `gsarti/flores_101` → empty lists. `TranslationEvaluator.evaluate_model` checks for the empty case and skips FLORES with a warning instead of crashing.
- **COMET as soft dependency** (`evaluation.py`): missing package, failed model download, and failed prediction are all caught; the score is reported as absent.
- **NaN rows in the Excel test set** (`data_loader.py:load_software_test_set`, `run_evaluation.py`): rows failing `pd.notna` are dropped and remaining values are `str()`-coerced and stripped.
- **Missing pad token** (`decoder_only_trainer.py`): if the causal-LM tokenizer has no pad token, EOS is substituted.
- **Padding excluded from loss**: label positions equal to the pad token are set to `-100` in both dataset code paths.
- **Decoder-only over-generation**: generated text is sliced past the prompt and truncated at the first newline, guarding against the causal LM continuing past the translation.
- **User interrupt** (`main.py`): `KeyboardInterrupt` exits cleanly with code 0; other exceptions are logged and re-raised.
- **Overfitting guards**: early stopping on `val_loss` (patience 3) and top-k checkpointing (k=2, plus `last`) in both trainers; gradient clipping at 1.0.
- **Hardware absence**: automatic CUDA → MPS → CPU selection in `main.py`, both trainers, and both evaluation scripts; mixed precision is only enabled on CUDA.

## Known gaps (visible in the code)

- `python main.py --mode evaluate --encoder-model-path/--decoder-model-path` does not actually load the checkpoints — the branches in `main.py` are placeholders — so this mode evaluates only the baseline. Fine-tuned models are evaluated only in the same process that trained them (or by wiring `load_lora_model()` / `from_pretrained` manually).
- In `load_software_test_set` and `run_evaluation.py`, sources and references are NaN-filtered *independently*; a NaN in only one column would silently misalign the parallel pairs downstream.
- Which training corpus was actually used depends on which Hub dataset loaded successfully at run time; the choice is logged but not recorded in the saved `training_config.json`, weakening run-to-run comparability.
- The `opus_books` fallback fetch is wrapped in a bare `except: pass`.
- `DataConfig.train_subset` says `"de-en"` (with a comment about filtering) while the loader requests `nl-en`/`en-nl`; the config field is effectively unused.
- Decoder-only training computes loss over the whole prompt (no prompt masking), so validation loss/perplexity partly measures the ability to reproduce the fixed instruction text.
- Translation metrics (BLEU/chrF/TER/COMET) are never computed during training; model selection is by token-level `val_loss` only.

## Proposed evaluation harness

None of the following exists yet; it is the harness this pipeline should have.

### 1. Unit and integration tests (pytest)

- `tests/test_data_loader.py`: paired NaN filtering keeps EN/NL alignment; fallback chain ordering; train/val split has no overlap; decoder-only tokenization masks padding and (once implemented) prompt tokens.
- `tests/test_evaluation.py`: metric functions against tiny fixtures with hand-computed BLEU/chrF/TER; identical hypothesis/reference yields BLEU 100, TER 0; empty-input behavior.
- `tests/test_translate.py`: `translate()` round-trip on a 2-sentence batch with a stubbed/tiny model — asserts output count equals input count and decoder-only outputs contain no prompt text.
- Smoke test: `--mode encoder` with `train_size=64`, one epoch, on CPU, asserting a checkpoint and a metrics JSON are produced.

### 2. Golden dataset

A frozen, versioned split committed to the repository (or pinned by hash):

- Shape: `{id, source_en, reference_nl, sub_domain (ui_string | spec | error_message), length_bucket}`.
- Sources: the existing 84-sample challenge set as the software-domain slice, plus a pinned FLORES devtest snapshot as the general-domain regression slice — the second slice exists to catch catastrophic forgetting of general-domain quality after domain fine-tuning.

### 3. Gates

Run on every change that touches training or data code:

- **Regression gate:** fine-tuned model must not drop more than an agreed epsilon (e.g. 1 BLEU / 1 chrF) on the FLORES slice relative to the recorded baseline.
- **Domain gate:** fine-tuned model must beat the committed baseline (BLEU 24.45 / chrF 76.35 / TER 34.32) on the software slice; otherwise the fine-tune is not earning its cost.
- **Determinism gate:** two seeded runs of the smoke config produce identical metric JSONs.

### 4. Metrics to add

- COMET reported per-slice (already implemented, but should be recorded in committed artifacts rather than best-effort).
- Terminology accuracy: exact-match rate on a curated software-domain glossary (e.g. "download", "instellingen", placeholder tokens like `{1}`), since placeholder corruption is the classic failure mode for UI-string MT and none of the current metrics isolates it.
- Statistical confidence: sacrebleu's paired bootstrap resampling when comparing models, so a 0.5-BLEU delta on 84 sentences is not over-read.
