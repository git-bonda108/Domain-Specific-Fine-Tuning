# Architecture

This document describes the system as it exists in the code. Every claim below is checkable against a named file.

## Component map

| Component | File | Responsibility |
|-----------|------|----------------|
| CLI orchestrator | `main.py` | Parses `--mode`, `--device`, `--no-wandb`; dispatches the training/evaluation phases; prints system info and timing |
| Configuration | `config.py` | Five dataclasses — `DataConfig`, `EncoderDecoderConfig`, `DecoderOnlyConfig`, `EvaluationConfig`, composed into `TrainingConfig`; `get_config()` returns defaults |
| Data layer | `data_loader.py` | `DataLoaderFactory` (training-corpus loading with fallback chain, FLORES devtest, Excel test set, train/val split, `DataLoader` construction) and `TranslationDataset` (tokenization for both model families) |
| Encoder-decoder track | `encoder_decoder_trainer.py` | `EncoderDecoderTranslator` (a `pl.LightningModule` wrapping MarianMT) and `EncoderDecoderTrainer` (callbacks, loggers, `pl.Trainer`, model saving) |
| Decoder-only track | `decoder_only_trainer.py` | `DecoderOnlyTranslator` (a `pl.LightningModule` wrapping BLOOM-560M with PEFT LoRA adapters, optional 4/8-bit quantization paths) and `DecoderOnlyTrainer` (same trainer scaffolding); `load_lora_model()` for adapter reload |
| Evaluation | `evaluation.py` | `TranslationEvaluator` (BLEU/chrF/TER via sacrebleu, optional COMET), `EvaluationResults` dataclass, `compare_models()` table builder, `run_baseline_evaluation()` |
| Standalone baseline script | `run_evaluation.py` | Self-contained top-to-bottom script: load Excel test set → translate with untuned MarianMT → compute BLEU/chrF/TER → write JSON/XLSX artifacts |
| Demo notebook | `notebooks/01_domain_finetuning_demo.ipynb` | Interactive version of the baseline evaluation plus model-initialization demos; imports the same modules |

## Data flow, end to end

1. **Corpus acquisition** (`DataLoaderFactory.load_wmt16_data`): attempts `wmt16` `nl-en` from the Hugging Face Hub; on failure falls back to `opus100` `en-nl`; on a second failure falls back to `_create_fallback_training_data()` — 20 hard-coded software-domain EN-NL pairs plus up to 10,000 `opus_books` `en-nl` sentences. The result is truncated to `DataConfig.train_size` (default 50,000).
2. **Split and tokenize** (`create_dataloaders`): the first `min(val_size, len/10)` samples become validation; the rest train. `TranslationDataset.__getitem__` tokenizes per model family:
   - *Encoder-decoder:* source and target tokenized separately (`text_target=`), padded to fixed `max_length=128`, pad tokens in labels masked to `-100`.
   - *Decoder-only:* source and target concatenated into one instruction prompt — `"Translate English to Dutch:\nEnglish: {src}\nDutch: {tgt}"` — labels are a copy of `input_ids` with padding masked to `-100`.
3. **Training** (`EncoderDecoderTrainer.train` / `DecoderOnlyTrainer.train`): each track builds its `LightningModule`, computes total optimizer steps for the linear-warmup scheduler, seeds with `pl.seed_everything(42)`, and runs `pl.Trainer.fit` with gradient clipping 1.0, gradient accumulation (2 / 4 steps), and `16-mixed` precision on CUDA only.
4. **Persistence** (`_save_model`): the encoder-decoder saves full model + tokenizer; the decoder-only track saves LoRA adapters + tokenizer. Both write a `training_config.json` alongside. Lightning checkpoints (top-2 by `val_loss`, plus last) land in `outputs/*/checkpoints/`.
5. **Evaluation** (`TranslationEvaluator.evaluate_model`): for each model, translate FLORES devtest (if loadable) and the 84-sample Excel test set via the model's `translate()` method, then compute corpus BLEU (with n-gram precisions, brevity penalty, length ratio), chrF, TER, and — when the COMET package and checkpoint load — COMET `wmt22-comet-da`. Results are written three ways: `{model}_metrics.json`, `{model}_translations.xlsx`, and a human-readable `{model}_report.txt` with 10 sample translations.
6. **Comparison** (`compare_models`): all evaluated models are flattened into one DataFrame and saved as `outputs/evaluation/model_comparison.xlsx`.

## Orchestration analysis: sequential by design

Everything is sequential and synchronous, in a single process:

- `run_full_pipeline` is three blocking calls in order: encoder training → decoder training → evaluation. There is no parallel fan-out, no async I/O, no queue, no inter-process coordination.
- This is the right shape for the workload: both trainings contend for the same single accelerator (`devices=1` in both `pl.Trainer` instances), so running them concurrently would thrash GPU memory; and evaluation depends on the trained models being in memory (they are passed as Python objects, not reloaded from disk).
- The only concurrency in the system is data-loading worker processes (`num_workers=4`, `pin_memory=True` in both `DataLoader`s) and whatever intra-op parallelism PyTorch applies.
- Inference is batched, not parallel: `translate()` loops over the input in batches (32 for the encoder-decoder, 8 for the decoder-only model) under `torch.no_grad()`.

## State and context engineering

**No session state.** The system is a batch job. Durable state is exclusively filesystem artifacts under `outputs/`: checkpoints, saved models/adapters, TensorBoard event files, and the JSON/XLSX/TXT evaluation reports. Optional Weights & Biases logging (`use_wandb`, default off) adds an external run record.

**Reproducibility state:** a single seed (42) applied via `pl.seed_everything`, `deterministic=True` on both trainers, and `training_config.json` snapshots written next to every saved model.

**Context assembly for the decoder-only model** is the one place with prompt engineering:

- Training context: the full instruction prompt including the reference translation, truncated to `max_source_length + max_target_length` (256) tokens.
- Inference context: the same prompt minus the answer (`...\nDutch:`), truncated at 256 tokens; generation is capped at `max_new_tokens=128` with beam search (`num_beams=4`, `do_sample=False`).
- Output extraction: the generated ids are sliced past the prompt length, decoded, and cut at the first newline — a deliberate guard against the causal LM continuing beyond the translation.

## Design decisions and trade-offs visible in the code

1. **Two adaptation strategies, one harness.** Full fine-tuning (MarianMT) and LoRA (BLOOM) share `DataLoaderFactory`, the Lightning trainer scaffolding, and `TranslationEvaluator`, differentiated by an `is_decoder_only` flag. This keeps the comparison honest — same data, same metrics — at the cost of some duplicated trainer code between the two files.
2. **Graceful degradation over hard failure.** Both the training-corpus loader and the FLORES loader are fallback chains; COMET is a soft dependency (`try: import` → skip if absent). The pipeline prefers producing *some* result to crashing. The trade-off is silent variability: which corpus actually trained the model depends on what the Hub served that day, which weakens reproducibility (see EVALUATION.md).
3. **Fixed-length padding.** `TranslationDataset` pads every sample to `max_length` rather than dynamic per-batch padding. Simpler collation (the batches stack directly; the custom `collate_fn` in `data_loader.py` is defined but never wired in), at the cost of wasted compute on short UI strings.
4. **No prompt masking in decoder-only training.** Labels are the full prompt including the English source and instruction text, so loss is computed over the prompt as well as the translation. This is the simplest causal-LM setup; masking the prompt tokens to `-100` would focus the loss on the translation and is a known improvement point.
5. **Architecture-aware LoRA targeting.** `_get_target_modules()` maps model families (BLOOM, LLaMA/Mistral, Phi, GPT-2, OPT) to their attention/MLP module names, with a generic `["q_proj", "v_proj"]` fallback — making the decoder-only track swappable to other base models without code changes elsewhere.
6. **Precision policy.** `16-mixed` only on CUDA; MPS and CPU run fp32. The decoder-only model additionally loads in fp16 when CUDA is available. Quantized (4/8-bit) paths exist in `DecoderOnlyTranslator` but are off by default and not exposed through `config.py`.
7. **Selection and stopping.** Both trainers checkpoint the top-2 models by `val_loss`, keep `last`, and early-stop with patience 3 — model selection is by validation loss, not by translation metrics; BLEU/chrF/TER are computed only in the separate evaluation phase (`on_validation_epoch_end` is intentionally a no-op).
