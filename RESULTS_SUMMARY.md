# Domain-Specific Fine-Tuning — Benchmark Report

EN→NL machine translation adapted to the software/technology domain, engineered
as a dual-track fine-tuning pipeline with a four-metric evaluation harness.

---

## Approach

### Model selection

| Track | Model | Parameters | Rationale |
|-------|-------|------------|-----------|
| Encoder–decoder | Helsinki-NLP/opus-mt-en-nl (MarianMT) | 148M | Pre-trained EN→NL, strong baseline |
| Decoder-only | bigscience/bloom-560m + LoRA | 560M (1.2M trainable) | Multilingual, parameter-efficient |

### Fine-tuning strategy

- **Encoder–decoder** — full-parameter fine-tuning in PyTorch Lightning; AdamW with
  linear warmup (`encoder_decoder_trainer.py`).
- **Decoder-only** — LoRA (Low-Rank Adaptation) in instruction format, training only
  0.2% of the model's parameters, with k-bit training preparation
  (`decoder_only_trainer.py`).

### Evaluation harness

| Metric | Purpose |
|--------|---------|
| BLEU (sacreBLEU) | N-gram precision — industry standard |
| chrF | Character-level F-score |
| TER | Translation edit rate |
| COMET | Neural semantic-quality estimation |

Benchmarked on FLORES-devtest (general domain) and an 84-sentence
software-domain test set, with per-sample translation audits.

---

## Baseline results — software-domain test set (84 samples)

**Model**: Helsinki-NLP/opus-mt-en-nl (untuned baseline)

| Metric | Score |
|--------|-------|
| **BLEU** | **24.45** |
| BLEU-1 | 42.86 |
| BLEU-2 | 33.33 |
| BLEU-3 | 20.00 |
| BLEU-4 | 12.50 |
| **chrF** | **76.35** |
| **TER** | **34.32** |

### Sample translations

| # | Source (EN) | Model output (NL) | Reference (NL) |
|---|-------------|-------------------|----------------|
| 1 | Disconnected {1} | Verbinding verbroken {1} | Verbinding verbroken {1} |
| 2 | Increased contrast | Verhoogd contrast | Verhoogd contrast |
| 3 | Update window | Venster bijwerken | Updateperiode |

The baseline numbers above establish the reference point the fine-tuning tracks
are measured against; full per-sample outputs are versioned under
`outputs/evaluation/` (`baseline_metrics.json`, `baseline_translations.xlsx`).

---

## Repository guide

| File | Description |
|------|-------------|
| `encoder_decoder_trainer.py` | MarianMT full fine-tuning (PyTorch Lightning) |
| `decoder_only_trainer.py` | BLOOM-560m LoRA instruction tuning (PEFT) |
| `evaluation.py` / `run_evaluation.py` | Four-metric evaluation harness |
| `demo/app.py` | Live Streamlit translation demo (CPU) |
| `data/software_domain_test_set.xlsx` | Software-domain test set (84 EN–NL pairs) |
| `outputs/evaluation/` | Versioned metrics and per-sample translation audits |
