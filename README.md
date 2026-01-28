# Domain-Specific Fine-Tuning Pipeline

## Challenge 1: Software Domain EN→NL Translation

A complete fine-tuning pipeline for domain-specific machine translation from English to Dutch, targeting the software/technology domain.

---

## Overview

This project implements:

1. **Encoder-Decoder Fine-Tuning** - MarianMT with PyTorch Lightning
2. **Decoder-Only Fine-Tuning with LoRA** - BLOOM-560M with PEFT
3. **Comprehensive Evaluation** - BLEU, COMET, chrF, TER metrics

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/git-bonda108/Domain-Specific-Fine-Tuning.git
cd Domain-Specific-Fine-Tuning

# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation
python run_evaluation.py

# Run full training pipeline
python main.py --mode all
```

---

## Project Structure

```
Domain-Specific-Fine-Tuning/
├── config.py                    # Configuration
├── data_loader.py               # Data loading
├── encoder_decoder_trainer.py   # Task 1: MarianMT
├── decoder_only_trainer.py      # Task 2: BLOOM + LoRA
├── evaluation.py                # Task 3: Metrics
├── main.py                      # Main script
├── run_evaluation.py            # Quick evaluation
├── requirements.txt             # Dependencies
├── SUBMISSION.md                # Assessment submission document
├── data/
│   └── Dataset_Challenge_1.xlsx
├── notebooks/
│   └── 01_domain_finetuning_demo.ipynb
└── outputs/
    └── evaluation/
        ├── baseline_metrics.json
        └── baseline_translations.xlsx
```

---

## Results

### Software Domain Test Set (84 samples)

| Metric | Score |
|--------|-------|
| BLEU | 24.45 |
| chrF | 76.35 |
| TER | 34.32 |

---

## Models

| Task | Model | Parameters |
|------|-------|------------|
| Encoder-Decoder | Helsinki-NLP/opus-mt-en-nl | 148M |
| Decoder-Only | bigscience/bloom-560m + LoRA | 560M (1.2M trainable) |

---

## Usage

```bash
# Encoder-decoder training only
python main.py --mode encoder

# Decoder-only with LoRA only
python main.py --mode decoder

# Evaluation only
python main.py --mode evaluate

# Full pipeline
python main.py --mode all
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS for GPU acceleration

---

## Author

Satya Bonda
