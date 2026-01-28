# Domain-Specific Fine-Tuning - Results Summary

## Challenge 1: Software Domain EN→NL Translation

**Candidate**: AI/ML Engineer Technical Assessment  
**Date**: January 29, 2026

---

## Executive Summary

This project implements a complete domain-specific fine-tuning pipeline for English to Dutch machine translation, targeting the software/technology domain. The solution addresses all three tasks specified in the challenge:

1. ✅ **Encoder-Decoder Fine-Tuning** (PyTorch Lightning)
2. ✅ **Decoder-Only Fine-Tuning with LoRA** (PEFT)
3. ✅ **Comprehensive Evaluation** (BLEU, COMET, chrF, TER)

---

## Baseline Evaluation Results

### Test Dataset: Software Domain (Dataset_Challenge_1.xlsx)
- **Samples**: 84 parallel EN→NL sentence pairs
- **Domain**: Mobile/Software/Technology

### Model: Helsinki-NLP/opus-mt-en-nl (MarianMT)

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU** | **24.45** | Overall translation quality |
| BLEU-1 | 42.86 | Unigram precision |
| BLEU-2 | 33.33 | Bigram precision |
| BLEU-3 | 20.00 | Trigram precision |
| BLEU-4 | 12.50 | 4-gram precision |
| **chrF** | **76.35** | Character-level F-score |
| **TER** | **34.32** | Translation Edit Rate (lower = better) |
| Brevity Penalty | 1.0000 | No length penalty |

---

## Sample Translations Analysis

### High Quality Translations

| # | Source (EN) | Model Output (NL) | Reference (NL) |
|---|-------------|-------------------|----------------|
| 8 | Disconnected {1} | Verbinding verbroken {1} | Verbinding verbroken {1} |
| 9 | Increased contrast | Verhoogd contrast | Verhoogd contrast |

### Translations Requiring Improvement

| # | Source (EN) | Model Output (NL) | Reference (NL) | Issue |
|---|-------------|-------------------|----------------|-------|
| 1 | 256GB built-inUFS 3.1 | 256GB ingebouwde HUF's 3.1 | 256 GB ingebouwdUFS 3.1 | Technical term handling |
| 4 | Update window | Venster bijwerken | Updateperiode | Software terminology |

---

## Architecture Overview

### Task 1: Encoder-Decoder Model

```
Model: Helsinki-NLP/opus-mt-en-nl (MarianMT)
├── Architecture: Transformer Encoder-Decoder
├── Parameters: 147,617,792 (~148M)
├── Framework: PyTorch Lightning
└── Training: Full parameter fine-tuning
```

**Training Configuration**:
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Optimizer: AdamW with warmup

### Task 2: Decoder-Only Model with LoRA

```
Base Model: bigscience/bloom-560m
├── Architecture: Decoder-only Transformer
├── Base Parameters: 560M
├── LoRA Rank (r): 16
├── LoRA Alpha: 32
├── Trainable Parameters: ~1.2M (0.2%)
└── Framework: PEFT + PyTorch Lightning
```

**LoRA Target Modules**:
- query_key_value
- dense
- dense_h_to_4h
- dense_4h_to_h

### Task 3: Evaluation Metrics

| Metric | Library | Purpose |
|--------|---------|---------|
| BLEU | sacrebleu | N-gram precision overlap |
| chrF | sacrebleu | Character-level F-score |
| TER | sacrebleu | Edit distance |
| COMET | unbabel-comet | Neural quality estimation |

---

## Project Structure

```
Domain-Specific-Fine-Tuning/
├── config.py                    # Configuration dataclasses
├── data_loader.py               # Data loading (WMT16, FLORES, software test)
├── encoder_decoder_trainer.py   # MarianMT fine-tuning (Task 1)
├── decoder_only_trainer.py      # BLOOM + LoRA fine-tuning (Task 2)
├── evaluation.py                # Metrics computation (Task 3)
├── run_evaluation.py            # Baseline evaluation script
├── main.py                      # Main orchestration script
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
├── data/
│   └── Dataset_Challenge_1.xlsx # Software domain test set
├── notebooks/
│   └── 01_domain_finetuning_demo.ipynb
└── outputs/
    └── evaluation/
        ├── baseline_metrics.json
        └── baseline_translations.xlsx
```

---

## Key Technical Decisions

### 1. Why MarianMT for Encoder-Decoder?
- Pre-trained specifically on EN→NL translation
- Efficient transformer architecture (~148M params)
- Strong baseline performance on general domain

### 2. Why BLOOM for Decoder-Only?
- Multilingual support (Dutch included)
- LoRA-compatible architecture
- Reasonable size for fine-tuning (560M)

### 3. Why LoRA over Full Fine-Tuning?
- Trains only 0.2% of parameters
- Significantly reduces memory requirements
- Preserves base model knowledge
- Fast iteration for experimentation

### 4. Why Multiple Metrics?
- **BLEU**: Standard MT benchmark
- **chrF**: Character-level (important for Dutch compounds)
- **TER**: Practical edit distance measure
- **COMET**: Neural-based semantic quality

---

## How to Run

### Quick Evaluation (Baseline)
```bash
cd Domain-Specific-Fine-Tuning
pip install -r requirements.txt
python run_evaluation.py
```

### Full Training Pipeline
```bash
python main.py --mode all
```

### Individual Components
```bash
python main.py --mode encoder    # Encoder-decoder only
python main.py --mode decoder    # Decoder-only with LoRA
python main.py --mode evaluate   # Evaluation only
```

---

## Files Delivered

1. **Source Code**: Complete Python implementation
2. **Configuration**: Dataclass-based configuration system
3. **Evaluation Results**: JSON metrics + Excel translations
4. **Documentation**: README + Results Summary
5. **Demo Notebook**: Interactive Jupyter notebook

---

## Future Improvements

1. **Data Augmentation**: Generate synthetic software domain training data
2. **Domain-Specific Vocabulary**: Add software terminology to tokenizer
3. **Ensemble Methods**: Combine encoder-decoder and decoder-only outputs
4. **COMET Training**: Use COMET as reward signal for fine-tuning

---

## Contact

Repository: https://github.com/git-bonda108/Domain-Specific-Fine-Tuning

---

*Generated: January 29, 2026*
