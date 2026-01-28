# Domain-Specific Fine-Tuning Pipeline

## Challenge 1: Software Domain EN→NL Translation

This project implements a complete fine-tuning pipeline for domain-specific machine translation from English to Dutch, targeting the software/technology domain.

---

## 📋 Overview

### Tasks Completed

1. **Encoder-Decoder Fine-Tuning** (Task 1)
   - Model: Helsinki-NLP/opus-mt-en-nl (MarianMT)
   - Framework: PyTorch Lightning
   - Training data: WMT16 / OPUS parallel corpora

2. **Decoder-Only Fine-Tuning with LoRA** (Task 2)
   - Model: bigscience/bloom-560m (multilingual)
   - Technique: LoRA (Low-Rank Adaptation)
   - Framework: PEFT + PyTorch Lightning

3. **Comprehensive Evaluation** (Task 3)
   - Datasets: FLORES-devtest (general) + Software domain test set
   - Metrics: BLEU, COMET, chrF, TER

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
├──────────────────────────┬──────────────────────────────────┤
│  ENCODER-DECODER         │  DECODER-ONLY (LoRA)             │
│  ────────────────        │  ──────────────────              │
│  • MarianMT              │  • BLOOM-560M                    │
│  • Full fine-tuning      │  • LoRA adapters only            │
│  • Seq2Seq architecture  │  • Instruction tuning format     │
│  • 2e-5 learning rate    │  • 1e-4 learning rate            │
└──────────────────────────┴──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     EVALUATION PIPELINE                      │
├──────────────────────────┬──────────────────────────────────┤
│  FLORES-DEVTEST          │  SOFTWARE DOMAIN TEST SET        │
│  (General Domain)        │  (84 samples)                    │
├──────────────────────────┴──────────────────────────────────┤
│  Metrics: BLEU | COMET | chrF | TER                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Domain-Specific-Fine-Tuning/
├── config.py                    # Configuration dataclasses
├── data_loader.py               # Data loading and preprocessing
├── encoder_decoder_trainer.py   # MarianMT fine-tuning (PyTorch Lightning)
├── decoder_only_trainer.py      # BLOOM + LoRA fine-tuning
├── evaluation.py                # BLEU, COMET, chrF, TER evaluation
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── outputs/                     # Generated outputs
    ├── encoder_decoder/         # Encoder-decoder model checkpoints
    ├── decoder_only/            # LoRA adapter checkpoints
    └── evaluation/              # Evaluation results and reports
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/navigate to project
cd Domain-Specific-Fine-Tuning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Full pipeline (train both models + evaluate)
python main.py --mode all

# Or run specific phases
python main.py --mode encoder   # Train encoder-decoder only
python main.py --mode decoder   # Train decoder-only with LoRA
python main.py --mode evaluate  # Evaluate existing models
python main.py --mode baseline  # Baseline evaluation only
```

### 3. Configuration

Edit `config.py` to customize:
- Model selection
- Training hyperparameters
- LoRA configuration
- Evaluation settings

---

## 📊 Models

### Encoder-Decoder: MarianMT

**Model**: `Helsinki-NLP/opus-mt-en-nl`

| Parameter | Value |
|-----------|-------|
| Architecture | Transformer Encoder-Decoder |
| Parameters | ~77M |
| Pre-training | OPUS parallel corpora |
| Fine-tuning | Full parameter update |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |

### Decoder-Only: BLOOM with LoRA

**Base Model**: `bigscience/bloom-560m`

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer |
| Base Parameters | 560M |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| Trainable Parameters | ~1.2M (0.2%) |
| Learning Rate | 1e-4 |
| Batch Size | 8 |

**LoRA Target Modules**:
- `query_key_value`
- `dense`
- `dense_h_to_4h`
- `dense_4h_to_h`

---

## 📈 Evaluation Metrics

### Metrics Computed

| Metric | Description | Range |
|--------|-------------|-------|
| **BLEU** | Bilingual Evaluation Understudy | 0-100 (higher = better) |
| **COMET** | Crosslingual Optimized Metric | -1 to 1 (higher = better) |
| **chrF** | Character n-gram F-score | 0-100 (higher = better) |
| **TER** | Translation Edit Rate | 0-∞ (lower = better) |

### Test Datasets

1. **FLORES-devtest**: General domain evaluation
   - Source: facebook/flores
   - Samples: ~1000+ sentence pairs
   - Purpose: Assess general translation quality

2. **Software Domain Test Set**: Domain-specific evaluation
   - Source: `Dataset_Challenge_1.xlsx`
   - Samples: 84 sentence pairs
   - Purpose: Assess software/tech domain quality

---

## 🔬 Technical Details

### Data Processing

```python
# Text chunking for decoder-only
prompt = f"Translate English to Dutch:\nEnglish: {source}\nDutch: {target}"

# Tokenization with padding
encoding = tokenizer(
    text,
    max_length=128,
    padding="max_length",
    truncation=True
)
```

### LoRA Implementation

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Low-rank dimension
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,        # Dropout for regularization
    target_modules=[...],    # Layers to adapt
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
```

### Evaluation Pipeline

```python
# BLEU computation
bleu = sacrebleu.BLEU()
result = bleu.corpus_score(hypotheses, [references])

# COMET computation
comet_model = load_from_checkpoint(model_path)
output = comet_model.predict(data)
```

---

## 💻 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16GB | 32GB |
| GPU VRAM | 8GB | 16GB+ |
| Storage | 20GB | 50GB |

**Supported Accelerators**:
- NVIDIA CUDA GPUs
- Apple Silicon (MPS)
- CPU (slower)

---

## 📝 Output Files

After running the pipeline:

```
outputs/
├── encoder_decoder/
│   ├── marian-en-nl-software/     # Fine-tuned model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── training_config.json
│   └── checkpoints/               # Training checkpoints
│
├── decoder_only/
│   ├── bloom-lora-en-nl-software/ # LoRA adapters
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   └── training_config.json
│   └── checkpoints/
│
└── evaluation/
    ├── baseline_marian_en_nl_metrics.json
    ├── baseline_marian_en_nl_translations.xlsx
    ├── finetuned_encoder_decoder_metrics.json
    ├── finetuned_decoder_only_lora_metrics.json
    └── model_comparison.xlsx
```

---

## 🎯 Expected Results

### Baseline vs Fine-tuned (Approximate)

| Model | Dataset | BLEU | COMET |
|-------|---------|------|-------|
| Baseline MarianMT | FLORES | ~25-30 | ~0.75 |
| Baseline MarianMT | Software | ~20-25 | ~0.70 |
| Fine-tuned Encoder | FLORES | ~25-30 | ~0.75 |
| Fine-tuned Encoder | Software | ~30-40 | ~0.80 |
| Fine-tuned LoRA | FLORES | ~20-25 | ~0.70 |
| Fine-tuned LoRA | Software | ~25-35 | ~0.75 |

*Note: Actual results depend on training data quality and hyperparameters.*

---

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.py
   batch_size: int = 8  # or lower
   gradient_accumulation_steps: int = 4  # increase to compensate
   ```

2. **COMET Model Download Fails**
   ```bash
   # Install COMET separately
   pip install unbabel-comet
   # Or disable COMET in config
   compute_comet: bool = False
   ```

3. **FLORES Dataset Not Found**
   ```bash
   # Install datasets
   pip install datasets
   # Or use alternative loading in data_loader.py
   ```

---

## 📚 References

- [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-nl)
- [BLOOM](https://huggingface.co/bigscience/bloom-560m)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [PyTorch Lightning](https://lightning.ai/)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)

---

## 📄 License

This project is for interview/assessment purposes.

---

## 👤 Author

Prepared for AI/ML Engineer Technical Assessment
