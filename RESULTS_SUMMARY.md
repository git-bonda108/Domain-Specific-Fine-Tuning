# Domain-Specific Fine-Tuning - Results Summary

## Challenge 1: Software Domain EN→NL Translation

**Submitted by**: Satya Bonda  

---

## Assessment Requirements

### Task 1: Encoder-Decoder Fine-Tuning
Design and implement a software domain-specific fine-tuning pipeline for a small encoder-decoder Transformer model using PyTorch Lightning. The model must support Dutch language.

### Task 2: Decoder-Only Fine-Tuning  
Perform fine-tuning with a decoder-only model using LoRA-based techniques, instruction tuning, etc.

### Task 3: Evaluation
Evaluate with FLORES-devtest (general domain) and provided test set (software domain). Provide all relevant metrics.

---

## My Approach

### Model Selection

| Task | Model | Parameters | Rationale |
|------|-------|------------|-----------|
| Encoder-Decoder | Helsinki-NLP/opus-mt-en-nl | 148M | Pre-trained EN→NL, strong baseline |
| Decoder-Only | bigscience/bloom-560m + LoRA | 560M (1.2M trainable) | Multilingual, parameter-efficient |

### Fine-Tuning Strategy

**Encoder-Decoder**: Full parameter fine-tuning with PyTorch Lightning, AdamW optimizer, linear warmup

**Decoder-Only**: LoRA (Low-Rank Adaptation) - trains only 0.2% of parameters using instruction format

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| BLEU | N-gram precision (industry standard) |
| chrF | Character-level F-score |
| TER | Translation Edit Rate |
| COMET | Neural semantic quality |

---

## Results

### Software Domain Test Set (84 samples)

**Model**: Helsinki-NLP/opus-mt-en-nl (Baseline)

| Metric | Score |
|--------|-------|
| **BLEU** | **24.45** |
| BLEU-1 | 42.86 |
| BLEU-2 | 33.33 |
| BLEU-3 | 20.00 |
| BLEU-4 | 12.50 |
| **chrF** | **76.35** |
| **TER** | **34.32** |

### Sample Translations

| # | Source (EN) | Model Output (NL) | Reference (NL) |
|---|-------------|-------------------|----------------|
| 1 | Disconnected {1} | Verbinding verbroken {1} | Verbinding verbroken {1} |
| 2 | Increased contrast | Verhoogd contrast | Verhoogd contrast |
| 3 | Update window | Venster bijwerken | Updateperiode |

---

## Files to Review

| Priority | File | Description |
|----------|------|-------------|
| 1 | `encoder_decoder_trainer.py` | Task 1 - MarianMT fine-tuning |
| 2 | `decoder_only_trainer.py` | Task 2 - BLOOM + LoRA |
| 3 | `evaluation.py` | Task 3 - Metrics implementation |
| 4 | `outputs/evaluation/baseline_metrics.json` | Evaluation results |
| 5 | `outputs/evaluation/baseline_translations.xlsx` | All 84 translations |

---

## Repository Structure

```
translation-domain-finetuning/
+-- config.py                      # Configuration
+-- data_loader.py                 # Data loading
+-- encoder_decoder_trainer.py     # Task 1
+-- decoder_only_trainer.py        # Task 2
+-- evaluation.py                  # Task 3
+-- main.py                        # Main script
+-- run_evaluation.py              # Quick evaluation
+-- requirements.txt               # Dependencies
+-- data/
|   +-- Dataset_Challenge_1.xlsx   # Test set (84 samples)
+-- outputs/
    +-- evaluation/
        +-- baseline_metrics.json
        +-- baseline_translations.xlsx
```

---

## Instructions to Run

### Setup
```bash
git clone https://github.com/git-bonda108/translation-domain-finetuning.git
cd translation-domain-finetuning
pip install -r requirements.txt
```

### Quick Evaluation
```bash
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

## Technical Details

### Encoder-Decoder Config
```python
model_name = "Helsinki-NLP/opus-mt-en-nl"
learning_rate = 2e-5
batch_size = 16
num_epochs = 3
```

### LoRA Config
```python
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1
target_modules = ["query_key_value", "dense"]
```

---

## Repository

**GitHub**: https://github.com/git-bonda108/translation-domain-finetuning
