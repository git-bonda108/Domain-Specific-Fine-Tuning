#!/usr/bin/env python3
"""
Quick evaluation script for baseline model on software domain test set
Produces results for the interview challenge
"""

import os
import sys
import torch
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Domain-Specific Fine-Tuning - Evaluation Pipeline")
print("Challenge 1: Software Domain EN→NL Translation")
print("=" * 70)

# System info
print(f"\nSystem Information:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
print(f"  MPS: {torch.backends.mps.is_available()}")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"  Using: {device}")

# Load test data
print("\n" + "=" * 70)
print("Loading Software Domain Test Set...")
print("=" * 70)

test_data_path = "data/Dataset_Challenge_1.xlsx"
df = pd.read_excel(test_data_path)
sources = df["English Source"].tolist()
references = df["Reference Translation"].tolist()

# Clean data
sources = [str(s).strip() for s in sources if pd.notna(s)]
references = [str(t).strip() for t in references if pd.notna(t)]

print(f"Loaded {len(sources)} test samples")
print(f"\nSample data:")
for i in range(min(3, len(sources))):
    print(f"  [{i+1}] EN: {sources[i][:60]}...")
    print(f"      NL: {references[i][:60]}...")

# Load baseline model
print("\n" + "=" * 70)
print("Loading Baseline Model: Helsinki-NLP/opus-mt-en-nl")
print("=" * 70)

from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-nl"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

print(f"Model loaded: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Translate
print("\n" + "=" * 70)
print("Translating Software Domain Test Set...")
print("=" * 70)

def translate_batch(texts, model, tokenizer, device, batch_size=16):
    translations = []
    total = len(texts)
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)
            
            print(f"  Progress: {min(i + batch_size, total)}/{total} samples")
    
    return translations

translations = translate_batch(sources, model, tokenizer, device)

# Compute metrics
print("\n" + "=" * 70)
print("Computing Evaluation Metrics...")
print("=" * 70)

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

bleu = BLEU()
chrf = CHRF()
ter = TER()

refs = [[ref] for ref in references]

bleu_result = bleu.corpus_score(translations, refs)
chrf_result = chrf.corpus_score(translations, refs)
ter_result = ter.corpus_score(translations, refs)

# Results
print("\n" + "=" * 70)
print("EVALUATION RESULTS - SOFTWARE DOMAIN TEST SET")
print("=" * 70)

results = {
    "dataset": "Software Domain (Dataset_Challenge_1.xlsx)",
    "num_samples": len(sources),
    "model": model_name,
    "metrics": {
        "BLEU": round(bleu_result.score, 2),
        "BLEU-1": round(bleu_result.precisions[0], 2),
        "BLEU-2": round(bleu_result.precisions[1], 2),
        "BLEU-3": round(bleu_result.precisions[2], 2),
        "BLEU-4": round(bleu_result.precisions[3], 2),
        "chrF": round(chrf_result.score, 2),
        "TER": round(ter_result.score, 2),
        "Brevity_Penalty": round(bleu_result.bp, 4)
    },
    "timestamp": datetime.now().isoformat()
}

print(f"\nModel: {model_name}")
print(f"Test Samples: {len(sources)}")
print(f"\nMetrics:")
print(f"  BLEU:         {results['metrics']['BLEU']:.2f}")
print(f"    BLEU-1:     {results['metrics']['BLEU-1']:.2f}")
print(f"    BLEU-2:     {results['metrics']['BLEU-2']:.2f}")
print(f"    BLEU-3:     {results['metrics']['BLEU-3']:.2f}")
print(f"    BLEU-4:     {results['metrics']['BLEU-4']:.2f}")
print(f"  chrF:         {results['metrics']['chrF']:.2f}")
print(f"  TER:          {results['metrics']['TER']:.2f}")
print(f"  Brevity Pen.: {results['metrics']['Brevity_Penalty']:.4f}")

# Save results
print("\n" + "=" * 70)
print("Saving Results...")
print("=" * 70)

output_dir = Path("outputs/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

# Save metrics JSON
metrics_path = output_dir / "baseline_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Metrics: {metrics_path}")

# Save translations comparison
comparison_df = pd.DataFrame({
    "English_Source": sources,
    "Model_Translation": translations,
    "Reference_Dutch": references
})
translations_path = output_dir / "baseline_translations.xlsx"
comparison_df.to_excel(translations_path, index=False)
print(f"  Translations: {translations_path}")

# Sample translations
print("\n" + "=" * 70)
print("SAMPLE TRANSLATIONS")
print("=" * 70)

for i in range(min(10, len(sources))):
    print(f"\n[{i+1}]")
    print(f"  Source:     {sources[i]}")
    print(f"  Model:      {translations[i]}")
    print(f"  Reference:  {references[i]}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
