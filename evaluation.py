"""
Evaluation Module for Translation Quality Assessment
Computes BLEU, COMET, chrF, and TER metrics on FLORES and Software domain test sets

Challenge 1, Task 3: Comprehensive evaluation with relevant metrics
"""

import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Evaluation libraries
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

# COMET
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    logging.warning("COMET not available. Install with: pip install unbabel-comet")

from config import TrainingConfig, get_config
from data_loader import DataLoaderFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    # Dataset info
    dataset_name: str
    num_samples: int
    
    # BLEU scores
    bleu_score: float
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    
    # Other metrics
    chrf_score: float
    ter_score: float
    
    # COMET score (if available)
    comet_score: Optional[float] = None
    
    # Timestamp
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Evaluation Results: {self.dataset_name}",
            f"{'='*60}",
            f"Number of samples: {self.num_samples}",
            f"\nBLEU Scores:",
            f"  BLEU:     {self.bleu_score:.2f}",
            f"  BLEU-1:   {self.bleu_1:.2f}",
            f"  BLEU-2:   {self.bleu_2:.2f}",
            f"  BLEU-3:   {self.bleu_3:.2f}",
            f"  BLEU-4:   {self.bleu_4:.2f}",
            f"\nOther Metrics:",
            f"  chrF:     {self.chrf_score:.2f}",
            f"  TER:      {self.ter_score:.2f}",
        ]
        
        if self.comet_score is not None:
            lines.append(f"  COMET:    {self.comet_score:.4f}")
        
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)


class TranslationEvaluator:
    """
    Comprehensive translation quality evaluator
    Computes multiple metrics for thorough quality assessment
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.eval_config = config.evaluation
        
        # Initialize metrics
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.ter = TER()
        
        # Initialize COMET if available
        self.comet_model = None
        if COMET_AVAILABLE and self.eval_config.compute_comet:
            try:
                logger.info(f"Loading COMET model: {self.eval_config.comet_model}")
                model_path = download_model(self.eval_config.comet_model)
                self.comet_model = load_from_checkpoint(model_path)
                logger.info("COMET model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load COMET model: {e}")
                self.comet_model = None
        
        # Output directory
        self.results_dir = Path(self.eval_config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Data loader
        self.data_factory = DataLoaderFactory(config)
    
    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU score and n-gram precisions"""
        
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]
        
        result = self.bleu.corpus_score(hypotheses, refs)
        
        return {
            "bleu": result.score,
            "bleu_1": result.precisions[0],
            "bleu_2": result.precisions[1],
            "bleu_3": result.precisions[2],
            "bleu_4": result.precisions[3],
            "bp": result.bp,  # Brevity penalty
            "ratio": result.sys_len / result.ref_len if result.ref_len > 0 else 0
        }
    
    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> float:
        """Compute chrF score (character-level F-score)"""
        refs = [[ref] for ref in references]
        result = self.chrf.corpus_score(hypotheses, refs)
        return result.score
    
    def compute_ter(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> float:
        """Compute TER (Translation Edit Rate)"""
        refs = [[ref] for ref in references]
        result = self.ter.corpus_score(hypotheses, refs)
        return result.score
    
    def compute_comet(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str]
    ) -> Optional[float]:
        """Compute COMET score using neural evaluation"""
        
        if self.comet_model is None:
            return None
        
        try:
            # Prepare data for COMET
            data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(sources, hypotheses, references)
            ]
            
            # Compute scores
            output = self.comet_model.predict(
                data,
                batch_size=self.eval_config.eval_batch_size,
                gpus=1 if torch.cuda.is_available() else 0
            )
            
            return output.system_score
            
        except Exception as e:
            logger.warning(f"COMET computation failed: {e}")
            return None
    
    def evaluate(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        dataset_name: str = "unknown"
    ) -> EvaluationResults:
        """
        Run full evaluation suite on translations
        
        Args:
            sources: Source texts (English)
            hypotheses: Model translations (Dutch)
            references: Reference translations (Dutch)
            dataset_name: Name for reporting
            
        Returns:
            EvaluationResults with all metrics
        """
        logger.info(f"Evaluating {len(hypotheses)} translations for {dataset_name}...")
        
        # Compute BLEU
        bleu_results = self.compute_bleu(hypotheses, references)
        
        # Compute chrF
        chrf_score = self.compute_chrf(hypotheses, references)
        
        # Compute TER
        ter_score = self.compute_ter(hypotheses, references)
        
        # Compute COMET
        comet_score = None
        if self.eval_config.compute_comet and self.comet_model is not None:
            comet_score = self.compute_comet(sources, hypotheses, references)
        
        results = EvaluationResults(
            dataset_name=dataset_name,
            num_samples=len(hypotheses),
            bleu_score=bleu_results["bleu"],
            bleu_1=bleu_results["bleu_1"],
            bleu_2=bleu_results["bleu_2"],
            bleu_3=bleu_results["bleu_3"],
            bleu_4=bleu_results["bleu_4"],
            chrf_score=chrf_score,
            ter_score=ter_score,
            comet_score=comet_score
        )
        
        logger.info(str(results))
        
        return results
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        is_decoder_only: bool = False
    ) -> Dict[str, EvaluationResults]:
        """
        Evaluate a model on both FLORES and Software domain test sets
        
        Args:
            model: Translation model with translate() method
            model_name: Name for reporting
            is_decoder_only: Whether model is decoder-only
            
        Returns:
            Dictionary with results for each test set
        """
        results = {}
        
        # Evaluate on FLORES devtest (general domain)
        logger.info("Loading FLORES devtest...")
        flores_sources, flores_refs = self.data_factory.load_flores_devtest()
        
        if flores_sources:
            logger.info(f"Translating {len(flores_sources)} FLORES samples...")
            flores_hyps = model.translate(flores_sources)
            
            results["flores_devtest"] = self.evaluate(
                sources=flores_sources,
                hypotheses=flores_hyps,
                references=flores_refs,
                dataset_name=f"{model_name}_FLORES_devtest"
            )
        else:
            logger.warning("FLORES devtest not available, skipping...")
        
        # Evaluate on Software domain test set
        logger.info("Loading Software domain test set...")
        software_sources, software_refs = self.data_factory.load_software_test_set()
        
        logger.info(f"Translating {len(software_sources)} Software domain samples...")
        software_hyps = model.translate(software_sources)
        
        results["software_domain"] = self.evaluate(
            sources=software_sources,
            hypotheses=software_hyps,
            references=software_refs,
            dataset_name=f"{model_name}_Software_Domain"
        )
        
        # Save detailed results with translations
        self._save_detailed_results(
            model_name=model_name,
            software_sources=software_sources,
            software_hyps=software_hyps,
            software_refs=software_refs,
            results=results
        )
        
        return results
    
    def _save_detailed_results(
        self,
        model_name: str,
        software_sources: List[str],
        software_hyps: List[str],
        software_refs: List[str],
        results: Dict[str, EvaluationResults]
    ) -> None:
        """Save detailed results including translations"""
        
        # Save metrics summary
        metrics_path = self.results_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {name: res.to_dict() for name, res in results.items()},
                f,
                indent=2
            )
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save translations comparison
        translations_df = pd.DataFrame({
            "source_english": software_sources,
            "model_translation": software_hyps,
            "reference_dutch": software_refs
        })
        
        translations_path = self.results_dir / f"{model_name}_translations.xlsx"
        translations_df.to_excel(translations_path, index=False)
        logger.info(f"Translations saved to {translations_path}")
        
        # Save human-readable report
        report_path = self.results_dir / f"{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Evaluation Report: {model_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            
            for name, res in results.items():
                f.write(str(res))
                f.write("\n")
            
            # Sample translations
            f.write("\n" + "=" * 60 + "\n")
            f.write("Sample Translations (Software Domain)\n")
            f.write("=" * 60 + "\n\n")
            
            for i in range(min(10, len(software_sources))):
                f.write(f"[{i+1}]\n")
                f.write(f"Source:     {software_sources[i]}\n")
                f.write(f"Model:      {software_hyps[i]}\n")
                f.write(f"Reference:  {software_refs[i]}\n")
                f.write("-" * 40 + "\n")
        
        logger.info(f"Report saved to {report_path}")


def compare_models(
    results_list: List[Tuple[str, Dict[str, EvaluationResults]]]
) -> pd.DataFrame:
    """
    Create comparison table for multiple models
    
    Args:
        results_list: List of (model_name, results_dict) tuples
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    
    for model_name, results in results_list:
        for dataset_name, eval_results in results.items():
            rows.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "BLEU": eval_results.bleu_score,
                "chrF": eval_results.chrf_score,
                "TER": eval_results.ter_score,
                "COMET": eval_results.comet_score if eval_results.comet_score else "N/A",
                "Samples": eval_results.num_samples
            })
    
    df = pd.DataFrame(rows)
    return df


def run_baseline_evaluation():
    """
    Evaluate baseline model (before fine-tuning) for comparison
    """
    from transformers import MarianMTModel, MarianTokenizer
    
    config = get_config()
    evaluator = TranslationEvaluator(config)
    
    # Load baseline model
    model_name = config.encoder_decoder.model_name
    logger.info(f"Loading baseline model: {model_name}")
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create simple translate function
    class BaselineWrapper:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        
        def translate(self, texts, batch_size=32):
            translations = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    ).to(self.device)
                    
                    outputs = self.model.generate(**inputs, max_length=128, num_beams=4)
                    decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    translations.extend(decoded)
            return translations
    
    wrapper = BaselineWrapper(model, tokenizer, device)
    
    results = evaluator.evaluate_model(
        model=wrapper,
        model_name="baseline_marian_en_nl"
    )
    
    return results


def main():
    """Run evaluation pipeline"""
    logger.info("Starting Evaluation Pipeline")
    
    # First, run baseline evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Baseline Model")
    logger.info("=" * 60)
    
    baseline_results = run_baseline_evaluation()
    
    print("\n\nBaseline Evaluation Complete!")
    print("=" * 60)
    for name, results in baseline_results.items():
        print(str(results))


if __name__ == "__main__":
    main()
