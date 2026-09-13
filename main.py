#!/usr/bin/env python3
"""
Domain-Specific Fine-Tuning Pipeline
Software Domain EN→NL Translation — Benchmark

This script orchestrates the complete fine-tuning and evaluation pipeline:
1. Encoder-Decoder model fine-tuning (MarianMT)
2. Decoder-Only model fine-tuning with LoRA (BLOOM)
3. Comprehensive evaluation on FLORES and Software domain test sets

Usage:
    python main.py --mode all           # Run complete pipeline
    python main.py --mode encoder       # Train encoder-decoder only
    python main.py --mode decoder       # Train decoder-only with LoRA
    python main.py --mode evaluate      # Evaluate existing models
    python main.py --mode baseline      # Evaluate baseline only
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd

from config import get_config, TrainingConfig
from encoder_decoder_trainer import EncoderDecoderTrainer, EncoderDecoderTranslator
from decoder_only_trainer import DecoderOnlyTrainer, DecoderOnlyTranslator
from evaluation import TranslationEvaluator, compare_models, run_baseline_evaluation
from data_loader import DataLoaderFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           Domain-Specific Fine-Tuning Pipeline                       ║
║       Software Domain EN→NL Translation — Benchmark Pipeline        ║
║                                                                      ║
║           • Encoder-Decoder (MarianMT)                               ║
║           • Decoder-Only with LoRA (BLOOM)                           ║
║           • Evaluation: BLEU, COMET, chrF, TER                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_info():
    """Print system and environment information"""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("=" * 60 + "\n")


def train_encoder_decoder(config: TrainingConfig):
    """Train encoder-decoder model"""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Encoder-Decoder Fine-Tuning (MarianMT)")
    logger.info("=" * 60)
    
    trainer = EncoderDecoderTrainer(config)
    model = trainer.train()
    
    return model


def train_decoder_only(config: TrainingConfig):
    """Train decoder-only model with LoRA"""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Decoder-Only Fine-Tuning with LoRA (BLOOM)")
    logger.info("=" * 60)
    
    trainer = DecoderOnlyTrainer(config)
    model = trainer.train()
    
    return model


def evaluate_models(config: TrainingConfig, encoder_model=None, decoder_model=None):
    """Evaluate all models"""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Comprehensive Evaluation")
    logger.info("=" * 60)
    
    evaluator = TranslationEvaluator(config)
    all_results = []
    
    # Baseline evaluation
    logger.info("\n--- Evaluating Baseline Model ---")
    baseline_results = run_baseline_evaluation()
    all_results.append(("Baseline (MarianMT)", baseline_results))
    
    # Encoder-decoder evaluation (if trained)
    if encoder_model is not None:
        logger.info("\n--- Evaluating Fine-tuned Encoder-Decoder ---")
        enc_results = evaluator.evaluate_model(
            model=encoder_model,
            model_name="finetuned_encoder_decoder"
        )
        all_results.append(("Fine-tuned Encoder-Decoder", enc_results))
    
    # Decoder-only evaluation (if trained)
    if decoder_model is not None:
        logger.info("\n--- Evaluating Fine-tuned Decoder-Only (LoRA) ---")
        dec_results = evaluator.evaluate_model(
            model=decoder_model,
            model_name="finetuned_decoder_only_lora",
            is_decoder_only=True
        )
        all_results.append(("Fine-tuned Decoder-Only (LoRA)", dec_results))
    
    # Create comparison table
    if all_results:
        comparison_df = compare_models(all_results)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = Path(config.evaluation.results_dir) / "model_comparison.xlsx"
        comparison_df.to_excel(comparison_path, index=False)
        logger.info(f"\nComparison saved to {comparison_path}")
    
    return all_results


def run_full_pipeline(config: TrainingConfig):
    """Run the complete training and evaluation pipeline"""
    
    # Train encoder-decoder
    encoder_model = train_encoder_decoder(config)
    
    # Train decoder-only with LoRA
    decoder_model = train_decoder_only(config)
    
    # Evaluate all models
    results = evaluate_models(config, encoder_model, decoder_model)
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Domain-Specific Fine-Tuning Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "encoder", "decoder", "evaluate", "baseline"],
        default="all",
        help="Execution mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (JSON)"
    )
    parser.add_argument(
        "--encoder-model-path",
        type=str,
        default=None,
        help="Path to pre-trained encoder model for evaluation"
    )
    parser.add_argument(
        "--decoder-model-path",
        type=str,
        default=None,
        help="Path to pre-trained decoder model for evaluation"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    print_system_info()
    
    # Load configuration
    config = get_config()
    
    if args.no_wandb:
        config.use_wandb = False
    
    if args.device != "auto":
        config.device = args.device
    
    # Execute based on mode
    start_time = datetime.now()
    logger.info(f"Starting pipeline at {start_time.isoformat()}")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == "all":
            results = run_full_pipeline(config)
            
        elif args.mode == "encoder":
            encoder_model = train_encoder_decoder(config)
            evaluate_models(config, encoder_model=encoder_model)
            
        elif args.mode == "decoder":
            decoder_model = train_decoder_only(config)
            evaluate_models(config, decoder_model=decoder_model)
            
        elif args.mode == "evaluate":
            # Load pre-trained models if paths provided
            encoder_model = None
            decoder_model = None
            
            if args.encoder_model_path:
                logger.info(f"Loading encoder model from {args.encoder_model_path}")
                # Load model logic here
                
            if args.decoder_model_path:
                logger.info(f"Loading decoder model from {args.decoder_model_path}")
                # Load model logic here
                
            evaluate_models(config, encoder_model, decoder_model)
            
        elif args.mode == "baseline":
            logger.info("Running baseline evaluation only")
            baseline_results = run_baseline_evaluation()
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise
    
    # Print completion summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Start time: {start_time.isoformat()}")
    print(f"End time: {end_time.isoformat()}")
    print(f"Duration: {duration}")
    print("=" * 60)


if __name__ == "__main__":
    main()
