"""
Configuration for Domain-Specific Fine-Tuning Pipeline
Challenge 1: Software Domain EN→NL Translation
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration"""
    # Training data
    train_dataset: str = "wmt16"
    train_subset: str = "de-en"  # We'll filter for software domain
    software_domain_dataset: str = "yhavinga/ccmatrix"  # Alternative Dutch data
    
    # Test data paths
    software_test_path: str = "data/Dataset_Challenge_1.xlsx"
    flores_dataset: str = "facebook/flores"
    flores_subset: str = "eng_Latn-nld_Latn"
    
    # Processing
    max_source_length: int = 128
    max_target_length: int = 128
    
    # Data splits
    train_size: int = 50000  # Number of training samples
    val_size: int = 2000


@dataclass
class EncoderDecoderConfig:
    """Configuration for Encoder-Decoder Model (MarianMT)"""
    model_name: str = "Helsinki-NLP/opus-mt-en-nl"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    
    # Model config
    max_length: int = 128
    num_beams: int = 4
    
    # Output
    output_dir: str = "outputs/encoder_decoder"
    model_save_name: str = "marian-en-nl-software"


@dataclass
class DecoderOnlyConfig:
    """Configuration for Decoder-Only Model with LoRA"""
    # Base model - using a smaller model suitable for translation
    model_name: str = "bigscience/bloom-560m"  # Multilingual decoder-only
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Generation config
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Output
    output_dir: str = "outputs/decoder_only"
    model_save_name: str = "bloom-lora-en-nl-software"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics
    compute_bleu: bool = True
    compute_comet: bool = True
    compute_chrf: bool = True
    compute_ter: bool = True
    
    # COMET model
    comet_model: str = "Unbabel/wmt22-comet-da"
    
    # Output
    results_dir: str = "outputs/evaluation"
    
    # Batch size for evaluation
    eval_batch_size: int = 32


@dataclass
class TrainingConfig:
    """Overall training configuration"""
    # Experiment
    experiment_name: str = "software-domain-finetuning"
    seed: int = 42
    
    # Hardware
    device: str = "auto"  # auto, cuda, mps, cpu
    fp16: bool = True
    bf16: bool = False
    
    # Logging
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "software-domain-finetuning"
    
    # Checkpointing
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    # Data
    data: DataConfig = field(default_factory=DataConfig)
    
    # Models
    encoder_decoder: EncoderDecoderConfig = field(default_factory=EncoderDecoderConfig)
    decoder_only: DecoderOnlyConfig = field(default_factory=DecoderOnlyConfig)
    
    # Evaluation
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_config() -> TrainingConfig:
    """Get default configuration"""
    return TrainingConfig()
