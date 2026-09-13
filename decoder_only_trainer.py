"""
Decoder-Only Fine-Tuning Pipeline with LoRA
Using BLOOM/LLaMA-style models with PEFT for EN→NL translation

Track 2: Decoder-only model fine-tuning with LoRA techniques
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from torch.optim import AdamW
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import re

from config import TrainingConfig, get_config
from data_loader import DataLoaderFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecoderOnlyTranslator(pl.LightningModule):
    """
    PyTorch Lightning module for Decoder-Only translation with LoRA
    Supports BLOOM, LLaMA, Phi, and other causal LM models
    """
    
    def __init__(
        self,
        model_name: str = "bigscience/bloom-560m",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        total_steps: Optional[int] = None,
        max_new_tokens: int = 128,
        use_4bit: bool = False,
        use_8bit: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Training params
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.max_new_tokens = max_new_tokens
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization if needed
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load base model
        logger.info(f"Loading base model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Prepare for k-bit training if using quantization
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        if lora_target_modules is None:
            # Default target modules for common architectures
            lora_target_modules = self._get_target_modules(model_name)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        logger.info("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Store LoRA config for saving
        self.lora_config = lora_config
        
    def _get_target_modules(self, model_name: str) -> List[str]:
        """Get target modules based on model architecture"""
        model_lower = model_name.lower()
        
        if "bloom" in model_lower:
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "llama" in model_lower or "mistral" in model_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "phi" in model_lower:
            return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif "gpt2" in model_lower:
            return ["c_attn", "c_proj", "c_fc"]
        elif "opt" in model_lower:
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        else:
            # Generic fallback
            return ["q_proj", "v_proj"]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log perplexity
        perplexity = torch.exp(loss)
        self.log("train_perplexity", perplexity, prog_bar=False, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step"""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        perplexity = torch.exp(loss)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True)
        
        return {"val_loss": loss}
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Only optimize LoRA parameters
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        if self.total_steps:
            warmup_steps = int(self.total_steps * self.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer
    
    def translate(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Translate a list of texts using instruction format"""
        self.model.eval()
        translations = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Format as instructions
                prompts = [
                    f"Translate English to Dutch:\nEnglish: {text}\nDutch:"
                    for text in batch_texts
                ]
                
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Generate
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=4,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode and extract translation
                for j, gen_ids in enumerate(generated_ids):
                    # Get only the generated part
                    input_length = inputs["input_ids"][j].shape[0]
                    generated_text = self.tokenizer.decode(
                        gen_ids[input_length:],
                        skip_special_tokens=True
                    ).strip()
                    
                    # Clean up - take first line/sentence
                    generated_text = generated_text.split("\n")[0].strip()
                    translations.append(generated_text)
        
        return translations


class DecoderOnlyTrainer:
    """
    Trainer class for Decoder-Only models with LoRA
    Handles training, validation, and model saving
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_config = config.decoder_only
        
        # Setup output directory
        self.output_dir = Path(self.model_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self) -> DecoderOnlyTranslator:
        """Run training pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Decoder-Only Fine-Tuning with LoRA")
        logger.info(f"Model: {self.model_config.model_name}")
        logger.info(f"LoRA r={self.model_config.lora_r}, alpha={self.model_config.lora_alpha}")
        logger.info("=" * 60)
        
        # Initialize model
        model = DecoderOnlyTranslator(
            model_name=self.model_config.model_name,
            lora_r=self.model_config.lora_r,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            lora_target_modules=self.model_config.lora_target_modules,
            learning_rate=self.model_config.learning_rate,
            warmup_ratio=self.model_config.warmup_ratio,
            weight_decay=self.model_config.weight_decay,
            max_new_tokens=self.model_config.max_new_tokens
        )
        
        # Create data loaders
        data_factory = DataLoaderFactory(self.config)
        dataloaders = data_factory.create_dataloaders(
            tokenizer=model.tokenizer,
            is_decoder_only=True
        )
        
        # Calculate total steps for scheduler
        total_steps = (
            len(dataloaders["train"]) * self.model_config.num_epochs
        ) // self.model_config.gradient_accumulation_steps
        model.total_steps = total_steps
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.output_dir),
            name="tensorboard_logs"
        )
        
        loggers = [tb_logger]
        
        if self.config.use_wandb:
            wandb_logger = WandbLogger(
                project=self.config.wandb_project,
                name=f"decoder_only_{self.model_config.model_save_name}"
            )
            loggers.append(wandb_logger)
        
        # Determine accelerator
        if self.config.device == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"
        else:
            accelerator = self.config.device
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=self.model_config.num_epochs,
            accelerator=accelerator,
            devices=1,
            accumulate_grad_batches=self.model_config.gradient_accumulation_steps,
            precision="16-mixed" if self.config.fp16 and accelerator == "cuda" else 32,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=self.config.logging_steps,
            val_check_interval=self.config.eval_steps,
            gradient_clip_val=1.0,
            deterministic=True
        )
        
        # Set seed
        pl.seed_everything(self.config.seed)
        
        # Train
        logger.info("Starting training...")
        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"]
        )
        
        # Save final model
        self._save_model(model)
        
        logger.info("=" * 60)
        logger.info("Decoder-Only Fine-Tuning with LoRA Complete!")
        logger.info(f"Model saved to: {self.output_dir / self.model_config.model_save_name}")
        logger.info("=" * 60)
        
        return model
    
    def _setup_callbacks(self) -> List[pl.Callback]:
        """Setup training callbacks"""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.output_dir / "checkpoints"),
            filename="decoder_only_lora-{epoch:02d}-{val_loss:.4f}",
            save_top_k=self.config.save_total_limit,
            monitor="val_loss",
            mode="min",
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
        # Progress bar
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
        
        return callbacks
    
    def _save_model(self, model: DecoderOnlyTranslator) -> None:
        """Save the fine-tuned LoRA model"""
        save_path = self.output_dir / self.model_config.model_save_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        model.model.save_pretrained(str(save_path))
        
        # Save tokenizer
        model.tokenizer.save_pretrained(str(save_path))
        
        # Save training config
        config_path = save_path / "training_config.json"
        import json
        with open(config_path, "w") as f:
            json.dump({
                "base_model": self.model_config.model_name,
                "lora_r": self.model_config.lora_r,
                "lora_alpha": self.model_config.lora_alpha,
                "lora_dropout": self.model_config.lora_dropout,
                "learning_rate": self.model_config.learning_rate,
                "batch_size": self.model_config.batch_size,
                "num_epochs": self.model_config.num_epochs
            }, f, indent=2)
        
        logger.info(f"LoRA adapters saved to {save_path}")


def load_lora_model(
    base_model_name: str,
    lora_path: str
) -> DecoderOnlyTranslator:
    """Load a trained LoRA model for inference"""
    
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    return model, tokenizer


def main():
    """Main entry point for decoder-only training"""
    config = get_config()
    trainer = DecoderOnlyTrainer(config)
    model = trainer.train()
    return model


if __name__ == "__main__":
    main()
