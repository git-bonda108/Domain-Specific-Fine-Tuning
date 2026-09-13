"""
Encoder-Decoder Fine-Tuning Pipeline
Using MarianMT (Helsinki-NLP/opus-mt-en-nl) with PyTorch Lightning

Track 1: Software domain-specific fine-tuning for EN→NL translation
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
    MarianMTModel,
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from config import TrainingConfig, get_config
from data_loader import DataLoaderFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderDecoderTranslator(pl.LightningModule):
    """
    PyTorch Lightning module for Encoder-Decoder translation model
    Supports MarianMT, mBART, and other seq2seq models
    """
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-nl",
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        total_steps: Optional[int] = None,
        max_length: int = 128,
        num_beams: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
        # Training params
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        
        # Generation params
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Metrics storage
        self.validation_outputs = []
        
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
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step"""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Generate translations for quality assessment
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=True
        )
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return {
            "val_loss": loss,
            "generated_ids": generated_ids,
            "labels": batch["labels"]
        }
    
    def on_validation_epoch_end(self) -> None:
        """Compute metrics at end of validation epoch"""
        pass  # Metrics computed in evaluation script
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8
        )
        
        if self.total_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
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
    
    def translate(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """Translate a list of texts"""
        self.model.eval()
        translations = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
                
                batch_translations = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                translations.extend(batch_translations)
        
        return translations


class EncoderDecoderTrainer:
    """
    Trainer class for Encoder-Decoder models
    Handles training, validation, and model saving
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_config = config.encoder_decoder
        
        # Setup output directory
        self.output_dir = Path(self.model_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self) -> EncoderDecoderTranslator:
        """Run training pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Encoder-Decoder Fine-Tuning")
        logger.info(f"Model: {self.model_config.model_name}")
        logger.info("=" * 60)
        
        # Initialize model
        model = EncoderDecoderTranslator(
            model_name=self.model_config.model_name,
            learning_rate=self.model_config.learning_rate,
            warmup_steps=self.model_config.warmup_steps,
            weight_decay=self.model_config.weight_decay,
            max_length=self.model_config.max_length,
            num_beams=self.model_config.num_beams
        )
        
        # Create data loaders
        data_factory = DataLoaderFactory(self.config)
        dataloaders = data_factory.create_dataloaders(
            tokenizer=model.tokenizer,
            is_decoder_only=False
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
                name=f"encoder_decoder_{self.model_config.model_save_name}"
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
        logger.info("Encoder-Decoder Fine-Tuning Complete!")
        logger.info(f"Model saved to: {self.output_dir / self.model_config.model_save_name}")
        logger.info("=" * 60)
        
        return model
    
    def _setup_callbacks(self) -> List[pl.Callback]:
        """Setup training callbacks"""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.output_dir / "checkpoints"),
            filename="encoder_decoder-{epoch:02d}-{val_loss:.4f}",
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
    
    def _save_model(self, model: EncoderDecoderTranslator) -> None:
        """Save the fine-tuned model"""
        save_path = self.output_dir / self.model_config.model_save_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model.model.save_pretrained(str(save_path))
        model.tokenizer.save_pretrained(str(save_path))
        
        # Save training config
        config_path = save_path / "training_config.json"
        import json
        with open(config_path, "w") as f:
            json.dump({
                "model_name": self.model_config.model_name,
                "learning_rate": self.model_config.learning_rate,
                "batch_size": self.model_config.batch_size,
                "num_epochs": self.model_config.num_epochs,
                "max_length": self.model_config.max_length
            }, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")


def main():
    """Main entry point for encoder-decoder training"""
    config = get_config()
    trainer = EncoderDecoderTrainer(config)
    model = trainer.train()
    return model


if __name__ == "__main__":
    main()
