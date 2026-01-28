"""
Data Loading and Processing Module
Handles WMT16, FLORES, and custom software domain test set
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationDataset(Dataset):
    """Custom dataset for EN→NL translation"""
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer: AutoTokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
        is_decoder_only: bool = False
    ):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_decoder_only = is_decoder_only
        
    def __len__(self) -> int:
        return len(self.sources)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source = self.sources[idx]
        target = self.targets[idx]
        
        if self.is_decoder_only:
            # For decoder-only: format as instruction
            prompt = f"Translate English to Dutch:\nEnglish: {source}\nDutch: {target}"
            
            encoding = self.tokenizer(
                prompt,
                max_length=self.max_source_length + self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create labels (shift for causal LM)
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            
            # Labels are input_ids for causal LM
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # For encoder-decoder (seq2seq)
            source_encoding = self.tokenizer(
                source,
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            target_encoding = self.tokenizer(
                text_target=target,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            labels = target_encoding["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": source_encoding["input_ids"].squeeze(),
                "attention_mask": source_encoding["attention_mask"].squeeze(),
                "labels": labels
            }


class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    def __init__(self, config):
        self.config = config
        
    def load_wmt16_data(self, num_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Load WMT16 data for EN→NL training
        Using available EN-DE and extracting software domain patterns
        """
        logger.info("Loading WMT16 training data...")
        
        try:
            # Try loading WMT16 EN-NL if available
            dataset = load_dataset(
                "wmt16", 
                "nl-en",
                split="train",
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"WMT16 nl-en not available: {e}")
            logger.info("Using alternative dataset: OPUS-100 en-nl")
            try:
                dataset = load_dataset(
                    "opus100",
                    "en-nl",
                    split="train",
                    trust_remote_code=True
                )
            except Exception as e2:
                logger.warning(f"OPUS-100 not available: {e2}")
                logger.info("Using CCMatrix for Dutch training data")
                # Fallback to a simpler approach - create synthetic data
                return self._create_fallback_training_data()
        
        sources = []
        targets = []
        
        for item in dataset:
            if "translation" in item:
                sources.append(item["translation"]["en"])
                targets.append(item["translation"]["nl"])
            elif "en" in item and "nl" in item:
                sources.append(item["en"])
                targets.append(item["nl"])
        
        if num_samples:
            sources = sources[:num_samples]
            targets = targets[:num_samples]
            
        logger.info(f"Loaded {len(sources)} training samples")
        return sources, targets
    
    def _create_fallback_training_data(self) -> Tuple[List[str], List[str]]:
        """Create software domain training data from available sources"""
        logger.info("Creating software domain training data...")
        
        # Software domain parallel sentences for fine-tuning
        # These are representative software/tech domain translations
        software_pairs = [
            ("Click the button to continue", "Klik op de knop om door te gaan"),
            ("Download the latest version", "Download de nieuwste versie"),
            ("Install the software", "Installeer de software"),
            ("Update your settings", "Werk je instellingen bij"),
            ("Enter your password", "Voer je wachtwoord in"),
            ("The file has been saved", "Het bestand is opgeslagen"),
            ("Connection failed", "Verbinding mislukt"),
            ("Loading...", "Laden..."),
            ("Please wait", "Even geduld"),
            ("Error occurred", "Er is een fout opgetreden"),
            ("Memory: 256GB storage", "Geheugen: 256GB opslag"),
            ("Battery life: 24 hours", "Batterijduur: 24 uur"),
            ("Screen resolution: 1080p", "Schermresolutie: 1080p"),
            ("Wireless charging supported", "Draadloos opladen ondersteund"),
            ("5G connectivity", "5G-connectiviteit"),
            ("Camera: 48MP main sensor", "Camera: 48MP hoofdsensor"),
            ("Operating system: Android 14", "Besturingssysteem: Android 14"),
            ("Processor: Snapdragon 8 Gen 3", "Processor: Snapdragon 8 Gen 3"),
            ("RAM: 12GB", "RAM: 12GB"),
            ("Display: 6.7 inch AMOLED", "Scherm: 6,7 inch AMOLED"),
        ]
        
        # Expand with variations
        sources = [pair[0] for pair in software_pairs]
        targets = [pair[1] for pair in software_pairs]
        
        # Try loading OPUS-MT training data
        try:
            opus = load_dataset("opus_books", "en-nl", split="train[:10000]")
            for item in opus:
                sources.append(item["translation"]["en"])
                targets.append(item["translation"]["nl"])
        except:
            pass
            
        logger.info(f"Created {len(sources)} training samples")
        return sources, targets
    
    def load_software_test_set(self) -> Tuple[List[str], List[str]]:
        """Load the software domain test set from Excel"""
        logger.info(f"Loading software test set from {self.config.data.software_test_path}")
        
        df = pd.read_excel(self.config.data.software_test_path)
        
        sources = df["English Source"].tolist()
        targets = df["Reference Translation"].tolist()
        
        # Clean data
        sources = [str(s).strip() for s in sources if pd.notna(s)]
        targets = [str(t).strip() for t in targets if pd.notna(t)]
        
        logger.info(f"Loaded {len(sources)} software domain test samples")
        return sources, targets
    
    def load_flores_devtest(self) -> Tuple[List[str], List[str]]:
        """Load FLORES devtest for general domain evaluation"""
        logger.info("Loading FLORES devtest dataset...")
        
        try:
            # Load FLORES-200
            dataset = load_dataset(
                "facebook/flores",
                "eng_Latn-nld_Latn",
                split="devtest",
                trust_remote_code=True
            )
            
            sources = []
            targets = []
            
            for item in dataset:
                sources.append(item["sentence_eng_Latn"])
                targets.append(item["sentence_nld_Latn"])
                
        except Exception as e:
            logger.warning(f"FLORES loading failed: {e}")
            logger.info("Trying alternative FLORES loading...")
            
            try:
                # Alternative: flores101
                eng_dataset = load_dataset("gsarti/flores_101", "eng", split="devtest")
                nld_dataset = load_dataset("gsarti/flores_101", "nld", split="devtest")
                
                sources = eng_dataset["sentence"]
                targets = nld_dataset["sentence"]
            except Exception as e2:
                logger.error(f"Could not load FLORES: {e2}")
                # Return empty for now, will be handled in evaluation
                return [], []
        
        logger.info(f"Loaded {len(sources)} FLORES devtest samples")
        return sources, targets
    
    def create_dataloaders(
        self,
        tokenizer: AutoTokenizer,
        is_decoder_only: bool = False
    ) -> Dict[str, DataLoader]:
        """Create all necessary dataloaders"""
        
        # Load training data
        train_sources, train_targets = self.load_wmt16_data(
            num_samples=self.config.data.train_size
        )
        
        # Split into train/val
        val_size = min(self.config.data.val_size, len(train_sources) // 10)
        
        train_dataset = TranslationDataset(
            sources=train_sources[val_size:],
            targets=train_targets[val_size:],
            tokenizer=tokenizer,
            max_source_length=self.config.data.max_source_length,
            max_target_length=self.config.data.max_target_length,
            is_decoder_only=is_decoder_only
        )
        
        val_dataset = TranslationDataset(
            sources=train_sources[:val_size],
            targets=train_targets[:val_size],
            tokenizer=tokenizer,
            max_source_length=self.config.data.max_source_length,
            max_target_length=self.config.data.max_target_length,
            is_decoder_only=is_decoder_only
        )
        
        # Batch sizes
        if is_decoder_only:
            batch_size = self.config.decoder_only.batch_size
        else:
            batch_size = self.config.encoder_decoder.batch_size
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return {
            "train": train_loader,
            "val": val_loader
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
