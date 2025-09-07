"""Dataset and data loading utilities for GPT training."""

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Dataset for autoregressive language modeling."""
    
    def __init__(self, text, tokenizer, block_size=128):
        """
        Initialize dataset with tokenized text.
        
        Args:
            text: Input text corpus
            tokenizer: Tokenizer instance
            block_size: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)
        print(f"Dataset size: {len(self.tokens)} tokens")
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        # Get input and target sequences
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_dataloaders(text, tokenizer, block_size=128, batch_size=64, 
                       train_split=0.9, num_workers=0):
    """
    Create training and validation dataloaders.
    
    Args:
        text: Input text corpus
        tokenizer: Tokenizer instance
        block_size: Maximum sequence length
        batch_size: Batch size for training
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = TextDataset(text, tokenizer, block_size)
    
    # Split into train and validation
    n = len(dataset)
    n_train = int(train_split * n)
    n_val = n - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader