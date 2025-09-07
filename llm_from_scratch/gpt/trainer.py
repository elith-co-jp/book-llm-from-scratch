"""Training utilities for GPT model."""

import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class GPTTrainer:
    """Trainer class for GPT model."""
    
    def __init__(self, model, train_loader, val_loader, 
                 learning_rate=3e-4, weight_decay=0.1, 
                 warmup_steps=1000, max_steps=10000,
                 grad_clip=1.0, device=None):
        """
        Initialize trainer with model and data loaders.
        
        Args:
            model: GPT model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            grad_clip: Gradient clipping value
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self.configure_optimizer(learning_rate, weight_decay)
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=max_steps - warmup_steps
        )
        
    def configure_optimizer(self, learning_rate, weight_decay):
        """Configure AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_groups, lr=learning_rate, betas=(0.9, 0.95))
        return optimizer
    
    def get_lr(self, step):
        """Calculate learning rate with warmup."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.optimizer.param_groups[0]['lr'] * step / self.warmup_steps
        else:
            # Use scheduler
            return self.optimizer.param_groups[0]['lr']
    
    @torch.no_grad()
    def evaluate(self, max_batches=10):
        """Evaluate model on validation set."""
        self.model.eval()
        losses = []
        
        for i, (x, y) in enumerate(self.val_loader):
            if i >= max_batches:
                break
                
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses) if losses else float('inf')
    
    def train_step(self, x, y):
        """Single training step."""
        # Forward pass
        logits, loss = self.model(x, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, log_interval=100, eval_interval=500):
        """
        Main training loop.
        
        Args:
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            
        Returns:
            Dictionary with training and validation losses
        """
        self.model.train()
        
        train_losses = []
        val_losses = []
        
        step = 0
        epoch = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
        start_time = time.time()
        
        while step < self.max_steps:
            epoch += 1
            
            for x, y in self.train_loader:
                if step >= self.max_steps:
                    break
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Training step
                loss = self.train_step(x, y)
                train_losses.append(loss)
                
                # Update learning rate
                if step >= self.warmup_steps:
                    self.scheduler.step()
                
                # Logging
                if step % log_interval == 0:
                    avg_loss = np.mean(train_losses[-log_interval:]) if len(train_losses) >= log_interval else loss
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{self.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {self.get_lr(step):.6f} | "
                          f"Time: {elapsed:.1f}s")
                
                # Evaluation
                if step % eval_interval == 0 and step > 0:
                    val_loss = self.evaluate()
                    val_losses.append(val_loss)
                    print(f"Validation loss: {val_loss:.4f}")
                
                step += 1
        
        print(f"Training completed! Total time: {time.time() - start_time:.1f}s")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded from {path}")