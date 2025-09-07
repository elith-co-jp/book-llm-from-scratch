"""Example script for training a small GPT model."""

import torch
from llm_from_scratch.gpt import (
    GPT, GPTConfig, SimpleTokenizer, 
    create_dataloaders, GPTTrainer
)


def train_small_gpt():
    """Train a small GPT model on sample text."""
    
    # Sample text data
    text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Deep learning models can learn complex patterns from data.
    Natural language processing enables computers to understand human language.
    Transformers have revolutionized the field of NLP.
    GPT models are powerful language models based on the transformer architecture.
    Training neural networks requires careful tuning of hyperparameters.
    The attention mechanism allows models to focus on relevant parts of the input.
    """ * 100  # Repeat for more training data
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        text, tokenizer, 
        block_size=64,
        batch_size=8,
        train_split=0.9
    )
    
    # Model configuration (small model for demonstration)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=128,      # Embedding dimension
        n_layer=4,       # Number of layers
        n_head=4,        # Number of attention heads
        block_size=64,   # Context length
        dropout=0.1
    )
    
    # Create model
    model = GPT(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout
    )
    
    print(f"Model size: ~{config.get_model_size():.2f}M parameters")
    
    # Create trainer
    trainer = GPTTrainer(
        model, train_loader, val_loader,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=1000,
        grad_clip=1.0
    )
    
    # Train model
    print("\nStarting training...")
    losses = trainer.train(log_interval=50, eval_interval=200)
    
    # Generate sample text
    print("\nGenerating sample text...")
    model.eval()
    
    # Start with a prompt
    prompt = "The quick"
    prompt_tokens = torch.tensor(
        tokenizer.encode(prompt), 
        dtype=torch.long
    ).unsqueeze(0).to(trainer.device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            prompt_tokens, 
            max_new_tokens=100,
            temperature=0.8,
            top_k=40
        )
    
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Save model checkpoint
    checkpoint_path = "gpt_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")
    
    return model, tokenizer, losses


if __name__ == "__main__":
    model, tokenizer, losses = train_small_gpt()