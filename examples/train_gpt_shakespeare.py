"""Train GPT on Shakespeare dataset (similar to nanoGPT)."""

import os
import torch
import requests
from llm_from_scratch.gpt import (
    GPT, GPTConfig, SimpleTokenizer,
    create_dataloaders, GPTTrainer
)


def download_shakespeare():
    """Download Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    # Create data directory
    os.makedirs("data/shakespeare", exist_ok=True)
    
    # Download if not exists
    filepath = "data/shakespeare/input.txt"
    if not os.path.exists(filepath):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {filepath}")
    
    # Read the text
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text)} characters")
    return text


def train_shakespeare_gpt():
    """Train GPT on Shakespeare text (character-level)."""
    
    # Download and load dataset
    text = download_shakespeare()
    
    # Create character-level tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        text, tokenizer,
        block_size=256,      # Context length
        batch_size=64,       # Batch size
        train_split=0.9
    )
    
    # Model configuration (similar to nanoGPT's shakespeare config)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=384,          # Embedding dimension
        n_layer=6,           # Number of layers
        n_head=6,            # Number of attention heads
        block_size=256,      # Context window
        dropout=0.2
    )
    
    print(f"Model size: ~{config.get_model_size():.2f}M parameters")
    
    # Create model
    model = GPT(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout
    )
    
    # Create trainer
    trainer = GPTTrainer(
        model, train_loader, val_loader,
        learning_rate=1e-3,
        weight_decay=0.1,
        warmup_steps=100,
        max_steps=5000,      # Train for 5000 steps
        grad_clip=1.0
    )
    
    # Train model
    print("\nStarting training...")
    losses = trainer.train(log_interval=100, eval_interval=500)
    
    # Generate sample text
    print("\n" + "="*50)
    print("Generating Shakespeare-style text...")
    print("="*50)
    
    model.eval()
    
    # Start with a prompt
    prompts = [
        "ROMEO:",
        "To be or not to be",
        "All the world's a stage",
        "HAMLET:"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 30)
        
        # Encode prompt
        prompt_tokens = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(trainer.device)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                prompt_tokens,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40
            )
        
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        print(generated_text)
    
    # Save model
    checkpoint_path = "shakespeare_gpt_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")
    
    return model, tokenizer, losses


if __name__ == "__main__":
    # Train on Shakespeare
    model, tokenizer, losses = train_shakespeare_gpt()