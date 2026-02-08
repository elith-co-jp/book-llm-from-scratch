# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Japanese educational repository for learning Large Language Models (LLMs) from scratch. It implements Transformer architecture and GPT models step-by-step, with accompanying documentation in `docs/` and interactive notebooks in `notebooks/`.

## Development Commands

### Environment Setup
```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
```

### Running Examples
```bash
# Run the example training script (trains GPT on Natsume Soseki corpus)
uv run python examples/train_gpt_soseki.py
```

### Testing
```bash
# Run tests with pytest
uv run pytest

# Run specific test file
uv run pytest tests/test_file.py
```

### Code Quality
```bash
# Format code with black
uv run black llm_from_scratch/

# Sort imports with isort
uv run isort llm_from_scratch/

# Lint with flake8
uv run flake8 llm_from_scratch/
```

### Working with Notebooks
- Jupyter notebooks are located in `notebooks/chapter02/` and `notebooks/chapter04/`
- Start Jupyter with: `uv run jupyter notebook`
- Chapter 4 notebooks may use local data from `notebooks/chapter04/data/`

## Architecture

### Module Structure

The codebase is organized into two main components:

**1. `llm_from_scratch.transformer`** - Base Transformer implementation
- `attention.py`: Multi-head attention mechanism
- `transformer.py`: Full encoder-decoder Transformer with `Encoder`, `Decoder`, `EncoderBlock`, `DecoderBlock` classes
- `utils.py`: LayerNorm, PositionalEncoding utilities

**2. `llm_from_scratch.gpt`** - GPT (decoder-only) implementation
- `model.py`: GPT model with `GPTMultiHeadAttention`, `TransformerBlock`, `GPT`, `GPTConfig`
- `tokenizer.py`: `SimpleTokenizer` for character-level tokenization
- `dataset.py`: `TextDataset` and `create_dataloaders()` for text data handling
- `trainer.py`: `GPTTrainer` with learning rate scheduling, gradient clipping, checkpointing

**3. `llm_from_scratch.data`** - Data preprocessing utilities
- `preprocess.py`: Dataset preprocessing functions

### Key Design Patterns

**GPT vs Transformer**: The GPT implementation reuses the base `MultiHeadAttention` from the transformer module but wraps it in `GPTMultiHeadAttention` which adds causal masking for autoregressive generation. GPT is decoder-only (no encoder).

**Model Configuration**: Uses `GPTConfig` dataclass pattern for model hyperparameters (vocab_size, n_embd, n_layer, n_head, block_size, dropout).

**Training Pipeline**:
1. Download/load text corpus (e.g., from Aozora Bunko for Japanese texts)
2. Create tokenizer (character-level with `SimpleTokenizer`)
3. Create dataloaders with `create_dataloaders(text, tokenizer, block_size, batch_size, train_split)`
4. Initialize model with `GPTConfig`
5. Train with `GPTTrainer` which handles optimizer, scheduler, checkpointing
6. Generate text with `model.generate(prompt_tokens, max_new_tokens, temperature, top_k)`

**Chapter 4 Focus**: The chapter04 notebooks demonstrate:
- `section01`: Dataset preprocessing pipeline (normalization, deduplication, filtering, splitting)
- `section02`: Distributed training (data parallel, tensor parallel)
- `section03`: Pre-training GPT-2 on Aozora Bunko corpus using HuggingFace libraries

### Data Flow

Text → Tokenizer (character-level) → TextDataset → DataLoader → GPT Model → Training → Checkpoints (`.pt` files in `models/`)

### Important Notes

- The repository uses **uv** for package management (not pip or poetry directly)
- Python version: `>=3.12,<3.13` (specified in pyproject.toml)
- All documentation is in **Japanese**
- Model checkpoints and data files are gitignored (`data/`, `models/`, `*.pt`, `*.pth`)
- The codebase demonstrates educational implementations alongside practical HuggingFace-based examples in Chapter 4

### Notebook Execution Context

When working with notebooks in `notebooks/chapter04/`:
- They may reference data saved to `data/` or local `notebooks/chapter04/data/`
- Chapter 4.3 notebooks demonstrate HuggingFace Transformers integration for real-world pre-training
- Use `section02_data_parallel.py` and `section02_tensor_parallel.py` for distributed training examples
