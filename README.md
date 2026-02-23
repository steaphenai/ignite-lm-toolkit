# ignite-lm-toolkit

A language model training toolkit built with PyTorch-Ignite.

## What is this?

A prototype toolkit for training GPT-style language models using 
pytorch-ignite. The goal is to demonstrate best practices for LM training with ignite.

## Key Features

- GPT-style transformer implementation
- **Perplexity metric** — missing from pytorch-ignite, implemented here
- Character-level text generation
- Model checkpointing based on best perplexity
- YAML-based configuration

- `Engine` — training and evaluation loops
- `Metric` — custom Perplexity metric
- `ModelCheckpoint` — saves best model based on Perplexity
- `ProgressBar` — training progress tracking
- `Events` — epoch-level logging


## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Results

Trained on TinyShakespeare dataset. Sample generated text after training:
```
Admit your? Therefore are the glad with Marclus.
Now is the provost, just at my soul hath denied
Intended be in the God's hand; but heaven way,
```

## Built With

- PyTorch
- PyTorch-Ignite
- TinyShakespeare dataset (Andrej Karpathy)