import yaml
import torch
from dataset import create_dataloaders
from model import GPT
from trainer import create_trainer, create_evaluator, attach_checkpoint, attach_logging
import os


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config("config.yaml")

    device = config["trainer"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Load data
    print("Loading dataset...")
    train_loader, val_loader, char_to_idx, idx_to_char = create_dataloaders(
        url=config["dataset"]["url"],
        block_size=config["dataset"]["block_size"],
        batch_size=config["dataset"]["batch_size"],
        train_split=config["dataset"]["train_split"],
    )

    vocab_size = len(char_to_idx)
    print(f"Vocab size: {vocab_size}")

    # Build model
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        block_size=config["model"]["block_size"],
        dropout=config["model"]["dropout"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["trainer"]["learning_rate"]
    )

    # Create engines
    trainer = create_trainer(model, optimizer, device)
    evaluator = create_evaluator(model, device)

    # Attach handlers
    os.makedirs(config["trainer"]["checkpoint_dir"], exist_ok=True)
    attach_checkpoint(
        trainer, evaluator, model, optimizer,
        config["trainer"]["checkpoint_dir"]
    )
    attach_logging(
        trainer, evaluator,
        train_loader, val_loader,
        config["trainer"]["log_every"]
    )

    # Generate sample before training
    print("\nSample before training:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=100)
    print("".join([idx_to_char[i.item()] for i in generated[0]]))

    # Train
    print("\nStarting training...")
    trainer.run(train_loader, max_epochs=config["trainer"]["epochs"])

    # Generate sample after training
    print("\nSample after training:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200)
    print("".join([idx_to_char[i.item()] for i in generated[0]]))


if __name__ == "__main__":
    main()