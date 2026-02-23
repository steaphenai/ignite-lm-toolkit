import torch
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from metrics import Perplexity


def create_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Engine:
    """Create ignite trainer engine for LM training."""

    def train_step(engine: Engine, batch: tuple) -> dict:
        model.train()
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        # Reshape for cross entropy
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return {"loss": loss.item()}

    trainer = Engine(train_step)

    # Attach progress bar
    ProgressBar().attach(trainer, output_transform=lambda x: x)

    return trainer


def create_evaluator(
    model: torch.nn.Module,
    device: str,
) -> Engine:
    """Create ignite evaluator engine with Perplexity metric."""

    def eval_step(engine: Engine, batch: tuple) -> tuple:
        model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            return logits, y

    evaluator = Engine(eval_step)

    # Attach metrics
    Perplexity().attach(evaluator, "perplexity")

    return evaluator


def attach_checkpoint(
    trainer: Engine,
    evaluator: Engine,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
) -> None:
    """Save best model checkpoint based on perplexity."""

    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename_prefix="best",
        n_saved=2,
        score_function=lambda e: -e.state.metrics["perplexity"],
        score_name="perplexity",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        checkpoint,
        {"model": model, "optimizer": optimizer},
    )


def attach_logging(
    trainer: Engine,
    evaluator: Engine,
    train_loader,
    val_loader,
    log_every: int,
) -> None:
    """Log metrics every epoch."""

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine: Engine) -> None:
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            f"Epoch {engine.state.epoch} "
            f"| Loss: {engine.state.output['loss']:.4f} "
            f"| Perplexity: {metrics['perplexity']:.2f}"
        )