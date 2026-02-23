import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class Perplexity(Metric):
    """
    Perplexity metric for language models.

    Perplexity = exp(average cross-entropy loss)

    This metric is missing from pytorch-ignite and is essential
    for evaluating language models.

    Args:
        output_transform: a callable to transform engine.state.output
            to (y_pred, y) form.

    Example:
        .. code-block:: python

            perplexity = Perplexity()
            perplexity.attach(evaluator, "perplexity")
    """

    def __init__(self, output_transform=lambda x: x):
        super(Perplexity, self).__init__(output_transform=output_transform)

    def reset(self) -> None:
        self._sum_loss = 0
        self._num_examples = 0

    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.dim() == 3:
            # (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            y_pred = y_pred.view(-1, y_pred.size(-1))
            y = y.view(-1)

        loss = torch.nn.functional.cross_entropy(y_pred, y)
        self._sum_loss += loss.item() * y.size(0)
        self._num_examples += y.size(0)

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError(
                "Perplexity must have at least one example before it can be computed."
            )
        avg_loss = self._sum_loss / self._num_examples
        return torch.exp(torch.tensor(avg_loss)).item()