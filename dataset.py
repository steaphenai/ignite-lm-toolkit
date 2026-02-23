import torch
from torch.utils.data import Dataset, DataLoader
import requests


def load_text(url: str) -> str:
    """Download TinyShakespeare dataset."""
    response = requests.get(url)
    return response.text


def build_vocab(text: str) -> tuple[dict, dict]:
    """Build character-level vocabulary."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


class TextDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, text: str, char_to_idx: dict, block_size: int):
        self.block_size = block_size
        self.char_to_idx = char_to_idx
        self.data = torch.tensor(
            [char_to_idx[ch] for ch in text], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_dataloaders(
    url: str,
    block_size: int,
    batch_size: int,
    train_split: float,
) -> tuple[DataLoader, DataLoader, dict, dict]:
    """Create train and val dataloaders."""
    text = load_text(url)
    char_to_idx, idx_to_char = build_vocab(text)

    n = int(train_split * len(text))
    train_dataset = TextDataset(text[:n], char_to_idx, block_size)
    val_dataset = TextDataset(text[n:], char_to_idx, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, char_to_idx, idx_to_char