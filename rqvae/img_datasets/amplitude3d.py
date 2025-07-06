import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

class Amplitude3D(Dataset):
    """Dataset loading 3D amplitude tensors from npz files."""

    def __init__(self, root: str, split: str = "train", transform=None,
                 key: str = "amplitudes", max_index: Optional[int] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.key = key
        self.max_index = max_index

        file = self.root / f"{split}.npz"
        if not file.exists():
            raise FileNotFoundError(f"Amplitude file not found: {file}")

        data = np.load(file)[key]
        if self.max_index is not None:
            data = data[: self.max_index]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        if self.transform:
            sample = self.transform(sample)
        return sample, 0
