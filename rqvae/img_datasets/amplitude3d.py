import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple
import h5py


def generate_hkl_list(hkl_max_index: int) -> List[Tuple[int, int, int]]:
    """Return the list of (h, k, l) coordinates for a given ``hkl_max_index``.

    The ordering is identical to the layout expected by ``Amplitude3D`` when a
    flat array is reshaped into a volume. The central (0, 0, 0) element is
    omitted from the returned list.
    """

    return [
        (h, k, l_idx)
        for h in range(-hkl_max_index, hkl_max_index + 1)
        for k in range(-hkl_max_index, hkl_max_index + 1)
        for l_idx in range(0, hkl_max_index + 1)
        if not (h == 0 and k == 0 and l_idx == 0)
    ]


class Amplitude3D(Dataset):
    """Dataset loading 3D amplitude tensors from ``npz`` or ``h5`` files."""

    def __init__(self, root: str, split: str = "train", transform=None,
                 key: str = "amplitudes", max_index: Optional[int] = None,
                 val_split: float = 0.1):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.key = key
        self.max_index = max_index
        self.hkl_list = generate_hkl_list(max_index) if max_index is not None else None
        self.val_split = val_split

        if self.root.is_file():
            if self.root.suffix == '.npz':
                data_all = np.load(self.root)[key]
            elif self.root.suffix in ['.h5', '.hdf5']:
                with h5py.File(self.root, 'r') as f:
                    data_all = f[key][:]
            else:
                raise ValueError(f'Unsupported extension: {self.root.suffix}')

            split_idx = int(len(data_all) * (1 - self.val_split))
            if split == 'train':
                data = data_all[:split_idx]
            elif split == 'val':
                data = data_all[split_idx:]
            else:
                raise ValueError(f'Unknown split: {split}')
        else:
            npz_file = self.root / f"{split}.npz"
            h5_file = self.root / f"{split}.h5"

            if npz_file.exists():
                data = np.load(npz_file)[key]
            elif h5_file.exists():
                with h5py.File(h5_file, 'r') as f:
                    data = f[key][:]
            else:
                raise FileNotFoundError(
                    f"Amplitude file not found: {npz_file} or {h5_file}")
        if self.max_index is not None:
            data = data[: self.max_index]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = torch.from_numpy(self.data[idx]).float()

        if amplitude.dim() == 1 and self.max_index is not None:
            side = 2 * self.max_index + 1
            num_slices = self.max_index + 1

            volume = amplitude.new_zeros(num_slices, side, side)
            for value, (h, k, l_idx) in zip(amplitude, self.hkl_list):
                volume[l_idx, h + self.max_index, k + self.max_index] = value

            amplitude = volume

        if amplitude.dim() == 3:
            # convert to (C, D, H, W) for ``Conv3d`` with a single channel
            amplitude = amplitude.unsqueeze(0)

        if self.transform:
            amplitude = self.transform(amplitude)

        return amplitude, 0
