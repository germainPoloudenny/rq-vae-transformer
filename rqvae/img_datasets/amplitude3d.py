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
                 val_split: float = 0.1, *, pad_to_cube: bool = False,
                 cube_size: Optional[int] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.key = key
        self.max_index = max_index
        self.hkl_list = generate_hkl_list(max_index) if max_index is not None else None
        self.val_split = val_split
        self.pad_to_cube = pad_to_cube
        self.cube_size = cube_size

        self.h5_file = None
        if self.root.is_file():
            if self.root.suffix == '.npz':
                dataset = np.load(self.root, mmap_mode='r')[key]
            elif self.root.suffix in ['.h5', '.hdf5']:
                self.h5_file = h5py.File(self.root, 'r')
                dataset = self.h5_file[key]
            else:
                raise ValueError(f'Unsupported extension: {self.root.suffix}')

            total_len = len(dataset)
            split_idx = int(total_len * (1 - self.val_split))
            if split == 'train':
                self.start = 0
                self.end = split_idx
            elif split == 'val':
                self.start = split_idx
                self.end = total_len
            else:
                raise ValueError(f'Unknown split: {split}')
        else:
            npz_file = self.root / f"{split}.npz"
            h5_file = self.root / f"{split}.h5"

            if npz_file.exists():
                dataset = np.load(npz_file, mmap_mode='r')[key]
            elif h5_file.exists():
                self.h5_file = h5py.File(h5_file, 'r')
                dataset = self.h5_file[key]
            else:
                raise FileNotFoundError(
                    f"Amplitude file not found: {npz_file} or {h5_file}")
            self.start = 0
            self.end = len(dataset)

        if self.max_index is not None:
            self.end = min(self.start + self.max_index, self.end)

        self.data = dataset

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        amplitude = torch.from_numpy(self.data[self.start + idx]).float()

        side = 2 * self.max_index + 1
        num_slices = self.max_index + 1

        volume = amplitude.new_zeros(num_slices, side, side)
        for value, (h, k, l_idx) in zip(amplitude, self.hkl_list):
            volume[l_idx, h + self.max_index, k + self.max_index] = value

        amplitude = volume

        if self.pad_to_cube:
            # mirror positive l slices to negative indices and insert
            # the l=0 slice at the center to build a full cube
            positive = amplitude[1:]
            amplitude = torch.cat([positive.flip(0), amplitude[:1], positive], dim=0)

            cube_side = amplitude.shape[-1]
            cube_size = self.cube_size if self.cube_size is not None else cube_side
            if cube_size < cube_side:
                raise ValueError('cube_size must be >= %d' % cube_side)
            if cube_size != cube_side:
                cube = amplitude.new_zeros(cube_size, cube_size, cube_size)
                start = (cube_size - cube_side) // 2
                cube[start:start+cube_side,
                     start:start+cube_side,
                     start:start+cube_side] = amplitude
                amplitude = cube

        amplitude = amplitude.unsqueeze(0)

        if self.transform:
            amplitude = self.transform(amplitude)

        return amplitude, 0

    def __del__(self):
        if getattr(self, 'h5_file', None) is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
