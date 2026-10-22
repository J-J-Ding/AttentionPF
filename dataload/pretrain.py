"""Pure temporal patching and masking utilities for V1 pretraining."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from dataload.dataloader import DataBundle
from utils.seed import seed_worker


def _validate_patch_geometry(length: int, patch_length: int, patch_stride: int) -> int:
    if patch_length <= 0 or patch_stride <= 0:
        raise ValueError("patch_length and patch_stride must be positive")
    if patch_length > length:
        raise ValueError(f"patch_length {patch_length} exceeds sequence length {length}")
    remainder = (length - patch_length) % patch_stride
    if remainder:
        raise ValueError(
            f"patches do not exactly cover sequence length {length}: remainder={remainder}"
        )
    return 1 + (length - patch_length) // patch_stride


def patchify(x: Tensor, patch_length: int, patch_stride: int) -> Tensor:
    """Convert ``[B,L,C]`` into channel-major flattened patches ``[B,N,C*S]``."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B,L,C], got {tuple(x.shape)}")
    _validate_patch_geometry(x.shape[1], patch_length, patch_stride)
    patches = x.unfold(dimension=1, size=patch_length, step=patch_stride)
    return patches.contiguous().flatten(start_dim=2)


def _masked_count(n_patches: int, ratio: float) -> int:
    if n_patches < 2:
        raise ValueError("n_patches must be at least 2 to keep masked and visible patches")
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError(f"mask ratio must be in (0, 1), got {ratio}")
    return min(n_patches - 1, max(1, int(round(n_patches * float(ratio)))))


def sample_random_mask(
    batch_size: int,
    n_patches: int,
    ratio: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Sample a reproducible, fixed-cardinality boolean mask per sample."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    count = _masked_count(n_patches, ratio)
    scores = torch.rand((batch_size, n_patches), generator=generator, device="cpu")
    indices = scores.topk(count, dim=1, largest=False).indices
    mask = torch.zeros((batch_size, n_patches), dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.to(device=device)


def sample_tail_mask(
    batch_size: int,
    n_patches: int,
    ratio: float,
    *,
    device: torch.device,
) -> Tensor:
    """Mask one contiguous suffix while retaining at least one visible patch."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    count = _masked_count(n_patches, ratio)
    mask = torch.zeros((batch_size, n_patches), dtype=torch.bool, device=device)
    mask[:, -count:] = True
    return mask


def choose_mask(
    *,
    batch_size: int,
    n_patches: int,
    random_ratio: float,
    tail_ratio: float,
    random_probability: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, str]:
    """Choose Random Mask or Tail Mask once for a complete batch."""
    if not 0.0 <= float(random_probability) <= 1.0:
        raise ValueError(
            f"random_probability must be in [0, 1], got {random_probability}"
        )
    use_random = bool(
        torch.rand((), generator=generator, device="cpu").item() < random_probability
    )
    if use_random:
        return (
            sample_random_mask(
                batch_size,
                n_patches,
                random_ratio,
                generator=generator,
                device=device,
            ),
            "random",
        )
    return sample_tail_mask(batch_size, n_patches, tail_ratio, device=device), "tail"


def create_pretrain_loaders(
    bundle: DataBundle, config: dict
) -> tuple[DataLoader, DataLoader]:
    """Chronologically split only the official train-window dataset."""
    dataset = bundle.train_loader.dataset
    count = len(dataset)
    if count < 2:
        raise ValueError("pretraining requires at least two official train windows")
    split = int(count * (1.0 - float(config["pretrain"]["val_ratio"])))
    split = min(count - 1, max(1, split))
    train_dataset = Subset(dataset, range(0, split))
    val_dataset = Subset(dataset, range(split, count))
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    options = {
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": bool(config["data"]["pin_memory"]),
        "worker_init_fn": seed_worker,
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **options),
        DataLoader(val_dataset, shuffle=False, **options),
    )
