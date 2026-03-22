from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from .transformer import Batch
except ImportError:  # script-mode fallback
    from transformer import Batch  # type: ignore


class TextDataset(Dataset):
    """
    Простой текстовый датасет для language modeling.

    Логика:
      1. Читаем весь файл как одну строку (UTF‑8).
      2. Токенизируем его HuggingFace‑токенизатором.
      3. Режем на последовательности длиной `seq_len + 1`.

    Каждая последовательность используется как:
      - первые `seq_len` токенов — вход,
      - сдвинутая версия — целевые токены.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        seq_len: int,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Читаем файл целиком и токенизируем его.
        text = self.path.read_text(encoding="utf-8")
        # Один длинный ряд токенов: [N]
        input_ids = tokenizer.encode(text, return_tensors="pt").squeeze(0)

        # Сколько полных последовательностей длиной (seq_len + 1) мы можем взять.
        n_full = (input_ids.shape[0] - 1) // self.seq_len
        self.chunks: List[torch.Tensor] = []
        for i in range(n_full):
            start = i * self.seq_len
            end = start + self.seq_len + 1
            self.chunks.append(input_ids[start:end])

    def __len__(self) -> int:
        # Количество доступных чанков.
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Возвращаем последовательность токенов длиной (seq_len + 1).
        return self.chunks[idx]


def collate_lm(
    batch_tensors: Iterable[torch.Tensor],
    device: torch.device,
) -> Batch:
    """
    Коллатор для задачи language modeling.

    На вход:
        список тензоров [seq_len + 1] (каждый — пример).

    На выход:
        Batch с:
        - input_ids:     [B, seq_len]   — все токены кроме последнего;
        - target_tokens: [B, seq_len]   — все токены кроме первого;
        - loss_masks:    [B, seq_len]   — единицы (считаем loss на каждом токене).
    """
    tensors = list(batch_tensors)
    assert len(tensors) > 0

    T = tensors[0].shape[0]
    src_list = [t[:-1] for t in tensors]
    tgt_list = [t[1:] for t in tensors]

    # Склеиваем примеры в батч по первой размерности.
    src = torch.stack(src_list, dim=0).to(device)   # [B, T-1]
    tgt = torch.stack(tgt_list, dim=0).to(device)   # [B, T-1]
    mask = torch.ones_like(tgt, dtype=torch.float32, device=device)

    return Batch(
        input_ids=src,
        target_tokens=tgt,
        loss_masks=mask,
    )


def create_dataloader(
    path: str | Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    dataset_fraction: float = 1.0,
) -> DataLoader:
    """
    Построить DataLoader для обучения языковой модели.

    dataset_fraction:
        доля датасета (0.1 = 10%)
    """

    dataset = TextDataset(path, tokenizer=tokenizer, seq_len=seq_len)

    # Ограничиваем размер датасета
    if dataset_fraction < 1.0:
        n_samples = int(len(dataset) * dataset_fraction)
        indices = torch.randperm(len(dataset))[:n_samples]
        dataset = Subset(dataset, indices)

    def _collate(batch: List[torch.Tensor]) -> Batch:
        return collate_lm(batch, device=device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )
