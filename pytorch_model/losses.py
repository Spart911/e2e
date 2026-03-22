from __future__ import annotations

import torch

try:
    from .transformer import cross_entropy_loss_and_accuracy
except ImportError:  # script-mode fallback
    from transformer import cross_entropy_loss_and_accuracy  # type: ignore


def language_modeling_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    loss_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Классический cross‑entropy loss для языковой модели.

    Параметры:
      - logits: [B, T, V] — логиты модели по всему словарю;
      - target_tokens: [B, T] — целевые токены;
      - loss_masks: [B, T] (опционально) — маска валидных позиций.

    Возвращает:
      - усреднённый по батчу скалярный loss.

    Под капотом делегирует работу функции `cross_entropy_loss_and_accuracy`
    из `transformer.py` и забирает только компоненту loss.
    """
    loss, _ = cross_entropy_loss_and_accuracy(
        logits, target_tokens, loss_masks
    )
    return loss

