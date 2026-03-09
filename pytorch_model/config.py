from __future__ import annotations

import torch

from .transformer import ModelConfig


# Базовая конфигурация модели, аналогичная `MODEL_CFG` из `inference_ttt.py`.
# Если вы хотите изменить архитектуру (кол-во слоёв, размер скрытого слоя и т.п.),
# делайте это здесь и перetrain‑ьте модель с нуля или через конвертацию весов.
MODEL_CFG = ModelConfig(
    vocab_size=128256,
    output_size=128256,
    hidden_size=768,
    intermediate_size=1664,
    num_hidden_layers=12,
    num_attention_heads=12,
    suffix_len=3,          # последние 3 блока — suffix (TTT блоки)
    prime=True,            # prime_storage активен
    qk_norm=True,
    pre_norm=True,
    post_norm=True,
    compute_dtype="float32",
    param_dtype="float32",
    state_dtype="float32",
)


def get_device(device_str: str | None = None) -> torch.device:
    """
    Удобная функция выбора устройства.

    Приоритет:
      1. Если явно передана строка (например, "cuda" или "cpu") — используем её.
      2. Иначе берём "cuda", если она доступна, иначе "cpu".
    """
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

