from __future__ import annotations

"""
Полное дообучение модели на текстовом датасете.

Сценарий:
    1. Есть базовый чекпоинт `checkpoint_in` (например, 125m_pytorch.pt).
    2. Есть текстовый файл `data_path` с обучающим корпусом.
    3. Запускаем этот скрипт, модель дообучается и сохраняется в `save_path`.
"""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from .architecture import build_model
from .config import MODEL_CFG, get_device
from .data import create_dataloader
from .losses import language_modeling_loss


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Одна эпоха обучения по всему датасету.

    На каждом шаге:
      - прогоняем батч через модель;
      - считаем language modeling loss;
      - делаем шаг оптимизации.
    """
    model.train()
    total_loss = 0.0
    n_steps = 0

    for batch in dataloader:
        # `batch` — объект `Batch` из transformer.py (см. collate_fn в data.py).
        optimizer.zero_grad()

        # Для простоты инициализируем состояние блоков как список None.
        state = [None] * MODEL_CFG.num_hidden_layers
        out = model(state=state, seq=batch)
        loss = language_modeling_loss(
            out.logits, batch.target_tokens, batch.loss_masks
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full fine-tuning of the transformer LM")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Доля датасета для обучения (0.1 = 10%)")
    parser.add_argument("--checkpoint_in", type=str, default=None,help="Путь к исходному чекпоинту (.pt) или None для случайной инициализации")
    parser.add_argument("--data_path", type=str, required=True,help="Текстовый файл с данными для обучения")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="finetuned.pt")

    args = parser.parse_args()

    # Определяем устройство (cpu / cuda) с учётом аргумента командной строки.
    device = get_device(args.device)
    print(f"[info] device: {device}")

    # Загружаем HuggingFace‑токенизатор для подготовки данных.
    print(f"[info] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Строим модель и, при необходимости, загружаем исходный чекпоинт.
    print(f"[info] building model...")
    model = build_model(device=device, checkpoint_path=args.checkpoint_in)

    # Создаём DataLoader поверх текстового файла.
    print(f"[info] creating dataloader from {args.data_path}")
    dataloader = create_dataloader(
        path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dataset_fraction=args.dataset_fraction,
        device=device,
        shuffle=True,
    )

    # AdamW как стандартный оптимизатор для дообучения трансформеров.
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Основной обучающий цикл по эпохам.
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"[epoch {epoch}/{args.epochs}] loss = {avg_loss:.4f}")

    # Сохраняем только веса модели в формате, совместимом с build_model.
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_weights": model.state_dict()},
        save_path,
    )
    print(f"[info] saved finetuned checkpoint to {save_path}")


if __name__ == "__main__":
    main()

