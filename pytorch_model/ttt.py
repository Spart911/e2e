from __future__ import annotations

import re
from copy import deepcopy

import torch

try:
    from .transformer import CausalLM, Batch, cross_entropy_loss_and_accuracy
    from .config import MODEL_CFG
except ImportError:  # script-mode fallback
    from transformer import CausalLM, Batch, cross_entropy_loss_and_accuracy  # type: ignore
    from config import MODEL_CFG  # type: ignore


def get_inner_params(model: CausalLM) -> dict[str, torch.nn.Parameter]:
    """
    Выбрать inner‑параметры для Test-Time Training (TTT).

    Идея:
      - дообучаем только параметры suffix‑блоков (последние `suffix_len` блоков),
        чтобы не трогать всю модель и стабилизировать адаптацию;
      - опираемся на имена параметров вида `model.h.blocks.<idx>.…`;
      - если по какой‑то причине имена отличаются, есть несколько fallback‑веток,
        чтобы не получить пустой список параметров.
    """
    suffix_len = MODEL_CFG.suffix_len
    n_layers = MODEL_CFG.num_hidden_layers
    suffix_start = n_layers - suffix_len if suffix_len > 0 else 0

    inner_params: dict[str, torch.nn.Parameter] = {}

    # 1) Основной путь: отдельные блоки `model.h.blocks.<idx>.*`.
    for name, p in model.named_parameters():
        m = re.search(r"\bmodel\.h\.blocks\.(\d+)\.", name)
        if not m:
            continue
        block_idx = int(m.group(1))
        if block_idx >= suffix_start:
            inner_params[name] = p

    # 2) Fallback: если имена без индексов — берём всё из `model.h.blocks`.
    if not inner_params:
        for name, p in model.named_parameters():
            if "model.h.blocks" in name:
                inner_params[name] = p

    # 3) Последний fallback: все параметры, кроме embedding и lm_head.
    if not inner_params:
        for name, p in model.named_parameters():
            if "wte." in name or "lm_head" in name:
                continue
            inner_params[name] = p

    return inner_params


def make_batch(context_ids: torch.Tensor, device: torch.device) -> Batch:
    """
    Построить `Batch` для TTT по контексту пользователя.

    Вход:
        context_ids: [T] — токены контекста (например, промпт или реплика).

    Выход:
        Batch [1, T-1] с классическим сдвигом на 1.
    """
    T = context_ids.shape[0]
    src = context_ids[:-1]
    tgt = context_ids[1:]
    mask = torch.ones(T - 1, dtype=torch.float32, device=device)
    return Batch(
        input_ids=src.unsqueeze(0).to(device),
        target_tokens=tgt.unsqueeze(0).to(device),
        loss_masks=mask.unsqueeze(0).to(device),
    )


def ttt_adapt(
    model: CausalLM,
    context_ids: torch.Tensor,
    device: torch.device,
    n_steps: int = 5,
    lr: float = 1e-3,
    verbose: bool = True,
) -> CausalLM:
    """
    Выполнить Test-Time Training (TTT) на заданном контексте.

    Что происходит:
      1. Создаём **копию** исходной модели (оригинал не трогаем).
      2. Размораживаем только inner‑параметры (см. `get_inner_params`).
      3. Запускаем несколько шагов SGD по cross‑entropy loss на контексте.

    Возвращает:
      - новую модель `adapted`, готовую к генерации ответа под данный контекст.
    """
    # Работаем на копии, чтобы не портить исходную модель.
    adapted = deepcopy(model)
    adapted.train()

    # Выбираем подмножество параметров, которые будут обучаться.
    inner_params = get_inner_params(adapted)
    for name, p in adapted.named_parameters():
        p.requires_grad_(name in inner_params)

    # Простой SGD для inner‑параметров.
    optimizer = torch.optim.SGD(
        [p for p in inner_params.values()],
        lr=lr,
        momentum=0.0,
    )

    # Строим батч из одного контекста.
    batch = make_batch(context_ids, device)

    if verbose:
        print(
            f"\n[TTT] Адаптация на {context_ids.shape[0]} токенах, "
            f"{n_steps} шагов, lr={lr}"
        )
        print(
            f"[TTT] Дообучаемых параметров: "
            f"{sum(p.numel() for p in inner_params.values()):,}"
        )

    # Небольшой inner‑loop по контексту.
    for step in range(n_steps):
        optimizer.zero_grad()

        # Для простоты инициализируем состояние блоков как None.
        state = [None] * MODEL_CFG.num_hidden_layers
        out = adapted(state=state, seq=batch)
        logits = out.logits  # [1, T-1, vocab]

        # Cross‑entropy loss поверх контекста.
        loss, _ = cross_entropy_loss_and_accuracy(
            logits, batch.target_tokens, batch.loss_masks
        )
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"  step {step + 1}/{n_steps}  loss={loss.item():.4f}")

    # После адаптации сводим модель обратно в eval‑режим и выключаем градиенты.
    adapted.eval()
    for p in adapted.parameters():
        p.requires_grad_(False)

    return adapted

