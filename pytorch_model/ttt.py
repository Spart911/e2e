from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch

try:
    from .transformer import CausalLM, Batch, cross_entropy_loss_and_accuracy
    from .config import MODEL_CFG
except ImportError:  # script-mode fallback
    from transformer import CausalLM, Batch, cross_entropy_loss_and_accuracy  # type: ignore
    from config import MODEL_CFG  # type: ignore


def get_inner_params(model: CausalLM) -> dict[str, torch.nn.Parameter]:
    """
    Выбрать inner-параметры для Test-Time Training (TTT).

    Идея:
      - дообучаем только параметры suffix-блоков (последние `suffix_len` блоков),
        чтобы не трогать всю модель и стабилизировать адаптацию;
      - опираемся на имена параметров вида `model.h.blocks.<idx>.…`;
      - если по какой-то причине имена отличаются, есть несколько fallback-веток,
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


# --- новая функция ---
def extract_inner_state_dict(model: CausalLM) -> dict[str, torch.Tensor]:
    """
    Снять слепок только inner-параметров модели.

    Важно:
      - возвращаем detached+clone тензоры, чтобы дальнейшие optimizer.step()
        не меняли уже извлечённый state;
      - device/dtype сохраняются как у исходных параметров; при сериализации
        вызывающий код обычно переносит их на CPU.
    """
    return {
        name: param.detach().clone()
        for name, param in get_inner_params(model).items()
    }


# --- новая функция ---
def load_inner_state_dict(
    model: CausalLM,
    inner_state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Наложить inner_state на модель.

    strict=True:
      - проверяем, что набор ключей совпадает с текущими inner-параметрами;
      - проверяем совпадение shape;
      - при несовместимости выбрасываем исключение.

    Возвращает отчёт для логирования/отладки.
    """
    inner_params = get_inner_params(model)
    expected_keys = set(inner_params.keys())
    provided_keys = set(inner_state.keys())

    missing_keys = sorted(expected_keys - provided_keys)
    unexpected_keys = sorted(provided_keys - expected_keys)

    shape_mismatches: list[str] = []
    loaded_keys: list[str] = []

    with torch.no_grad():
        for name, tensor in inner_state.items():
            if name not in inner_params:
                continue
            param = inner_params[name]
            if tuple(param.shape) != tuple(tensor.shape):
                shape_mismatches.append(
                    f"{name}: expected {tuple(param.shape)}, got {tuple(tensor.shape)}"
                )
                continue
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded_keys.append(name)

    if strict and (missing_keys or unexpected_keys or shape_mismatches):
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing_keys={missing_keys[:10]}")
        if unexpected_keys:
            problems.append(f"unexpected_keys={unexpected_keys[:10]}")
        if shape_mismatches:
            problems.append(f"shape_mismatches={shape_mismatches[:10]}")
        raise ValueError("inner_state несовместим с моделью: " + "; ".join(problems))

    return {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
    }


def make_batch(context_ids: torch.Tensor, device: torch.device) -> Batch:
    """
    Построить `Batch` для TTT по контексту пользователя.

    Вход:
        context_ids: [T] — токены контекста (например, промпт или реплика).

    Выход:
        Batch [1, T-1] с классическим сдвигом на 1.
    """
    # --- защита от некорректного контекста ---
    if context_ids.ndim != 1:
        raise ValueError(
            f"context_ids должен быть 1D, получили shape={tuple(context_ids.shape)}"
        )

    T = int(context_ids.shape[0])
    if T < 2:
        raise ValueError(
            "Для TTT нужно минимум 2 токена в context_ids: иначе нельзя "
            "построить (src, tgt) со сдвигом на 1."
        )

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
    *,
    clone_model: bool = True,  # ---
    step_callback: Callable[[int, CausalLM, dict[str, torch.Tensor], float], None] | None = None,  # ---
) -> CausalLM:
    """
    Выполнить Test-Time Training (TTT) на заданном контексте.

    Что происходит:
      1. При clone_model=True создаём копию исходной модели (оригинал не трогаем).
         При clone_model=False адаптируем переданный объект in-place.
      2. Размораживаем только inner-параметры (см. `get_inner_params`).
      3. Запускаем несколько шагов SGD по cross-entropy loss на контексте.
      4. После каждого optimizer.step() можем вызвать step_callback, чтобы
         сохранить промежуточный inner_state.

    Возвращает:
      - модель `adapted`, готовую к генерации ответа под данный контекст.
    """
    adapted = deepcopy(model) if clone_model else model
    adapted.train()

    inner_params = get_inner_params(adapted)
    if not inner_params:
        raise RuntimeError("get_inner_params(...) вернул пустой набор параметров для TTT")

    for name, p in adapted.named_parameters():
        p.requires_grad_(name in inner_params)

    optimizer = torch.optim.SGD(
        list(inner_params.values()),
        lr=lr,
        momentum=0.0,
    )

    batch = make_batch(context_ids, device)

    if verbose:
        print(
            f"\n[TTT] Адаптация на {context_ids.shape[0]} токенах, "
            f"{n_steps} шагов, lr={lr}, clone_model={clone_model}"
        )
        print(
            "[TTT] Дообучаемых параметров: "
            f"{sum(p.numel() for p in inner_params.values()):,}"
        )

    for step_idx in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        state = [None] * MODEL_CFG.num_hidden_layers
        out = adapted(state=state, seq=batch)
        logits = out.logits  # [1, T-1, vocab]

        loss, _ = cross_entropy_loss_and_accuracy(
            logits, batch.target_tokens, batch.loss_masks
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"TTT loss не является конечным числом: {loss}")

        loss.backward()
        optimizer.step()

        # --- optional callback после optimizer.step() ---
        if step_callback is not None:
            step_callback(
                step_idx + 1,
                adapted,
                extract_inner_state_dict(adapted),
                float(loss.detach().item()),
            )

        if verbose:
            print(f"  step {step_idx + 1}/{n_steps}  loss={loss.item():.4f}")

    adapted.eval()
    for p in adapted.parameters():
        p.requires_grad_(False)

    return adapted