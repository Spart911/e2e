from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .transformer import Batch, CausalLM, cross_entropy_loss_and_accuracy
from .config import MODEL_CFG


@dataclass
class GenerationConfig:
    """
    Параметры, управляющие стратегией генерации текста.

    Важно:
    - max_new_tokens: сколько новых токенов добавляем поверх промпта;
    - temperature: сглаживание распределения (1.0 — без изменений);
    - top_p / top_k: nucleus / top-k отбор кандидатов;
    - repetition_penalty: штраф за повторяющиеся токены;
    - eos_token_id: специальный токен завершения (если встречен — стоп).
    """
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    eos_token_id: int = 128001


def build_model(device: torch.device, checkpoint_path: str | None = None) -> CausalLM:
    """
    Создать экземпляр `CausalLM` с конфигом `MODEL_CFG` и,
    при необходимости, загрузить веса из PyTorch‑чекпоинта.

    Ожидаемый формат чекпоинта:
        {"model_weights": <state_dict>}
    совместим с исходным `125m_pytorch.pt`.
    """
    # Создаём модель с нужной архитектурой (слои, размерности и т.п.).
    model = CausalLM(MODEL_CFG)

    if checkpoint_path is not None:
        # Загружаем словарь весов на CPU (дешевле по памяти/GPU).
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = payload["model_weights"]

        # В исходном коде lm_head может быть "tied" к эмбеддингам.
        # Если голова отсутствует, но есть wte, создаём её из эмбеддингов.
        if "lm_head.weight" not in weights and "model.wte.weight" in weights:
            weights["lm_head.weight"] = (
                weights["model.wte.weight"].clone().t().contiguous()
            )

        # Грузим веса с `strict=False`, чтобы не падать из‑за несущественных расхождений.
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"[warn] Missing keys: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys: {unexpected}")

    # Переносим модель на нужное устройство и ставим в режим eval по умолчанию.
    model.to(device)
    model.eval()
    return model


def make_batch(input_ids: torch.Tensor, device: torch.device) -> Batch:
    """
    Построить `Batch` для задачи language modeling из ОДНОЙ последовательности токенов.

    Вход:
        input_ids: [T] — последовательность токенов (например, токенизированный текст).

    Выход:
        Batch c:
        - input_ids: [1, T-1]  — все токены, кроме последнего;
        - target_tokens: [1, T-1] — все токены, кроме первого (сдвиг на 1);
        - loss_masks: [1, T-1] — единицы (считать loss на каждом токене).
    """
    T = input_ids.shape[0]
    src = input_ids[:-1]          # [T-1] входные токены
    tgt = input_ids[1:]           # [T-1] целевые токены
    mask = torch.ones(T - 1, dtype=torch.float32, device=device)
    return Batch(
        input_ids=src.unsqueeze(0).to(device),      # [1, T-1]
        target_tokens=tgt.unsqueeze(0).to(device),  # [1, T-1]
        loss_masks=mask.unsqueeze(0).to(device),    # [1, T-1]
    )


@torch.no_grad()
def generate(
    model: CausalLM,
    prompt_ids: torch.Tensor,
    device: torch.device,
    gen_cfg: GenerationConfig | None = None,
) -> torch.Tensor:
    """
    Авторегрессивно сгенерировать продолжение последовательности токенов.

    На каждом шаге:
      1. Прогоняем всю текущую последовательность через модель.
      2. Берём логиты последнего токена.
      3. Применяем repetition penalty / temperature / top-k / top-p.
      4. Семплируем следующий токен и дописываем его в последовательность.
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()

    model.eval()
    # Начальное состояние — только токены промпта.
    generated = prompt_ids.clone().to(device)  # [T_prompt]

    for _ in range(gen_cfg.max_new_tokens):
        T = generated.shape[0]
        dummy_tgt = torch.zeros(1, T, dtype=torch.long, device=device)
        dummy_mask = torch.zeros(1, T, dtype=torch.float32, device=device)

        # Формируем псевдо‑батч из всей последовательности: модель ожидает объект `Batch`.
        batch = Batch(
            input_ids=generated.unsqueeze(0),   # [1, T]
            target_tokens=dummy_tgt,
            loss_masks=dummy_mask,
        )

        from .config import MODEL_CFG as _CFG  # локальный импорт, чтобы избежать циклов

        # В простом сценарии инициализируем состояние блоков как None
        # (можно заменить на более сложный KV‑кэш, если потребуется).
        state = [None] * _CFG.num_hidden_layers
        out = model(state=state, seq=batch)
        logits = out.logits[0, -1, :]  # [vocab] — последний токен

        # Repetition penalty: подавляем/усиливаем уже встречавшиеся токены.
        if gen_cfg.repetition_penalty != 1.0:
            for tok in generated:
                val = logits[tok]
                if val > 0:
                    logits[tok] = val / gen_cfg.repetition_penalty
                else:
                    logits[tok] = val * gen_cfg.repetition_penalty

        # Temperature: сглаживаем (T>1) или заостряем (T<1) распределение.
        if gen_cfg.temperature != 1.0:
            logits = logits / gen_cfg.temperature

        # Top-k: считаем, что вероятность вне top_k кандидатов = 0.
        if gen_cfg.top_k > 0:
            top_k_vals, _ = torch.topk(logits, gen_cfg.top_k)
            logits[logits < top_k_vals[-1]] = float("-inf")

        # Top-p (nucleus): оставляем минимальное множество токенов,
        # суммарная вероятность которых ≥ top_p.
        if gen_cfg.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs_sorted = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs_sorted, dim=-1)
            remove = cum_probs - probs_sorted > gen_cfg.top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

        # Итоговое распределение вероятностей по словарю.
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        # Дописываем сэмплированный токен к последовательности.
        generated = torch.cat([generated, next_tok], dim=0)

        # Останавливаемся, если встретили токен окончания последовательности.
        if next_tok.item() == gen_cfg.eos_token_id:
            break

    return generated[prompt_ids.shape[0]:]


def lm_loss(
    model: CausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Удобная обёртка: посчитать loss для одной токенизированной последовательности.

    Под капотом:
      - строит `Batch` через `make_batch`;
      - прогоняет через модель;
      - считает cross‑entropy loss.
    Удобно использовать в отладке или для одиночных примеров.
    """
    batch = make_batch(input_ids.to(device), device)
    from .config import MODEL_CFG as _CFG

    state = [None] * _CFG.num_hidden_layers
    out = model(state=state, seq=batch)
    logits = out.logits
    loss, _ = cross_entropy_loss_and_accuracy(
        logits, batch.target_tokens, batch.loss_masks
    )
    return loss

