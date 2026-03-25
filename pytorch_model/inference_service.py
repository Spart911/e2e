from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from langdetect import detect
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, pipeline

# Импортируем пакетом или как самостоятельный скрипт
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from architecture import GenerationConfig, build_model  # type: ignore
    from config import get_device  # type: ignore
    from session_cache import RedisTTTSessionCache  # type: ignore
    from transformer import Batch  # type: ignore
    from ttt import extract_inner_state_dict, load_inner_state_dict, ttt_adapt  # type: ignore
else:
    from .architecture import GenerationConfig, build_model
    from .config import get_device
    from .session_cache import RedisTTTSessionCache
    from .transformer import Batch
    from .ttt import extract_inner_state_dict, load_inner_state_dict, ttt_adapt

# Разрешаем TF32 для ускорения на Ampere+ (если доступно)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ------------------------------
# Pydantic модели под OpenAI API
# ------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None  # для tool/function сообщений


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: List[ChatMessage]
    max_tokens: int = Field(200, alias="max_tokens")
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    stream: bool = False

    # user оставляем для OpenAI-совместимых клиентов
    # session_id — наш явный способ управлять TTT-сессией
    user: Optional[str] = None
    session_id: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# ------------------------------
# Очередь для батчевой обработки
# ------------------------------


@dataclass
class QueueItem:
    request: ChatCompletionRequest
    future: asyncio.Future


MAX_BATCH_SIZE = 8
BATCH_DELAY = 0.01  # секунды, чтобы накопить несколько запросов
REQUEST_TIMEOUT = 60  # таймаут ожидания ответа для клиента

request_queue: asyncio.Queue[QueueItem] = asyncio.Queue()


# ------------------------------
# Инициализация модели/токенизатора
# ------------------------------


device = get_device(os.environ.get("DEVICE"))
checkpoint_path = os.environ.get("CHECKPOINT_PATH", "125m_pytorch.pt")
tokenizer_name = os.environ.get("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B")

print(f"[startup] device: {device}")
print(f"[startup] checkpoint: {checkpoint_path}")
print(f"[startup] tokenizer: {tokenizer_name}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = build_model(device=device, checkpoint_path=checkpoint_path)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# --- конфиг сессионного TTT ---
TTT_STEPS = int(os.environ.get("TTT_STEPS", 5))
TTT_LR = float(os.environ.get("TTT_LR", 1e-3))
TTT_SAVE_EACH_STEP = os.environ.get("TTT_SAVE_EACH_STEP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SESSION_TTL_SEC = int(os.environ.get("SESSION_TTL_SEC", 3600))
SESSION_LOCK_TTL_SEC = int(os.environ.get("SESSION_LOCK_TTL_SEC", 30))
SESSION_LOCK_BLOCKING_TIMEOUT_SEC = float(
    os.environ.get("SESSION_LOCK_BLOCKING_TIMEOUT_SEC", 10.0)
)
MODEL_REVISION = os.environ.get("MODEL_REVISION") or os.path.basename(checkpoint_path)

REDIS_URL = os.environ.get("REDIS_URL", "").strip()
SESSION_CACHE: RedisTTTSessionCache | None = None
if REDIS_URL:
    SESSION_CACHE = RedisTTTSessionCache(
        redis_url=REDIS_URL,
        ttl_sec=SESSION_TTL_SEC,
        lock_ttl_sec=SESSION_LOCK_TTL_SEC,
        lock_blocking_timeout_sec=SESSION_LOCK_BLOCKING_TIMEOUT_SEC,
    )
    print(
        f"[startup] Redis session cache enabled: "
        f"redis_url={REDIS_URL}, model_revision={MODEL_REVISION}"
    )
else:
    print("[startup] Redis session cache disabled")

# Токсичность (ленивая инициализация)
toxicity_clf: Any | None = None
TOXICITY_MODEL = os.environ.get("TOXICITY_MODEL", "").strip()


def _get_toxicity_clf():
    global toxicity_clf
    if toxicity_clf is None:
        model_name = TOXICITY_MODEL or "unitary/toxic-bert"
        toxicity_clf = pipeline(
            "text-classification",
            model=model_name,
            device=0 if device.type == "cuda" else -1,
        )
    return toxicity_clf


# ------------------------------
# Метрики Prometheus (только наши)
# ------------------------------

REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Время ответа /v1/chat/completions (сек)",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20),
    registry=REGISTRY,
)
PROMPT_LENGTH = Histogram(
    "prompt_length_chars",
    "Длина промпта (символы)",
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    registry=REGISTRY,
)
RESPONSE_LENGTH = Histogram(
    "response_length_chars",
    "Длина ответа (символы)",
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    registry=REGISTRY,
)
LANG_COUNTER = Counter(
    "prompt_language_total",
    "Распределение языков промптов",
    ["lang"],
    registry=REGISTRY,
)
TOXICITY_SCORE = Histogram(
    "response_toxicity_score",
    "Скор токсичности",
    buckets=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    registry=REGISTRY,
)
JSON_SUCCESS = Counter(
    "response_json_valid_total",
    "Валидность JSON-ответов (valid/invalid)",
    ["status"],
    registry=REGISTRY,
)


# ------------------------------
# Вспомогательные функции
# ------------------------------


def messages_to_chatml(messages: List[ChatMessage]) -> str:
    """
    Строим точный ChatML-промпт:
    <|im_start|>{role}\n{content}<|im_end|>
    ...
    <|im_start|>assistant\n
    """
    allowed_roles = {"system", "developer", "user", "assistant", "tool", "function"}
    parts: List[str] = []
    for m in messages:
        if m.role not in allowed_roles:
            raise HTTPException(status_code=400, detail=f"Недопустимая роль: {m.role}")
        name_suffix = f" name={m.name}" if m.name else ""
        parts.append(f"<|im_start|>{m.role}{name_suffix}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", 4096))


def record_prompt_metrics(prompt_text: str) -> None:
    PROMPT_LENGTH.observe(len(prompt_text))
    try:
        lang = detect(prompt_text)
    except Exception:
        lang = "unknown"
    LANG_COUNTER.labels(lang=lang).inc()


def record_response_metrics(reply_text: str) -> None:
    RESPONSE_LENGTH.observe(len(reply_text))

    # Попытка оценить токсичность, если модель доступна
    try:
        clf = _get_toxicity_clf()
        score = clf(reply_text)[0]["score"]
        TOXICITY_SCORE.observe(score)
    except Exception:
        pass

    # Проверка валидности JSON
    try:
        json.loads(reply_text)
        JSON_SUCCESS.labels(status="valid").inc()
    except Exception:
        JSON_SUCCESS.labels(status="invalid").inc()


def build_prompt_ids(req: ChatCompletionRequest) -> torch.Tensor:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages пусты")

    prompt_text = messages_to_chatml(req.messages)
    record_prompt_metrics(prompt_text)

    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0)
    if prompt_ids.shape[0] > MAX_CONTEXT_TOKENS:
        prompt_ids = prompt_ids[-MAX_CONTEXT_TOKENS:]
    return prompt_ids.to(device)


def build_generation_config(req: ChatCompletionRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id or 128001,
    )


def extract_session_id(req: ChatCompletionRequest) -> str | None:
    session_id = (req.session_id or req.user or "").strip()
    return session_id or None


def extract_last_user_message(messages: List[ChatMessage]) -> str | None:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content
    return None


def _build_one_shot_ttt_model(user_text: str):
    context_ids = tokenizer.encode(user_text, return_tensors="pt").squeeze(0).to(device)
    if int(context_ids.shape[0]) < 2 or TTT_STEPS <= 0:
        return model

    return ttt_adapt(
        model,
        context_ids=context_ids,
        device=device,
        n_steps=TTT_STEPS,
        lr=TTT_LR,
        verbose=False,
        clone_model=True,
    )


def build_inference_model_for_request(req: ChatCompletionRequest):
    """
    Возвращает модель для генерации:
      - base model, если TTT не нужен;
      - одноразово адаптированную копию, если нет session_id;
      - сессионно адаптированную копию с load/save через Redis, если session_id есть.
    """
    user_text = extract_last_user_message(req.messages)
    if not user_text or TTT_STEPS <= 0:
        return model

    context_ids = tokenizer.encode(user_text, return_tensors="pt").squeeze(0).to(device)
    session_id = extract_session_id(req)

    # Если TTT невозможен на одном токене — просто работаем без нового шага адаптации.
    # При session_id ниже всё равно можно использовать уже восстановленное состояние.
    can_run_ttt = int(context_ids.shape[0]) >= 2

    # Нет Redis или нет session_id -> одноразовая адаптация
    if SESSION_CACHE is None or not session_id:
        if not can_run_ttt:
            return model
        return ttt_adapt(
            model,
            context_ids=context_ids,
            device=device,
            n_steps=TTT_STEPS,
            lr=TTT_LR,
            verbose=False,
            clone_model=True,
        )

    assert SESSION_CACHE is not None

    def save_step_callback(
        step_idx: int,
        model_after_step,
        inner_state: dict[str, torch.Tensor],
        loss_value: float,
    ) -> None:
        if not TTT_SAVE_EACH_STEP:
            return
        try:
            SESSION_CACHE.save_inner_state(
                session_id,
                inner_state,
                checkpoint_id=MODEL_REVISION,
                extra_meta={
                    "mode": "intermediate",
                    "step": step_idx,
                    "loss": loss_value,
                },
            )
        except Exception as exc:
            print(
                f"[warn] failed to save intermediate inner_state for "
                f"session_id={session_id}: {exc}"
            )

    # lock держим только на фазе load -> apply -> ttt -> save
    # генерация идёт уже после unlock
    try:
        with SESSION_CACHE.session_lock(
            session_id,
            timeout_sec=SESSION_LOCK_TTL_SEC,
            blocking_timeout_sec=SESSION_LOCK_BLOCKING_TIMEOUT_SEC,
        ):
            session_model = deepcopy(model)

            cached_inner_state: dict[str, torch.Tensor] | None = None
            try:
                cached_inner_state = SESSION_CACHE.load_inner_state(
                    session_id,
                    checkpoint_id=MODEL_REVISION,
                    device=device,
                )
            except Exception as exc:
                print(
                    f"[warn] failed to load session cache for session_id={session_id}: {exc}"
                )
                with contextlib.suppress(Exception):
                    SESSION_CACHE.delete_session(session_id)

            if cached_inner_state is not None:
                try:
                    load_inner_state_dict(session_model, cached_inner_state, strict=True)
                except Exception as exc:
                    print(
                        f"[warn] incompatible cached inner_state for session_id={session_id}, "
                        f"resetting cache: {exc}"
                    )
                    with contextlib.suppress(Exception):
                        SESSION_CACHE.delete_session(session_id)

            if can_run_ttt:
                adapted_model = ttt_adapt(
                    session_model,
                    context_ids=context_ids,
                    device=device,
                    n_steps=TTT_STEPS,
                    lr=TTT_LR,
                    verbose=False,
                    clone_model=False,
                    step_callback=save_step_callback if TTT_SAVE_EACH_STEP else None,
                )
                try:
                    SESSION_CACHE.save_inner_state(
                        session_id,
                        extract_inner_state_dict(adapted_model),
                        checkpoint_id=MODEL_REVISION,
                        extra_meta={
                            "mode": "final",
                            "prompt_tokens": int(context_ids.shape[0]),
                        },
                    )
                except Exception as exc:
                    print(
                        f"[warn] failed to save final inner_state for "
                        f"session_id={session_id}: {exc}"
                    )
                return adapted_model

            # TTT на этом сообщении не делаем, но сохранённое состояние сессии уже применено.
            session_model.eval()
            for p in session_model.parameters():
                p.requires_grad_(False)
            return session_model

    except TimeoutError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"session_id '{session_id}' is busy",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        print(
            f"[warn] session cache path failed for session_id={session_id}, "
            f"fallback to one-shot TTT/base model: {exc}"
        )
        if not can_run_ttt:
            return model
        return _build_one_shot_ttt_model(user_text)


def run_inference(req: ChatCompletionRequest) -> ChatCompletionResponse:
    prompt_ids = build_prompt_ids(req)
    gen_cfg = build_generation_config(req)
    inference_model = build_inference_model_for_request(req)

    with torch.no_grad():
        new_ids = generate_with_cache(
            inference_model,
            prompt_ids=prompt_ids,
            device=device,
            gen_cfg=gen_cfg,
        )

    reply_text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
    record_response_metrics(reply_text)

    prompt_tokens = int(prompt_ids.shape[0])
    completion_tokens = len(new_ids)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


def sample_next_token(
    logits: torch.Tensor,
    generated: torch.Tensor,
    gen_cfg: GenerationConfig,
) -> torch.Tensor:
    # Защита: убеждаемся, что logits 1D
    if logits.dim() > 1:
        logits = logits.view(-1)
    vocab_size = logits.shape[0]

    # repetition penalty
    if gen_cfg.repetition_penalty != 1.0:
        for tok in generated.tolist():
            if tok >= vocab_size:
                continue
            val = logits[tok]
            if val > 0:
                logits[tok] = val / gen_cfg.repetition_penalty
            else:
                logits[tok] = val * gen_cfg.repetition_penalty

    if gen_cfg.temperature != 1.0:
        logits = logits / gen_cfg.temperature
    if gen_cfg.top_k > 0:
        k = min(gen_cfg.top_k, vocab_size)
        top_k_vals, _ = torch.topk(logits, k)
        logits[logits < top_k_vals[-1]] = float("-inf")
    if gen_cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs_sorted, dim=-1)
        remove = cum_probs - probs_sorted > gen_cfg.top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_stream_with_cache(
    model,
    prompt_ids: torch.Tensor,
    device: torch.device,
    gen_cfg: GenerationConfig,
):
    """
    Стриминговая генерация с KV-кэшем (накопление k/v в state блоков).
    Возвращает по одному сгенерированному токену.
    """
    model.eval()
    generated = prompt_ids.clone().to(device)  # включает префикс

    # prefill: пропускаем весь промпт, собираем state с KV
    T = generated.shape[0]
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    dummy_tgt = torch.zeros(1, T, dtype=torch.long, device=device)
    dummy_mask = torch.zeros(1, T, dtype=torch.float32, device=device)
    batch = Batch(
        input_ids=generated.unsqueeze(0),
        target_tokens=dummy_tgt,
        loss_masks=dummy_mask,
        position_ids=position_ids,
    )
    state = [None] * model.config.num_hidden_layers
    out = model(state=state, seq=batch)
    state = out.new_state

    for step in range(gen_cfg.max_new_tokens):
        logits_last = out.logits[0, -1, :].clone()
        next_tok = sample_next_token(logits_last, generated, gen_cfg)
        generated = torch.cat([generated, next_tok], dim=0)

        yield next_tok.item()

        if next_tok.item() == gen_cfg.eos_token_id:
            break

        pos_id = torch.tensor([[T + step]], device=device)
        batch = Batch(
            input_ids=next_tok.unsqueeze(0).unsqueeze(0),
            target_tokens=torch.zeros(1, 1, dtype=torch.long, device=device),
            loss_masks=torch.zeros(1, 1, dtype=torch.float32, device=device),
            position_ids=pos_id,
        )
        out = model(state=state, seq=batch)
        state = out.new_state


def generate_with_cache(
    model,
    prompt_ids: torch.Tensor,
    device: torch.device,
    gen_cfg: GenerationConfig,
) -> torch.Tensor:
    """
    Нестриминговый вариант генерации с KV-кэшем — возвращает весь сгенерированный
    хвост токенов (без токенов промпта).
    """
    model.eval()
    generated = prompt_ids.clone().to(device)  # включает префикс

    # prefill
    T = generated.shape[0]
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    dummy_tgt = torch.zeros(1, T, dtype=torch.long, device=device)
    dummy_mask = torch.zeros(1, T, dtype=torch.float32, device=device)
    batch = Batch(
        input_ids=generated.unsqueeze(0),
        target_tokens=dummy_tgt,
        loss_masks=dummy_mask,
        position_ids=position_ids,
    )
    state = [None] * model.config.num_hidden_layers
    out = model(state=state, seq=batch)
    state = out.new_state

    new_tokens = []
    for step in range(gen_cfg.max_new_tokens):
        logits_last = out.logits[0, -1, :].clone()
        next_tok = sample_next_token(logits_last, generated, gen_cfg)
        tok_id = next_tok.item()
        new_tokens.append(tok_id)
        generated = torch.cat([generated, next_tok], dim=0)

        if tok_id == gen_cfg.eos_token_id:
            break

        pos_id = torch.tensor([[T + step]], device=device)
        batch = Batch(
            input_ids=next_tok.unsqueeze(0).unsqueeze(0),
            target_tokens=torch.zeros(1, 1, dtype=torch.long, device=device),
            loss_masks=torch.zeros(1, 1, dtype=torch.float32, device=device),
            position_ids=pos_id,
        )
        out = model(state=state, seq=batch)
        state = out.new_state

    return torch.tensor(new_tokens, device=device)


async def batch_worker() -> None:
    while True:
        first: QueueItem = await request_queue.get()
        batch = [first]

        await asyncio.sleep(BATCH_DELAY)
        while len(batch) < MAX_BATCH_SIZE:
            try:
                batch.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        for qi in batch:
            try:
                result = run_inference(qi.request)
                if not qi.future.done():
                    qi.future.set_result(result)
            except Exception as exc:  # pylint: disable=broad-except
                if not qi.future.done():
                    qi.future.set_exception(exc)
            finally:
                request_queue.task_done()


async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(batch_worker())
    yield
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task


app = FastAPI(title="Local LM Inference (OpenAI API совместимость)", lifespan=lifespan)


# ------------------------------
# FastAPI endpoints
# ------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    start = time.monotonic()

    # Стриминговые запросы обрабатываем сразу (без батчевой очереди)
    if req.stream:
        prompt_ids = build_prompt_ids(req)
        gen_cfg = build_generation_config(req)
        inference_model = build_inference_model_for_request(req)

        async def event_stream():
            accumulated_text = ""
            prompt_tokens = len(prompt_ids)
            completion_tokens = 0
            created_ts = int(time.time())

            try:
                for tok_id in generate_stream_with_cache(
                    inference_model,
                    prompt_ids,
                    device,
                    gen_cfg,
                ):
                    completion_tokens += 1
                    token_text = tokenizer.decode([tok_id], skip_special_tokens=True)
                    accumulated_text += token_text

                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": token_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                record_response_metrics(accumulated_text)
                REQUEST_LATENCY.observe(time.monotonic() - start)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await request_queue.put(QueueItem(request=req, future=fut))

    try:
        resp = await asyncio.wait_for(fut, timeout=REQUEST_TIMEOUT)
        return resp
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="inference timeout") from exc
    finally:
        REQUEST_LATENCY.observe(time.monotonic() - start)


if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8000))

    # Если запускаем как скрипт из каталога пакета (без родительского PYTHONPATH),
    # используем объект app; если пакет доступен, можно дернуть строкой.
    if __package__ in (None, ""):
        uvicorn.run(app, host=host, port=port, reload=False)
    else:
        uvicorn.run(
            "pytorch_model.inference_service:app",
            host=host,
            port=port,
            reload=False,
        )