from __future__ import annotations

"""
Инференс в виде чата с TTT и опциональным Redis-кэшем по session_id.

Пример запуска:

    python -m pytorch_model.chat_ttt \
        --checkpoint 125m_pytorch.pt \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --ttt_steps 5 \
        --ttt_lr 1e-3 \
        --redis_url redis://localhost:6379/0 \
        --session_id demo-user-1 \
        --max_new_tokens 200

Внутри запустится REPL:
    вы пишете реплики, модель отвечает; перед каждым ответом
    проводится TTT-адаптация на последнем сообщении пользователя.
    Если задан Redis, inner-веса между сообщениями сохраняются по session_id.
"""

import argparse
import os
import sys
import uuid
from copy import deepcopy

import torch
from transformers import AutoTokenizer

try:
    from .architecture import build_model, generate, GenerationConfig
    from .config import get_device
    from .session_cache import RedisTTTSessionCache
    from .ttt import (
        extract_inner_state_dict,
        load_inner_state_dict,
        ttt_adapt,
    )
except ImportError:  # script-mode fallback
    from architecture import build_model, generate, GenerationConfig  # type: ignore
    from config import get_device  # type: ignore
    from session_cache import RedisTTTSessionCache  # type: ignore
    from ttt import extract_inner_state_dict, load_inner_state_dict, ttt_adapt  # type: ignore


def tokenize(text: str, tokenizer) -> torch.Tensor:
    """
    Удобная обёртка вокруг токенизатора HuggingFace.

    Возвращает 1D-тензор токенов (без batch-размерности).
    """
    return tokenizer.encode(text, return_tensors="pt").squeeze(0)


def str2bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8", errors="ignore")
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Chat inference with TTT")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Путь к чекпоинту (.pt) формата {'model_weights': ...}",
    )
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ttt_steps", type=int, default=5)
    parser.add_argument("--ttt_lr", type=float, default=1e-3)
    parser.add_argument(
        "--ttt_save_each_step",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    # --- опции Redis/session cache ---
    parser.add_argument("--redis_url", type=str, default="")
    parser.add_argument("--session_id", type=str, default="")
    parser.add_argument("--session_ttl_sec", type=int, default=3600)
    parser.add_argument("--session_lock_ttl_sec", type=int, default=30)
    parser.add_argument("--session_lock_blocking_timeout_sec", type=float, default=10.0)
    parser.add_argument("--checkpoint_id", type=str, default="")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[info] device: {device}")

    print(f"[info] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"[info] loading model from {args.checkpoint}")
    base_model = build_model(device=device, checkpoint_path=args.checkpoint)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    checkpoint_id = args.checkpoint_id or os.path.basename(args.checkpoint)
    redis_url = args.redis_url.strip()
    session_id = args.session_id.strip()

    cache: RedisTTTSessionCache | None = None
    if redis_url:
        cache = RedisTTTSessionCache(
            redis_url=redis_url,
            ttl_sec=args.session_ttl_sec,
            lock_ttl_sec=args.session_lock_ttl_sec,
            lock_blocking_timeout_sec=args.session_lock_blocking_timeout_sec,
        )
        if not session_id:
            session_id = f"cli-{uuid.uuid4().hex}"
        print(
            f"[info] Redis session cache enabled: "
            f"session_id={session_id}, checkpoint_id={checkpoint_id}"
        )
    else:
        print("[info] Redis session cache disabled")

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id or 128001,
    )

    print("=== Chat with TTT ===")
    print("Введите сообщение и нажмите Enter. Для выхода введите `/exit`.")
    if session_id:
        print(f"[info] session_id: {session_id}")

    history = []

    while True:
        try:
            user_text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text == "":
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break

        history.append(("user", user_text))

        context_ids = tokenize(user_text, tokenizer).to(device)
        prompt_ids = tokenize(user_text, tokenizer).to(device)

        adapted_model = base_model

        # --- fallback без Redis = одноразовая TTT-адаптация ---
        if cache is None:
            if int(context_ids.shape[0]) >= 2:
                print("[TTT] одноразовая адаптация на вашем сообщении...")
                adapted_model = ttt_adapt(
                    base_model,
                    context_ids=context_ids,
                    device=device,
                    n_steps=args.ttt_steps,
                    lr=args.ttt_lr,
                    verbose=False,
                    clone_model=True,
                )

        # --- полноценная сессионная логика через Redis ---
        else:
            print("[TTT] восстановление/обновление сессионных inner-весов...")

            def save_step_callback(
                step_idx: int,
                model_after_step,
                inner_state: dict[str, torch.Tensor],
                loss_value: float,
            ) -> None:
                if not args.ttt_save_each_step:
                    return
                cache.save_inner_state(
                    session_id,
                    inner_state,
                    checkpoint_id=checkpoint_id,
                    extra_meta={
                        "step": step_idx,
                        "loss": loss_value,
                        "mode": "intermediate",
                    },
                )

            with cache.session_lock(session_id):
                # Делаем одну копию на запрос, потом адаптируем её in-place.
                session_model = deepcopy(base_model)

                cached_inner_state = cache.load_inner_state(
                    session_id,
                    checkpoint_id=checkpoint_id,
                    device=device,
                )
                if cached_inner_state is not None:
                    try:
                        load_inner_state_dict(session_model, cached_inner_state, strict=True)
                    except Exception as exc:
                        print(
                            f"[warn] cached inner_state несовместим с моделью, "
                            f"cache reset: {exc}"
                        )
                        cache.delete_session(session_id)

                if int(context_ids.shape[0]) >= 2:
                    adapted_model = ttt_adapt(
                        session_model,
                        context_ids=context_ids,
                        device=device,
                        n_steps=args.ttt_steps,
                        lr=args.ttt_lr,
                        verbose=False,
                        clone_model=False,
                        step_callback=save_step_callback if args.ttt_save_each_step else None,
                    )
                    cache.save_inner_state(
                        session_id,
                        extract_inner_state_dict(adapted_model),
                        checkpoint_id=checkpoint_id,
                        extra_meta={
                            "mode": "final",
                            "prompt_tokens": int(context_ids.shape[0]),
                        },
                    )
                else:
                    # Если на этом сообщении TTT невозможен (<2 токенов),
                    # просто используем уже восстановленное сессионное состояние.
                    adapted_model = session_model.eval()
                    for p in adapted_model.parameters():
                        p.requires_grad_(False)

        with torch.no_grad():
            new_ids = generate(
                adapted_model,
                prompt_ids=prompt_ids,
                device=device,
                gen_cfg=gen_cfg,
            )

        reply_text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
        history.append(("assistant", reply_text))

        print(f"model: {reply_text}")


if __name__ == "__main__":
    main()