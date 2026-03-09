from __future__ import annotations

"""
Инференс в виде чата с TTT:

Пример запуска:

    python -m pytorch_model.chat_ttt \
        --checkpoint 125m_pytorch.pt \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --ttt_steps 5 \
        --ttt_lr 1e-3 \
        --max_new_tokens 200

Внутри запустится REPL:
    вы пишете реплики, модель отвечает, перед каждым ответом
    проводится TTT-адаптация на вашем последнем сообщении.
"""

import argparse

import torch
from transformers import AutoTokenizer

from .architecture import build_model, generate, GenerationConfig
from .config import get_device
from .ttt import ttt_adapt


def tokenize(text: str, tokenizer) -> torch.Tensor:
    """
    Удобная обёртка вокруг токенизатора HuggingFace.

    Возвращает 1D‑тензор токенов (без batch‑размерности).
    """
    return tokenizer.encode(text, return_tensors="pt").squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat inference with TTT")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к чекпоинту (.pt) формата {'model_weights': ...}")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ttt_steps", type=int, default=5)
    parser.add_argument("--ttt_lr", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    # Выбираем устройство (cpu / cuda).
    device = get_device(args.device)
    print(f"[info] device: {device}")

    # Грузим токенизатор и модель.
    print(f"[info] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"[info] loading model from {args.checkpoint}")
    base_model = build_model(device=device, checkpoint_path=args.checkpoint)

    # Конфигурация генерации для всех ответов в чате.
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

    # Храним историю диалога (пока используется только для возможного расширения).
    history = []  # можно расширить до полноценного диалога при желании

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

        # Контекст для TTT — текущее сообщение пользователя.
        context_ids = tokenize(user_text, tokenizer).to(device)

        print("[TTT] адаптация на вашем сообщении...")
        adapted_model = ttt_adapt(
            base_model,
            context_ids=context_ids,
            device=device,
            n_steps=args.ttt_steps,
            lr=args.ttt_lr,
            verbose=False,
        )

        # Промпт для генерации: берём то же сообщение пользователя.
        prompt_ids = tokenize(user_text, tokenizer).to(device)
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

