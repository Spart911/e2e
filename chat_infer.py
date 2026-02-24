import argparse
import copy
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint import options as ocp_opt
from transformers import AutoTokenizer

from ttt.config import Config, TrainingConfig, register_configs
from ttt.model.data import Batch
from ttt.model.transformer import MetaModel
from ttt.utils.jax_utils import set_random_seed


def parse_args():
    p = argparse.ArgumentParser(description="Интерактивный чат с моделью TTT (JAX).")
    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Каталог с чекпоинтами Orbax (корень, внутри которого лежат папки шагов, например 4799/).",
    )
    p.add_argument(
        "--model_size",
        type=str,
        default="125m",
        choices=["125m", "350m", "760m", "1b", "3b"],
        help="Размер модели (определяет архитектуру). Для скачанного чекпоинта используйте 125m/1b/3b и т.п.",
    )
    p.add_argument(
        "--tokenizer_name",
        default="meta-llama/Meta-Llama-3-8B",
        help="Имя токенизатора HuggingFace.",
    )
    p.add_argument("--max_new_tokens", type=int, default=128, help="Максимальное количество новых токенов в ответе.")
    p.add_argument("--temperature", type=float, default=0.8, help="Температура сэмплирования.")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) сэмплирование.")
    p.add_argument("--seed", type=int, default=0, help="Сид для генерации.")
    p.add_argument(
        "--window_size",
        type=int,
        default=8192,
        help="SWA sliding_window_size. Можно уменьшить для ускорения/экономии памяти.",
    )
    p.add_argument(
        "--decode_mini_batch_size",
        type=int,
        default=1,
        help="SWA mini_batch_size для декодинга. 1 = по токену (инкрементально, с кешем).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Сделать один прогревочный прогон (генерация пары токенов) перед началом чата для JIT-компиляции.",
    )
    p.add_argument(
        "--system_prompt",
        type=str,
        default="Вы — дружелюбный и полезный русскоязычный ассистент.",
        help="Системный промпт для настройки поведения модели.",
    )
    return p.parse_args()


def make_config(checkpoint_dir: str, model_size: str, *, window_size: int, decode_mini_batch_size: int) -> Config:
    register_configs()
    cfg = Config()

    # Базовые настройки инференса
    cfg.training.load_part = TrainingConfig.LoadPart.params
    cfg.training.dummy_dataset = True
    cfg.training.log_wandb = False

    # Выбираем архитектуру из конфигов модели/эксперимента (как в train)
    if model_size == "125m":
        cfg.model.vocab_size = 128256
        cfg.model.hidden_size = 768
        cfg.model.num_hidden_layers = 12
        cfg.model.num_attention_heads = 12
        # из configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml
        cfg.model.intermediate_size = 1664
        cfg.model.suffix_len = 3
    elif model_size == "350m":
        cfg.model.vocab_size = 128256
        cfg.model.hidden_size = 1024
        cfg.model.num_hidden_layers = 24
        cfg.model.num_attention_heads = 16
        cfg.model.intermediate_size = 2816
        cfg.model.suffix_len = 3
    elif model_size == "760m":
        cfg.model.vocab_size = 128256
        cfg.model.hidden_size = 1536
        cfg.model.num_hidden_layers = 24
        cfg.model.num_attention_heads = 16
        cfg.model.intermediate_size = 4096
        cfg.model.suffix_len = 4
    elif model_size == "1b":
        cfg.model.vocab_size = 128256
        cfg.model.hidden_size = 2048
        cfg.model.num_hidden_layers = 24
        cfg.model.num_attention_heads = 32
        # из configs/experiment/1b/pretrain/pretrain-1b-e2e.yaml
        cfg.model.intermediate_size = 4352
        cfg.model.suffix_len = 6
    elif model_size == "3b":
        cfg.model.vocab_size = 128256
        cfg.model.hidden_size = 2560
        cfg.model.num_hidden_layers = 32
        cfg.model.num_attention_heads = 32
        # из configs/experiment/3b/pretrain/pretrain-3b-e2e.yaml
        cfg.model.intermediate_size = 5632
        cfg.model.suffix_len = 8

    # Настройки, соответствующие DCLM-8k, SWA и prime
    cfg.model.seq_len = 8192
    cfg.model.sliding_window_size = int(window_size)
    cfg.model.mini_batch_size = int(decode_mini_batch_size)
    cfg.model.rope_theta = 500000.0
    cfg.training.seq_length = 8192

    cfg.model.tie_word_embeddings = True
    cfg.model.output_size = cfg.model.vocab_size
    cfg.model.seq_modeling_block = "SWA"
    cfg.model.prime = True
    cfg.model.bos_token_id = 128000
    cfg.model.eos_token_id = 128001

    # Пути до чекпоинта
    cfg.deploy_paths.checkpoint = checkpoint_dir
    cfg.checkpoint.checkpoint_dir = checkpoint_dir
    cfg.checkpoint.resume_checkpoint_dir = checkpoint_dir

    return cfg


def make_batch(token_ids):
    ids = jnp.array(token_ids, dtype=jnp.int32)
    targets = jnp.concatenate([ids[1:], ids[-1:]])
    loss_masks = jnp.ones_like(ids)

    return Batch(
        input_ids=ids,
        target_tokens=targets,
        loss_masks=loss_masks,
        attention_mask=None,
        position_ids=None,
    )


def load_model(cfg: Config, key):
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
    target_weights = model.weights()

    handler_registry = ocp.DefaultCheckpointHandlerRegistry()
    mp_opts = ocp_opt.MultiprocessingOptions(primary_host=0)
    ckpt_opts = ocp.CheckpointManagerOptions(max_to_keep=1, multiprocessing_options=mp_opts)

    manager = ocp.CheckpointManager(
        str(Path(cfg.checkpoint.checkpoint_dir)),
        options=ckpt_opts,
        handler_registry=handler_registry,
    )

    step = manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"В каталоге {cfg.checkpoint.checkpoint_dir} не найден ни один чекпоинт.")

    restored = manager.restore(
        step=step,
        args=ocp.args.Composite(
            model_weights=ocp.args.StandardRestore(target_weights, strict=True),
        ),
    )
    model = eqx.combine(restored["model_weights"], model)
    return model, state


def sample_token(logits, temperature, top_p, key):
    if temperature <= 0:
        return int(jnp.argmax(logits))

    logits = logits / temperature

    if top_p < 1.0:
        sorted_logits = jnp.sort(logits)[::-1]
        sorted_indices = jnp.argsort(logits)[::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
        cutoff = jnp.searchsorted(cumulative_probs, top_p)
        mask = jnp.ones_like(logits, dtype=bool)
        mask = mask.at[sorted_indices[cutoff + 1 :]].set(False)
        logits = jnp.where(mask, logits, -jnp.inf)

    probs = jax.nn.softmax(logits)
    return int(jax.random.categorical(key, jnp.log(probs)))


def forward_full(model: MetaModel, state, seq: Batch, cfg: Config):
    # Полный проход через языковую модель, как в train, без ручного split prefix/suffix.
    lm_out = model.language_model(state=state, seq=seq)

    class Out:
        def __init__(self, logits, state):
            self.logits = logits
            self.state = state

    return Out(logits=lm_out.logits, state=lm_out.new_state)


@eqx.filter_jit
def forward_one_token(model: MetaModel, state, token_id: jnp.ndarray):
    """Один инкрементальный шаг (декодинг): прогнать 1 токен и получить logits для следующего."""
    token_id = token_id.astype(jnp.int32)
    seq = Batch(
        input_ids=jnp.asarray([token_id], dtype=jnp.int32),
        target_tokens=jnp.asarray([token_id], dtype=jnp.int32),
        loss_masks=jnp.ones((1,), dtype=jnp.float32),
        attention_mask=None,
        position_ids=None,
    )
    lm_out = model.language_model(state=state, seq=seq)
    return lm_out.logits[0], lm_out.new_state


@eqx.filter_jit
def sample_from_logits(logits: jnp.ndarray, key: jax.Array, temperature: float, top_p: float):
    key, subkey = jax.random.split(key)
    if temperature <= 0:
        return jnp.argmax(logits).astype(jnp.int32), key

    l = logits / jnp.asarray(temperature, dtype=logits.dtype)
    if top_p < 1.0:
        sorted_indices = jnp.argsort(l)[::-1]
        sorted_logits = l[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
        cutoff = jnp.searchsorted(cumulative_probs, jnp.asarray(top_p, dtype=cumulative_probs.dtype))
        keep_sorted = jnp.arange(l.shape[0]) <= cutoff
        keep = jnp.zeros_like(keep_sorted, dtype=bool).at[sorted_indices].set(keep_sorted)
        l = jnp.where(keep, l, -jnp.inf)

    probs = jax.nn.softmax(l)
    next_id = jax.random.categorical(subkey, jnp.log(probs)).astype(jnp.int32)
    return next_id, key


def consume_tokens(model: MetaModel, state, token_ids: list[int]):
    """Префилл: прогнать известные токены через модель, обновив SWA-кеш и state."""
    last_logits = None
    for tid in token_ids:
        last_logits, state = forward_one_token(model, state, jnp.asarray(tid, dtype=jnp.int32))
    return last_logits, state


def generate_response_incremental(cfg, model, state, tokenizer, prompt_token_ids: list[int], args, key, stream: bool = True):
    # Префилл: скармливаем токены запроса/префикса
    last_logits, state = consume_tokens(model, state, prompt_token_ids)
    if last_logits is None:
        last_logits, state = forward_one_token(model, state, jnp.asarray(cfg.model.bos_token_id, dtype=jnp.int32))

    out_token_ids: list[int] = []

    for _ in range(args.max_new_tokens):
        next_id, key = sample_from_logits(last_logits, key, float(args.temperature), float(args.top_p))
        next_id_i = int(jax.device_get(next_id))
        out_token_ids.append(next_id_i)

        if stream:
            piece = tokenizer.decode([next_id_i], skip_special_tokens=True)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()

        if next_id_i == cfg.model.eos_token_id:
            break

        last_logits, state = forward_one_token(model, state, jnp.asarray(next_id_i, dtype=jnp.int32))

    text = tokenizer.decode(out_token_ids, skip_special_tokens=True)
    return text, key, state


def generate_response(cfg, model, state, tokenizer, conversation_tokens, args, key, stream: bool = False):
    if len(conversation_tokens) + args.max_new_tokens > cfg.model.seq_len:
        overflow = len(conversation_tokens) + args.max_new_tokens - cfg.model.seq_len
        conversation_tokens = conversation_tokens[overflow:]

    start_len = len(conversation_tokens)

    for _ in range(args.max_new_tokens):
        batch = make_batch(conversation_tokens)
        outputs = forward_full(model, state, batch, cfg)
        logits = outputs.logits[-1]
        state = outputs.state

        key, subkey = jax.random.split(key)
        next_id = sample_token(logits, args.temperature, args.top_p, subkey)
        conversation_tokens.append(next_id)

        if stream:
            piece = tokenizer.decode([next_id], skip_special_tokens=True)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()

        if next_id == tokenizer.eos_token_id:
            break

    new_tokens = conversation_tokens[start_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, conversation_tokens, key, state


def chat_loop(cfg, model, state, tokenizer, args):
    print("Интерактивный режим чата. Введите 'exit' или 'выход' для завершения.\n")
    key = jax.random.PRNGKey(args.seed)

    if args.system_prompt:
        system_text = f"[СИСТЕМА]: {args.system_prompt}\n"
        _logits, state = consume_tokens(model, state, tokenizer.encode(system_text, add_special_tokens=False))

    while True:
        try:
            sys.stdout.write("Вы: ")
            sys.stdout.flush()
            raw = sys.stdin.buffer.readline()
            if not raw:
                print("\nЗавершение работы.")
                break
            # Безопасное декодирование любых локалей/байтов.
            user_text = raw.decode(errors="ignore").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break

        if user_text.lower() in {"exit", "quit", "выход"}:
            print("Завершение работы.")
            break

        turn_prefix = f"\n[ПОЛЬЗОВАТЕЛЬ]: {user_text}\n[МОДЕЛЬ]:"
        prompt_ids = tokenizer.encode(turn_prefix, add_special_tokens=False)

        print("Модель: ", end="", flush=True)
        response_text, key, state = generate_response_incremental(cfg, model, state, tokenizer, prompt_ids, args, key, stream=True)
        print("\n")


def main():
    args = parse_args()
    cfg = make_config(
        args.checkpoint_dir,
        args.model_size,
        window_size=args.window_size,
        decode_mini_batch_size=args.decode_mini_batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    key = set_random_seed(args.seed)
    model, state = load_model(cfg, key)

    if args.dry_run:
        print("Выполняю dry-run (прогрев JAX/JIT, это может занять время)...")
        warm_args = argparse.Namespace(**vars(args))
        warm_args.max_new_tokens = 4
        warm_state = copy.deepcopy(state)
        _text, _key, _warm_state = generate_response_incremental(
            cfg,
            model,
            warm_state,
            tokenizer,
            tokenizer.encode("Привет", add_special_tokens=False) or [cfg.model.bos_token_id],
            warm_args,
            jax.random.PRNGKey(args.seed),
            stream=False,
        )
        print("Dry-run завершён, теперь ответы на реальные запросы будут быстрее.\n")

    chat_loop(cfg, model, state, tokenizer, args)


if __name__ == "__main__":
    main()

