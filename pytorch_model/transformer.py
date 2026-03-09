"""
Упрощённая копия архитектуры из корневого `transformer.py`,
ограниченная только тем, что нужно для:

- инференса (`CausalLM`, `Batch`);
- лосса (`cross_entropy_loss_and_accuracy`);
- настройки модели (`ModelConfig`).

Никаких зависимостей на Hydra, датасеты, тренер и т.п. здесь нет —
модуль полностью самодостаточен внутри `pytorch_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(unsafe_hash=True, eq=True)
class ModelConfig:
    """
    Минимальная конфигурация модели, совместимая по полям с исходной.

    Здесь оставлены только те поля, которые реально используются
    архитектурой и нашим кодом.
    """

    class SeqModelingBlockType(StrEnum):
        self_attention = "self_attention"

    name: str = "unnamed"
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12

    # Параметры длины последовательности / окна (для RoPE и SWA, если понадобится)
    mini_batch_size: int = 1024
    sliding_window_size: int = 1024
    seq_len: int = 131072

    # Нормализации и инициализация
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # Токены BOS/EOS
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Dropout
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0

    # Привязка головы к эмбеддингам
    tie_word_embeddings: bool = False

    # Тип блока последовательности
    seq_modeling_block: str = "self_attention"

    # RoPE
    rope_theta: float = 10000.0

    # Выходной размер (обычно = vocab_size)
    output_size: int = 32000

    # Типы чисел
    compute_dtype: str = "bf16"
    param_dtype: str = "fp32"
    state_dtype: str = "fp32"

    # Префикс/суффикс и prime-блоки (для TTT)
    suffix_len: int = 0
    prime: bool = False
    qk_norm: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    feed_forward_prime: str = "swiglu"


# ──────────────────────────────────────────────────────────────────────────────
# Batch и вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Batch:
    """
    Минимальный контейнер батча, совместимый с архитектурой.
    """

    input_ids: torch.Tensor
    target_tokens: torch.Tensor
    loss_masks: torch.Tensor
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    index: int | slice | None = None

    @property
    def shape(self):
        return self.input_ids.shape

    def slice_index(self, index: int | slice) -> "Batch":
        return Batch(
            input_ids=self.input_ids[index],
            target_tokens=self.target_tokens[index],
            loss_masks=self.loss_masks[index],
            attention_mask=self.attention_mask[index] if self.attention_mask is not None else None,
            position_ids=self.position_ids[index] if self.position_ids is not None else None,
            index=index,
        )


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "float64": torch.float64,
}


def get_torch_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


def promote_dtype(*tensors: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    """Привести все тензоры к одному dtype (аналог jax-помощника)."""
    return [t.to(dtype) for t in tensors]


# ──────────────────────────────────────────────────────────────────────────────
# RoPE и линейный слой
# ──────────────────────────────────────────────────────────────────────────────


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute RoPE complex frequencies → [end, dim//2] complex tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: dim // 2] / dim))
    t = torch.arange(end, dtype=dtype)
    freqs = torch.outer(t, freqs)  # [end, dim//2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x.
      x:         [..., T, head_dim]
      freqs_cis: [T, head_dim//2]  complex
    """
    input_dtype = x.dtype
    x_f = x.float().reshape(*x.shape[:-1], -1, 2)
    x_c = torch.view_as_complex(x_f)          # [..., T, head_dim//2]
    shape = [1] * (x_c.ndim - 2) + list(freqs_cis.shape)
    x_out = torch.view_as_real(x_c * freqs_cis.view(*shape)).reshape(*x.shape)
    return x_out.to(input_dtype)


class NormalLinear(nn.Module):
    """Weight-only linear layer with normal initialization (no bias)."""

    def __init__(
        self,
        config: ModelConfig,
        in_features: int,
        out_features: int,
        *,
        name: str = "",
        std: float,
    ):
        super().__init__()
        self.compute_dtype = get_torch_dtype(config.compute_dtype)
        self.param_dtype = get_torch_dtype(config.param_dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = name

        weight = torch.empty(in_features, out_features, dtype=self.param_dtype)
        nn.init.normal_(weight, std=std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, weight = promote_dtype(x, self.weight, dtype=self.compute_dtype)
        return x @ weight


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────


class AttentionBase(nn.Module):
    """Базовый класс для разных вариантов attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)
        self.param_dtype = get_torch_dtype(config.param_dtype)

        embed_dim = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = embed_dim // self.num_heads

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        for attr_name in ("wq", "wk", "wv", "wo"):
            setattr(
                self,
                attr_name,
                NormalLinear(
                    config,
                    in_features=embed_dim,
                    out_features=embed_dim,
                    std=config.initializer_range,
                    name=attr_name,
                ),
            )

        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        # Pre-computed RoPE table (необучаемый буфер).
        self.register_buffer(
            "_freqs_cis",
            precompute_freqs_cis(self.head_dim, 2 * config.seq_len, theta=config.rope_theta),
            persistent=False,
        )

    @property
    def freqs_cis(self) -> torch.Tensor:
        return self._freqs_cis

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, T, D] -> [B, T, num_heads, head_dim]
        """
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[T, num_heads, head_dim] -> [T, D]"""
        return x.reshape(x.shape[0], self.num_heads * self.head_dim)

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

    def apply_rope(
        self,
        xis: tuple[torch.Tensor, ...],
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        freqs = self.freqs_cis[position_ids]
        return tuple(apply_rotary_emb(x, freqs) for x in xis)

    def get_attention_input(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xq, xk, xv = self.project_qkv(hidden_states)
        xq, xk, xv = self._split_heads(xq), self._split_heads(xk), self._split_heads(xv)
        if self.config.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq, xk = self.apply_rope((xq, xk), position_ids=position_ids)
        return xq, xk, xv

    def get_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.wo(attn_output))

    def core_attention_op(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        xq/xk/xv: [T_q or T_k, num_heads, head_dim]
        attention_mask: [T_q, T_k] bool (True = attend), or None → causal
        """
        if self.config.attn_pdrop > 0.0:
            raise ValueError("attn_pdrop > 0 not implemented")

        # SDPA ожидает [B, nh, T, hd]
        xq_ = xq.permute(0, 2, 1, 3)
        xk_ = xk.permute(0, 2, 1, 3)
        xv_ = xv.permute(0, 2, 1, 3)

        if attention_mask is not None:
            attn_bias = torch.zeros(
                1, 1, xq_.shape[2], xk_.shape[2], dtype=xq_.dtype, device=xq_.device
            )
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            out = F.scaled_dot_product_attention(xq_, xk_, xv_, attn_mask=attn_bias)
        else:
            out = F.scaled_dot_product_attention(xq_, xk_, xv_, is_causal=True)

        return self._merge_heads(out.permute(0, 2, 1, 3))

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Attention(AttentionBase):
    """Полный каузальный attention (как в исходном коде)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq: Batch,
        state,
        is_prefix: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        T = hidden_states.shape[0]
        position_ids = (
            torch.arange(T, device=hidden_states.device)
            if seq.position_ids is None
            else seq.position_ids
        )
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)

        xq_ = xq.permute(0, 2, 1, 3)
        xk_ = xk.permute(0, 2, 1, 3)
        xv_ = xv.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(xq_, xk_, xv_, is_causal=True)
        attn_out = self._merge_heads(attn_out.permute(0, 2, 1, 3).squeeze(0))
        return self.get_attention_output(attn_out), state


# ──────────────────────────────────────────────────────────────────────────────
# Transformer backbone
# ──────────────────────────────────────────────────────────────────────────────


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward блок."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.w1 = NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            std=config.initializer_range,
            name="w1",
        )
        self.w2 = NormalLinear(
            config,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            std=config.initializer_range,
            name="w2",
        )
        self.w3 = NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            std=config.initializer_range,
            name="w3",
        )
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1_act = F.silu(self.w1(x))
        x2 = z1_act * self.w3(x)
        return self.dropout(self.w2(x2))


class PrimeStorage(nn.Module):
    """
    Хранит стек prime‑FFN слоёв (по одному на каждую suffix‑позицию).
    Используется для TTT в suffix‑блоках.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.feed_forward_prime != "swiglu":
            raise NotImplementedError("Only feed_forward_prime='swiglu' is supported.")
        suffix_len = config.suffix_len
        self.feed_forward_prime = nn.ModuleList(
            [SwiGLUMLP(config) for _ in range(suffix_len)]
        )
        self.ffn_prime_norm = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(suffix_len)]
        )
        self.ffn_prime_post_norm = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(suffix_len)]
        )

    def forward(self):
        # Хранит параметры, но сам по себе не вызывается.
        pass


class Block(nn.Module):
    """
    Один transformer‑блок:
      - attention (self‑attention);
      - опциональный prime‑FFN (suffix‑блоки);
      - основной FFN.
    """

    def __init__(
        self,
        config: ModelConfig,
        feed_forward_prime: SwiGLUMLP | None = None,
        ffn_prime_norm: nn.RMSNorm | None = None,
        ffn_prime_post_norm: nn.RMSNorm | None = None,
    ):
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)

        # В этой упрощённой версии поддерживаем только self‑attention.
        if config.seq_modeling_block != "self_attention":
            raise NotImplementedError(
                f"Sequence Modeling Layer {config.seq_modeling_block} Not Implemented."
            )

        self.seq_modeling_block: AttentionBase = Attention(config)
        self.feed_forward = SwiGLUMLP(config)

        self.seq_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.seq_post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Optional prime FFN (injectable из PrimeStorage для suffix‑блоков).
        self.ffn_prime_norm: nn.RMSNorm | None = ffn_prime_norm
        self.ffn_prime_post_norm: nn.RMSNorm | None = ffn_prime_post_norm
        self.feed_forward_prime: SwiGLUMLP | None = feed_forward_prime

    def seq_modeling_forward(
        self,
        hidden_states: torch.Tensor,
        state,
        seq: Batch,
        is_prefix: bool,
    ) -> tuple[torch.Tensor, Any]:
        inp = self.seq_norm(hidden_states) if self.config.pre_norm else hidden_states
        out, state = self.seq_modeling_block(inp, seq, state, is_prefix=is_prefix)
        if self.config.post_norm:
            out = self.seq_post_norm(out)
        return out, state

    def ffn_forward(
        self,
        hidden_states: torch.Tensor,
        ffn_norm: nn.RMSNorm,
        feed_forward: SwiGLUMLP,
        ffn_post_norm: nn.RMSNorm,
    ) -> torch.Tensor:
        inp = ffn_norm(hidden_states) if self.config.pre_norm else hidden_states
        out = feed_forward(inp)
        if self.config.post_norm:
            out = ffn_post_norm(out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        state,
        seq: Batch,
        is_prefix: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        # 1. Блок attention.
        seq_out, state = self.seq_modeling_forward(hidden_states, state, seq, is_prefix=is_prefix)
        hidden_states = hidden_states + seq_out

        # 2. Опциональный prime‑FFN (для suffix‑блоков).
        if self.feed_forward_prime is not None:
            prime_out = self.ffn_forward(
                hidden_states,
                self.ffn_prime_norm,
                self.feed_forward_prime,
                self.ffn_prime_post_norm,
            )
            hidden_states = hidden_states + prime_out

        # 3. Основной FFN.
        ffn_out = self.ffn_forward(
            hidden_states, self.ffn_norm, self.feed_forward, self.ffn_post_norm
        )
        hidden_states = hidden_states + ffn_out

        return hidden_states, state


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    state: list | None = None


class BlockCollection(nn.Module):
    """Плоский список блоков трансформера (без разделения на prefix/suffix)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.prime_storage: PrimeStorage | None = (
            PrimeStorage(config) if config.prime else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: list,
        seq: Batch,
    ) -> BaseModelOutput:
        new_states = []
        for i, block in enumerate(self.blocks):
            sub = state[i] if (state is not None and i < len(state)) else None
            hidden_states, sub = block(hidden_states, sub, seq)
            new_states.append(sub)
        return BaseModelOutput(last_hidden_state=hidden_states, state=new_states)


class TransformerModel(nn.Module):
    """Backbone‑часть: эмбеддинги + стек блоков + финальный ln_f."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.wte.weight, std=config.initializer_range)

        self.dropout = nn.Dropout(p=config.embd_pdrop)
        self.h: BlockCollection = BlockCollection(config)
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def wte_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.wte(input_ids.long()).to(self.compute_dtype)
        return self.dropout(emb)

    def forward(self, state, seq: Batch) -> BaseModelOutput:
        hidden_states = self.wte_call(seq.input_ids)
        outputs: BaseModelOutput = self.h(hidden_states, state=state, seq=seq)
        return BaseModelOutput(
            last_hidden_state=self.ln_f(outputs.last_hidden_state),
            state=outputs.state,
        )


@dataclass
class CausalLMOutput:
    last_hidden_states: torch.Tensor
    logits: torch.Tensor
    new_state: Any


class CausalLM(nn.Module):
    """
    Language‑model над TransformerModel:
      - добавляет lm_head;
      - умеет считать логиты по скрытым состояниям.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)
        self.model = TransformerModel(config)
        self.lm_head: NormalLinear | None = (
            None
            if config.tie_word_embeddings
            else NormalLinear(
                config,
                in_features=config.hidden_size,
                out_features=config.output_size,
                std=config.initializer_range,
                name="lm_head",
            )
        )

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            hs, kernel = promote_dtype(hidden_states, self.model.wte.weight.T, dtype=self.compute_dtype)
            return hs @ kernel
        return self.lm_head(hidden_states)

    def wte_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.wte_call(input_ids)

    def forward(self, state, seq: Batch) -> CausalLMOutput:
        outputs = self.model(state, seq)
        hs = outputs.last_hidden_state
        assert hs.dtype == self.compute_dtype, (
            "hidden_states before lm_head should be in compute_dtype"
        )
        return CausalLMOutput(
            last_hidden_states=hs,
            logits=self._compute_logits(hs),
            new_state=outputs.state,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────


def cross_entropy_loss_and_accuracy(
    logits: torch.Tensor,              # [B, T, vocab_size]
    tokens: torch.Tensor,              # [B, T]
    valid: torch.Tensor | None = None, # [B, T] float mask
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cross‑entropy loss поверх логитов LM.

    Это почти точная копия оригинальной функции из `transformer.py`,
    только с учётом батчевого измерения.
    """
    if valid is None:
        valid = torch.ones(tokens.shape, dtype=torch.float32, device=tokens.device)
    valid = valid.float()
    valid_text_length = torch.clamp(valid.sum(dim=-1), min=1e-10)

    logits = logits.float()
    log_prob = F.log_softmax(logits, dim=-1)
    token_log_prob = log_prob.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1)
    token_log_prob = torch.where(valid > 0.0, token_log_prob, torch.zeros_like(token_log_prob))

    token_wise_loss = -token_log_prob
    loss_pure_ce = (token_wise_loss.sum(dim=-1) / valid_text_length).mean()
    loss = loss_pure_ce  # идентично оригиналу
    return loss, loss_pure_ce


def token_log_probs(
    logits: torch.Tensor,    # [B, T, vocab_size]
    targets: torch.Tensor,   # [B, T]
) -> torch.Tensor:
    """
    Лог‑вероятности целевых токенов по логитам модели.
    """
    return (
        F.log_softmax(logits, dim=-1)
        .gather(-1, targets.long().unsqueeze(-1))
        .squeeze(-1)
    )


