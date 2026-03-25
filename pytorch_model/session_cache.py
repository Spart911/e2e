from __future__ import annotations

import io
import time
from contextlib import contextmanager
from typing import Any, Iterator

import redis
import torch


class RedisTTTSessionCache:
    """
    Кэш для сессионных TTT-весов в Redis.

    Храним:
      - только inner_state (а не весь state_dict модели);
      - checkpoint_id/model_revision для защиты от смешивания кэша разных версий модели;
      - Redis-lock по session_id для защиты от race condition.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_sec: int = 3600,
        prefix: str = "ttt:session",
        *,
        lock_ttl_sec: int = 30,
        lock_blocking_timeout_sec: float = 10.0,
        lock_sleep_sec: float = 0.1,
    ) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.ttl_sec = ttl_sec
        self.prefix = prefix
        self.lock_ttl_sec = lock_ttl_sec
        self.lock_blocking_timeout_sec = lock_blocking_timeout_sec
        self.lock_sleep_sec = lock_sleep_sec

    def _weights_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:weights"

    def _meta_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:meta"

    # --- отдельный lock key ---
    def _lock_key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}:lock"

    # --- lock context manager ---
    @contextmanager
    def session_lock(
        self,
        session_id: str,
        *,
        timeout_sec: int | None = None,
        blocking_timeout_sec: float | None = None,
    ) -> Iterator[None]:
        """
        Распределённая блокировка на session_id.

        Нужна, чтобы два параллельных запроса одной сессии не выполнили:
          load -> adapt -> save
        одновременно и не перетёрли состояние друг друга.
        """
        lock = self.client.lock(
            self._lock_key(session_id),
            timeout=timeout_sec or self.lock_ttl_sec,
            sleep=self.lock_sleep_sec,
            blocking_timeout=(
                self.lock_blocking_timeout_sec
                if blocking_timeout_sec is None
                else blocking_timeout_sec
            ),
            thread_local=False,
            raise_on_release_error=False,
        )

        acquired = lock.acquire(blocking=True)
        if not acquired:
            raise TimeoutError(f"Не удалось получить Redis-lock для session_id={session_id}")

        try:
            yield
        finally:
            try:
                lock.release()
            except redis.exceptions.LockError:
                # Лок уже мог истечь или быть освобождён ранее
                pass

    def save_inner_state(
        self,
        session_id: str,
        inner_state: dict[str, torch.Tensor],
        *,
        checkpoint_id: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "inner_state": {k: v.detach().cpu() for k, v in inner_state.items()},
            "checkpoint_id": checkpoint_id,
            "saved_at": time.time(),
        }
        if extra_meta:
            payload["extra_meta"] = extra_meta

        buf = io.BytesIO()
        torch.save(payload, buf)
        blob = buf.getvalue()

        meta = {
            "checkpoint_id": checkpoint_id.encode("utf-8"),
            "saved_at": str(payload["saved_at"]).encode("utf-8"),
            "param_count": str(len(inner_state)).encode("utf-8"),
        }
        if extra_meta:
            for key, value in extra_meta.items():
                meta[f"extra:{key}"] = str(value).encode("utf-8")

        pipe = self.client.pipeline()
        pipe.set(self._weights_key(session_id), blob, ex=self.ttl_sec)
        pipe.hset(self._meta_key(session_id), mapping=meta)
        pipe.expire(self._meta_key(session_id), self.ttl_sec)
        pipe.execute()

    def load_inner_state(
        self,
        session_id: str,
        *,
        checkpoint_id: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        blob = self.client.get(self._weights_key(session_id))
        if blob is None:
            return None

        payload = torch.load(io.BytesIO(blob), map_location="cpu")

        cached_checkpoint_id = payload.get("checkpoint_id")
        if cached_checkpoint_id != checkpoint_id:
            self.delete_session(session_id)
            return None

        # Продлеваем TTL при обращении
        self.client.expire(self._weights_key(session_id), self.ttl_sec)
        self.client.expire(self._meta_key(session_id), self.ttl_sec)

        return {
            k: v.to(device)
            for k, v in payload["inner_state"].items()
        }

    # --- для отладки ---
    def get_meta(self, session_id: str) -> dict[bytes, bytes]:
        return self.client.hgetall(self._meta_key(session_id))

    def delete_session(self, session_id: str) -> None:
        self.client.delete(
            self._weights_key(session_id),
            self._meta_key(session_id),
        )