## Инференс (чат) для `125m_ttt_e2e_pretrain_dclm_8k_1x_cc`

В проекте есть скрипт **`chat_infer.py`**, который запускает инференс в виде **интерактивного чата** (вывод идёт по токенам).

### 1) Подготовка чекпоинта

Архив `125m_ttt_e2e_pretrain_dclm_8k_1x_cc.zip` лежит в корне проекта. Распакуй его:

```bash
unzip 125m_ttt_e2e_pretrain_dclm_8k_1x_cc.zip
```

После распаковки появится папка:

```text
./125m_ttt_e2e_pretrain_dclm_8k_1x_cc/
```

Именно её нужно передавать в `--checkpoint_dir`.

### 2) Установка зависимостей

Из корня проекта:

```bash
pip install -e .
```

Проверь, что JAX видит GPU:

```bash
python -c "import jax; print(jax.devices())"
```

Ожидаемо: `[CudaDevice(id=0)]`.

### 3) Запуск чата (инференс)

Рекомендуемый запуск (с прогревом JIT):

```bash
python chat_infer.py \
  --checkpoint_dir ./125m_ttt_e2e_pretrain_dclm_8k_1x_cc \
  --model_size 125m \
  --decode_mini_batch_size 1 \
  --window_size 8192 \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.9 \
  --dry_run
```

- `--dry_run`: один короткий прогон для прогрева JAX/XLA (первый раз может быть долго).
- `--decode_mini_batch_size 1`: инкрементальный декодинг по 1 токену (используется SWA‑кеш, нагрузка уходит на GPU).
- `--window_size`: окно внимания (можно уменьшить, если нужно быстрее/дешевле по памяти).

### 4) Как пользоваться

- Ввод: строка `Вы:`
- Выход: `exit` или `выход`