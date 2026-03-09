# PyTorch модель с TTT и чат-инференсом

Этот модуль собирает вокруг исходного `transformer.py` полный PyTorch‑проект:

- **архитектура модели** (`architecture.py`);
- **конфигурация** (`config.py`);
- **датасет и DataLoader** (`data.py`);
- **функции потерь** (`losses.py`);
- **TTT‑адаптация (Test-Time Training)** (`ttt.py`);
- **скрипт полного дообучения** (`train_full.py`);
- **чатовый инференс с TTT** (`chat_ttt.py`).

Ниже — краткий обзор того, как этим пользоваться.

---

## 1. Установка зависимостей

Необходимые пакеты (минимум):

- `torch`
- `transformers`

Пример:

```bash
pip install torch transformers
```

---

## 2. Формат данных для обучения

`train_full.py` ожидает **один текстовый файл** с обучающим корпусом:

- формат: обычный `.txt`, кодировка UTF‑8;
- содержимое: произвольный текст (диалоги, статьи, код и т.п.);
- путь передаётся в аргументе `--data_path`.

Текст токенизируется выбранным HuggingFace‑токенизатором и режется на последовательности длиной `seq_len + 1`.  
Из каждой последовательности строится стандартная задача language modeling со сдвигом на 1 токен.

---

## 3. Полное дообучение модели

Скрипт: `train_full.py`.

Пример запуска:

```bash
python -m pytorch_model.train_full \
  --checkpoint_in 125m_pytorch.pt \
  --data_path pytorch_model/data.txt \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --epochs 1 \
  --batch_size 4 \
  --seq_len 512 \
  --lr 1e-4 \
  --save_path finetuned.pt
```

Что делает:

- загружает токенизатор и модель (из исходного чекпоинта или с нуля);
- создаёт `DataLoader` из текстового файла;
- запускает несколько эпох обучения (AdamW + cross‑entropy);
- сохраняет новый чекпоинт в формате `{"model_weights": state_dict}`.

---

## 4. Чат с TTT‑адаптацией

Скрипт: `chat_ttt.py`.

Пример запуска:

```bash
python -m pytorch_model.chat_ttt \
  --checkpoint 125m_pytorch.pt \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --ttt_steps 5 \
  --ttt_lr 1e-3 \
  --max_new_tokens 200
```

Поведение:

1. Загружает токенизатор и модель из чекпоинта.
2. Запускает REPL:
   - вы вводите сообщение;
   - перед генерацией ответа модель дообучается (TTT) только на этом сообщении;
   - генерируется ответ и печатается в консоль.
3. Для выхода введите `/exit` или `/quit` (либо `Ctrl+C` / `Ctrl+D`).

---

## 5. Файлы по ролям

- `config.py` — конфигурация `ModelConfig` и выбор устройства (`cpu` / `cuda`).
- `architecture.py` — сборка `CausalLM` из `transformer.py`, генерация и вспомогательные функции.
- `data.py` — минимальный текстовый датасет и DataLoader для language modeling.
- `losses.py` — функции потерь (обёртка над `cross_entropy_loss_and_accuracy`).
- `ttt.py` — реализация Test-Time Training на подмножестве параметров (suffix‑блоки).
- `train_full.py` — обучение на тексте, сохранение нового чекпоинта.
- `chat_ttt.py` — интерактивный чат с адаптацией на каждое пользовательское сообщение.

