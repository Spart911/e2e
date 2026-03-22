# GPU-ориентированный образ (CUDA 12.1)
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Ставим зависимости, кроме torch (уже в базе)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники
COPY . .

# Значения по умолчанию (переопределяйте при запуске)
ENV DEVICE=cuda \
    CHECKPOINT_PATH=/models/model.pt \
    TOKENIZER_NAME=meta-llama/Meta-Llama-3-8B \
    MAX_CONTEXT_TOKENS=4096 \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
