FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Сначала лёгкая версия torch без видеокарты: образ получается в разы меньше
RUN pip install --upgrade pip \
 && pip install typing_extensions filelock sympy networkx jinja2 fsspec \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Модель эмбеддингов скачиваем при сборке, чтобы контейнер стартовал без интернета
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app ./app
COPY bot ./bot

# Запуск не от root
RUN useradd --create-home botuser && mkdir -p /app/data && chown -R botuser /app /models
USER botuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

CMD ["python", "-m", "bot.bot"]