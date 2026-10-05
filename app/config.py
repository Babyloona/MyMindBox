import os
from dotenv import load_dotenv

load_dotenv()

# Какие эмбеддинги использовать: "local" или "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Какая LLM отвечает за определение намерения: "openai" или "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Где хранится база Chroma
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")

# Пороги сходства (начальные значения, потом подберём на оценочном наборе)
THRESHOLD_FOUND = 0.40
THRESHOLD_MAYBE = 0.30
THRESHOLD_ALL = 0.20


# Сколько лучших кандидатов передавать на проверку LLM при поиске одной заметки.
# 1: проверяется только лучший по сходству (исходная ступень 4).
# 3: LLM выбирает подходящую из трёх (улучшение из раздела 9 ноутбука).
VERIFY_TOP_K = int(os.getenv("VERIFY_TOP_K", "1"))
SEARCH_ALL_TOP_K = int(os.getenv("SEARCH_ALL_TOP_K", "10"))

# Сходство, начиная с которого новая заметка считается дублем уже записанной
DUPLICATE_SCORE = 0.95