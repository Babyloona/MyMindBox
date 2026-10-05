# Проверка окружения: ключи, связь с OpenAI и Langfuse, трейс одного запроса.
# Запуск из корня проекта: python scripts/check_env.py
import os
import sys
import time

sys.path.insert(0, os.getcwd())
# Отдельная пустая база, чтобы проверка не трогала настоящие заметки
os.environ["CHROMA_DIR"] = "data/chroma_check"

from dotenv import load_dotenv
load_dotenv()

print("1. Ключи в .env:")
for name in ["OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TELEGRAM_BOT_TOKEN"]:
    print("  ", name, "есть" if os.getenv(name) else "НЕТ")
print("   модель:", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

print("\n2. OpenAI:")
from openai import OpenAI
oa = OpenAI()
r = oa.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0,
    messages=[{"role": "user", "content": "Ответь одним словом: ок"}])
print("   ответ:", r.choices[0].message.content,
      "| токены:", r.usage.prompt_tokens, "вход,", r.usage.completion_tokens, "выход")

print("\n3. Langfuse:")
from langfuse import get_client
lf = get_client()
print("   auth_check:", lf.auth_check())
print("   create_score есть:", hasattr(lf, "create_score"))
print("   api.trace.list есть:", hasattr(lf, "api") and hasattr(lf.api, "trace"))

print("\n4. Один запрос через граф с трейсингом:")
from langfuse.langchain import CallbackHandler
from app.graph import graph
handler = CallbackHandler()
state = {"user_id": "check", "text": "где зарядка?", "file_id": "",
         "file_type": "text", "steps": []}
out = graph.invoke(state, config={
    "callbacks": [handler], "run_name": "mindbox_check",
    "metadata": {"langfuse_session_id": "check", "langfuse_user_id": "check",
                 "langfuse_tags": ["check"]}})
print("   путь:", " -> ".join(out["steps"]))
print("   ответ:", out["answer"])
print("   last_trace_id:", getattr(handler, "last_trace_id", "АТРИБУТА НЕТ"))
lf.flush()

print("\n5. Чтение трейса обратно (новый API):")
from app.tracing import fetch_observations
trace_id = getattr(handler, "last_trace_id", None)
obs = fetch_observations(trace_id=trace_id)
print("   шагов в трейсе:", len(obs))
for o in sorted(obs, key=lambda x: x.get("startTime", "")):
    print(f"   {o.get('type', ''):10s} {str(o.get('name', ''))[:30]:30s}"
          f" | {o.get('latency')} | стоимость {o.get('totalCost')}"
                   f" | токены {o.get('inputTokens', o.get('usageDetails'))}/{o.get('outputTokens')}")
if obs:
    print("   поля одного шага:", sorted(obs[0].keys()))