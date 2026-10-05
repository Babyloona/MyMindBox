import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from app import config
from app.llm import get_llm
from app.database import save_note, search_notes
import time
from app.journal import write_event, hash_user
from app.tracing import get_trace_config
import re

# ---------- Состояние: что передаётся между узлами ----------

class State(TypedDict, total=False):
    user_id: str
    text: str
    file_id: str
    file_type: str
    intent: str          # save / search / search_all
    results: list        # найденные заметки
    status: str          # found / maybe / not_found / saved / need_caption
    answer: str          # текст ответа пользователю
    files: list          # что приложить к ответу (file_id и тип)
    steps: Annotated[list, operator.add]   # история прохода по узлам
    truncated: bool


# ---------- Узел 1: определяем намерение (единственный вызов LLM) ----------

CLASSIFY_PROMPT = """Ты помощник личной памяти. Человек пишет сообщение.
Определи, что он хочет, и ответь ОДНИМ словом:

save        записать факт: где что лежит, дело, идею, рецепт (утверждение),
            в том числе «сохрани ...», «запомни ...», «запиши ...»
search      найти или прислать ОДНУ конкретную вещь или заметку
search_all  прислать ВСЕ или несколько заметок: «все», «список», «всё про...»,
            а также общие вопросы про целую категорию: «что мне надо сделать»,
            «какие у меня дела», «что я хотела купить»

Просьба прислать или показать («скинь», «покажи», «дай», «напомни») это поиск,
а не запись, даже без вопросительного знака.
Сообщение может быть с опечатками («скиьн», «покжи», «гед»): понимай его по смыслу.

Сообщение: {text}
Ответ одним словом:"""


def classify(state):
    prompt = CLASSIFY_PROMPT.format(text=state["text"])
    raw = get_llm().invoke(prompt).content
    raw = (raw or "").strip().lower()

    word = ""
    if raw:
        word = raw.split()[0].strip(".,!\"'")

    # Если модель ответила что-то странное, берём простое правило
    if word not in ("save", "search", "search_all"):
        if state["text"].strip().endswith("?"):
            word = "search"
        else:
            word = "save"

    return {"intent": word, "steps": ["classify: " + word]}


# ---------- Узел 2: сохранение ----------
STOP_WORDS = ["да", "нет", "ок", "окей", "ага", "угу", "спасибо",
              "хорошо", "ладно", "понятно"]


def save(state):
    text = state.get("text", "").strip()

    # Без подписи файл потом не найти: поиск идёт только по тексту
    if text == "":
        return {
            "status": "need_caption",
            "answer": "Добавьте подпись к файлу, иначе потом я не смогу его найти.",
            "steps": ["save: нет подписи"],
        }

    # Короткие реплики вроде «да» или «спасибо» не заметки
    if state.get("file_id", "") == "":
        cleaned = text.lower().strip(".!? ")
        if cleaned in STOP_WORDS:
            return {
                "status": "skipped",
                "answer": "Это не похоже на заметку, поэтому не сохраняю. "
                          "Напишите факт полностью или задайте вопрос.",
                "steps": ["save: короткая реплика, пропущено"],
            }
    
    # Почти такая же заметка уже есть: не плодим дубли (только для текста без файла)
    if state.get("file_id", "") == "":
        same = search_notes(state["user_id"], text, top_k=1)
        if same and same[0]["score"] >= config.DUPLICATE_SCORE:
            return {
                "status": "duplicate",
                "answer": "Уже записано: " + same[0]["text"],
                "steps": ["save: дубль, не сохранено"],
            }
        
    save_note(
        state["user_id"],
        text,
        file_id=state.get("file_id", ""),
        file_type=state.get("file_type", "text"),
    )
    return {
        "status": "saved",
        "answer": "Сохранено: " + text,
        "steps": ["save: сохранено"],
    }


# ---------- Узел 3: поиск в базе ----------

def retrieve(state):
    if state["intent"] == "search_all":
        top_k = config.SEARCH_ALL_TOP_K
    else:
        top_k = 3

    results = search_notes(state["user_id"], state["text"], top_k=top_k)
    return {"results": results, "steps": ["retrieve: найдено " + str(len(results))]}


# ---------- Узел 4: оценка найденного по порогам (без LLM) ----------

def grade(state):
    results = state.get("results", [])

    # Для «покажи все» предфильтр мягче, для одиночного поиска строже
    if state["intent"] == "search_all":
        min_score = config.THRESHOLD_ALL
    else:
        min_score = config.THRESHOLD_MAYBE

    candidates = []
    for r in results:
        if r["score"] >= min_score:
            candidates.append(r)

    # Все K найденных прошли предфильтр: за пределом выдачи, скорее всего, есть ещё
    truncated = (state["intent"] == "search_all"
                 and len(candidates) >= config.SEARCH_ALL_TOP_K)

    if state["intent"] == "search":
        candidates = candidates[:config.VERIFY_TOP_K] 

    if len(candidates) == 0:
        return {"status": "not_found", "results": [],
                "steps": ["grade: ниже порога, LLM не нужна"]}

    return {"status": "candidates", "results": candidates, "truncated": truncated,
            "steps": ["grade: кандидатов " + str(len(candidates))]}


VERIFY_ONE_PROMPT = """Вопрос человека: {question}
Заметка из его личной базы: {note}

Отвечает ли эта заметка на вопрос? Если заметка про другой предмет, другое место
или другое событие, это «нет». Например, заметка про зонт не отвечает на вопрос про очки.
Ответь одним словом: да или нет."""


VERIFY_ALL_PROMPT = """Вопрос человека: {question}
Заметки из его личной базы:
{notes}

Какие из этих заметок подходят под вопрос? Заметка про другой предмет не подходит.
Если вопрос про целую категорию (дела, покупки, рецепты, документы), подходят
все заметки этой категории. Например, на вопрос «что мне надо сделать» подходят
все заметки с делами и задачами, даже если в них нет слова «дело».
Ответь номерами через запятую. Если ни одна не подходит, ответь: нет."""



VERIFY_PICK_PROMPT = """Вопрос человека: {question}
Заметки из его личной базы:
{notes}

Какая ОДНА заметка отвечает на вопрос? Заметка про другой предмет, другое место
или другое событие не подходит. Например, заметка про зонт не отвечает на вопрос про очки.
Ответь одним номером. Если ни одна не подходит, ответь: нет."""

def verify(state):
    question = state["text"]
    candidates = state["results"]
    llm = get_llm()

    approved = []

    if state["intent"] == "search" and len(candidates) == 1:
        note = candidates[0]
        prompt = VERIFY_ONE_PROMPT.format(question=question, note=note["text"])
        raw = (llm.invoke(prompt).content or "").strip().lower()
        if raw.startswith("да"):
            approved = [note]
    elif state["intent"] == "search":
        # Несколько кандидатов: LLM выбирает одну подходящую (по-прежнему один вызов)
        lines = []
        for i in range(len(candidates)):
            lines.append(str(i + 1) + ". " + candidates[i]["text"])
        prompt = VERIFY_PICK_PROMPT.format(question=question, notes="\n".join(lines))
        raw = (llm.invoke(prompt).content or "").strip().lower()
        first = raw.replace(".", " ").replace(",", " ").split()
        if first and first[0].isdigit():
            number = int(first[0])
            if 1 <= number <= len(candidates):
                approved = [candidates[number - 1]]
    else:
        lines = []
        for i in range(len(candidates)):
            lines.append(str(i + 1) + ". " + candidates[i]["text"])
        prompt = VERIFY_ALL_PROMPT.format(question=question, notes="\n".join(lines))
        raw = (llm.invoke(prompt).content or "").strip().lower()
        raw = raw.replace(",", " ").replace(".", " ")
        numbers = raw.split()
        for i in range(len(candidates)):
            if str(i + 1) in numbers:
                approved.append(candidates[i])

    if len(approved) == 0:
        return {"status": "not_found", "results": [],
                "steps": ["verify: LLM отклонила"]}

    if state["intent"] == "search":
        if approved[0]["score"] >= config.THRESHOLD_FOUND:
            status = "found"
        else:
            status = "maybe"
    else:
        approved = sorted(approved, key=lambda r: r["created"])
        status = "found"

    return {"status": status, "results": approved,
            "steps": ["verify: подтверждено " + str(len(approved)) + ", " + status]}


# ---------- Узлы 5-7: три варианта ответа (лестница вмешательства) ----------

def make_files(results):
    files = []
    for r in results:
        files.append({"text": r["text"], "file_id": r["file_id"], "file_type": r["file_type"]})
    return files


def reply_found(state):
    results = state["results"]
    if len(results) == 1:
        answer = "Нашлось: " + results[0]["text"]
    else:
        lines = []
        for i in range(len(results)):
            lines.append(str(i + 1) + ". " + results[i]["text"])
        answer = "Нашлось заметок: " + str(len(results)) + "\n" + "\n".join(lines)
    
    if state.get("truncated"):
        answer = answer + ("\n\nПоказаны лучшие совпадения из первых "
                           + str(config.SEARCH_ALL_TOP_K)
                           + " найденных. Возможно, есть ещё: уточните запрос.")
        
    return {"answer": answer, "files": make_files(results),
            "steps": ["reply_found"]}


def reply_maybe(state):
    r = state["results"][0]
    answer = "Не уверен, но возможно, это оно: " + r["text"] + "\nЕсли не то, уточните вопрос."
    return {"answer": answer, "files": make_files(state["results"]),
            "steps": ["reply_maybe"]}


def reply_not_found(state):
    return {"answer": "Такого не записано.", "files": [],
            "steps": ["reply_not_found"]}


# ---------- Развилки: куда идти дальше ----------

def route_start(state):
    # Пришёл файл: это точно сохранение, LLM не нужна
    if state.get("file_id"):
        return "save"
    return "classify"


def route_intent(state):
    if state["intent"] == "save":
        return "save"
    return "retrieve"


def route_grade(state):
    return state["status"]

def route_after_grade(state):
    if state["status"] == "not_found":
        return "not_found"
    return "verify"
# ---------- Сборка графа ----------

def build_graph():
    g = StateGraph(State)

    g.add_node("classify", classify)
    g.add_node("save", save)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("verify", verify)
    g.add_node("reply_found", reply_found)
    g.add_node("reply_maybe", reply_maybe)
    g.add_node("reply_not_found", reply_not_found)

    g.add_conditional_edges(START, route_start,
                            {"save": "save", "classify": "classify"})
    g.add_conditional_edges("classify", route_intent,
                            {"save": "save", "retrieve": "retrieve"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"verify": "verify", "not_found": "reply_not_found"})
    g.add_conditional_edges("verify", route_grade,
                            {"found": "reply_found",
                             "maybe": "reply_maybe",
                             "not_found": "reply_not_found"})

    g.add_edge("save", END)
    g.add_edge("reply_found", END)
    g.add_edge("reply_maybe", END)
    g.add_edge("reply_not_found", END)

    return g.compile()


graph = build_graph()

# ---------- Защита секретов: до графа, до LLM, до трекера и журнала ----------

SECRET_PATTERNS = [
    r"парол", r"\bпин\b", r"пин-код", r"\bpin\b", r"\bcvv\b", r"\bcvc\b",
    r"(?<!\d)\d{12}(?!\d)",              # ИИН: 12 цифр подряд
    r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)",   # номер карты: 16 цифр, можно через пробел
]

SECRET_ANSWER = ("Похоже, здесь секретные данные (пароль, пин-код, номер карты или ИИН). "
                 "Такое я не сохраняю и не отправляю во внешний сервис.")


def has_secret(text):
    low = (text or "").lower()
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, low):
            return True
    return False

def run(user_id, text, file_id="", file_type="text"):
    # Единая точка входа: её вызывает Telegram-бот
    
    start = time.time()
    
    # Секреты отсекаются до графа: текст не уходит ни в LLM, ни в Langfuse, ни в журнал
    if has_secret(text):
        steps = ["guard: секретные данные, граф не запускался"]
        write_event({"user": hash_user(user_id), "text": "[скрыто]", "has_file": file_id != "",
                     "intent": "blocked", "status": "blocked", "scores": [], "steps": steps,
                     "seconds": round(time.time() - start, 2)})
        return {"intent": "blocked", "status": "blocked", "answer": SECRET_ANSWER,
                "results": [], "files": [], "steps": steps}

    state = {
        "user_id": str(user_id),
        "text": text,
        "file_id": file_id,
        "file_type": file_type,
        "steps": [],
    }

    # Если в .env есть ключи Langfuse, каждый запрос уходит в трекер с метаданными
    trace_config = get_trace_config(user_id)
    if trace_config is not None:
        result = graph.invoke(state, config=trace_config)
    else:
        result = graph.invoke(state)

    seconds = round(time.time() - start, 2)

    scores = []
    for r in result.get("results", []):
        scores.append(r["score"])

    write_event({
        "user": hash_user(user_id),
        "text": text,
        "has_file": file_id != "",
        "intent": result.get("intent", "save_with_file"),
        "status": result.get("status", ""),
        "scores": scores,
        "steps": result.get("steps", []),
        "seconds": seconds,
    })

    return result