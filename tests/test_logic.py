# Тесты без сети и без моделей: проверяют логику графа, пороги и разбор ответов LLM.
# Запуск: python -m pytest tests -v
import os
import tempfile

# База для тестов во временной папке, чтобы не трогать настоящие заметки
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()

from app import config
from app import graph as g
from app.journal import hash_user


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    # Подставная модель: возвращает заранее заданный ответ
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, prompt):
        return FakeMessage(self.answer)


def make_results(scores):
    results = []
    for i in range(len(scores)):
        results.append({
            "text": "заметка " + str(i + 1),
            "file_id": "",
            "file_type": "text",
            "created": "2026-01-0" + str(i + 1),
            "score": scores[i],
        })
    return results


# ---------- grade: пороги ----------

def test_grade_below_threshold_is_not_found():
    state = {"intent": "search", "results": make_results([0.15, 0.10])}
    out = g.grade(state)
    assert out["status"] == "not_found"
    assert out["results"] == []


def test_grade_search_keeps_only_best_candidate(monkeypatch):
    monkeypatch.setattr(config, "VERIFY_TOP_K", 1)   # тест не зависит от .env
    state = {"intent": "search", "results": make_results([0.55, 0.45, 0.35])}
    out = g.grade(state)
    assert out["status"] == "candidates"
    assert len(out["results"]) == 1
    assert out["results"][0]["score"] == 0.55


def test_grade_search_all_uses_softer_threshold():
    # 0.25 ниже порога одиночного поиска (0.30), но выше порога для "покажи все" (0.20)
    scores = [0.25]
    single = g.grade({"intent": "search", "results": make_results(scores)})
    many = g.grade({"intent": "search_all", "results": make_results(scores)})
    assert single["status"] == "not_found"
    assert many["status"] == "candidates"


# ---------- развилки ----------

def test_file_goes_straight_to_save():
    assert g.route_start({"file_id": "ABC"}) == "save"
    assert g.route_start({"file_id": ""}) == "classify"


def test_route_intent():
    assert g.route_intent({"intent": "save"}) == "save"
    assert g.route_intent({"intent": "search"}) == "retrieve"
    assert g.route_intent({"intent": "search_all"}) == "retrieve"


def test_route_after_grade():
    assert g.route_after_grade({"status": "not_found"}) == "not_found"
    assert g.route_after_grade({"status": "candidates"}) == "verify"


# ---------- сохранение ----------

def test_save_skips_short_replies():
    out = g.save({"user_id": "u1", "text": "Да!", "file_id": ""})
    assert out["status"] == "skipped"


def test_save_requires_caption_for_file():
    out = g.save({"user_id": "u1", "text": "  ", "file_id": "PHOTO_1", "file_type": "photo"})
    assert out["status"] == "need_caption"


# ---------- classify: запасное правило ----------

def test_classify_uses_fallback_on_garbage(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("не знаю, что сказать"))
    assert g.classify({"text": "где зарядка?"})["intent"] == "search"
    assert g.classify({"text": "провода на балконе"})["intent"] == "save"


def test_classify_reads_valid_answer(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("search_all"))
    assert g.classify({"text": "скинь все акты"})["intent"] == "search_all"


# ---------- verify: разбор ответа LLM ----------

def test_verify_one_yes_confident(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("да"))
    state = {"text": "где зарядка?", "intent": "search",
             "results": make_results([config.THRESHOLD_FOUND + 0.05])}
    assert g.verify(state)["status"] == "found"


def test_verify_one_yes_but_low_score_is_maybe(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("да"))
    state = {"text": "где зарядка?", "intent": "search",
             "results": make_results([config.THRESHOLD_MAYBE + 0.01])}
    assert g.verify(state)["status"] == "maybe"


def test_verify_one_no(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("нет"))
    state = {"text": "где ключи от гаража?", "intent": "search",
             "results": make_results([0.75])}
    assert g.verify(state)["status"] == "not_found"


def test_verify_all_picks_numbers(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("1, 3"))
    state = {"text": "скинь все акты", "intent": "search_all",
             "results": make_results([0.5, 0.4, 0.3])}
    out = g.verify(state)
    assert out["status"] == "found"
    assert len(out["results"]) == 2


def test_verify_all_none(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("нет"))
    state = {"text": "что-то", "intent": "search_all",
             "results": make_results([0.5, 0.4])}
    assert g.verify(state)["status"] == "not_found"


# ---------- журнал ----------

def test_user_id_is_hashed():
    hashed = hash_user(953288255)
    assert "953288255" not in hashed
    assert len(hashed) == 8




# ---------- проверка нескольких кандидатов (VERIFY_TOP_K) ----------

def test_grade_top_k_passes_several(monkeypatch):
    monkeypatch.setattr(config, "VERIFY_TOP_K", 3)
    out = g.grade({"intent": "search", "results": make_results([0.55, 0.45, 0.35])})
    assert len(out["results"]) == 3


def test_verify_pick_chooses_second(monkeypatch):
    monkeypatch.setattr(g, "get_llm", lambda: FakeLLM("2"))
    state = {"text": "как приготовить шарлотку", "intent": "search",
             "results": make_results([0.50, config.THRESHOLD_FOUND + 0.02, 0.31])}
    out = g.verify(state)
    assert out["status"] == "found"
    assert out["results"][0]["text"] == "заметка 2"





# ---------- защита секретов и дублей ----------

def test_secret_guard_detects():
    assert g.has_secret("мой пароль от ноутбука 949494")
    assert g.has_secret("карта 4400 4301 2345 6789")
    assert g.has_secret("ИИН 950101300123")
    assert not g.has_secret("провода в чёрном чемодане на балконе")
    assert not g.has_secret("сдать отчёт до 15 октября")


def test_secret_never_reaches_llm(monkeypatch):
    def llm_must_not_be_called():
        raise AssertionError("LLM не должна вызываться")
    monkeypatch.setattr(g, "get_llm", llm_must_not_be_called)
    monkeypatch.setattr(g, "write_event", lambda event: None)
    out = g.run("u1", "мой пин-код 1234")
    assert out["status"] == "blocked"


def test_save_skips_duplicate(monkeypatch):
    monkeypatch.setattr(g, "search_notes",
                        lambda user, text, top_k=1: [{"text": "надо починить рамку", "score": 0.99}])
    saved = []
    monkeypatch.setattr(g, "save_note", lambda *a, **k: saved.append(1))
    out = g.save({"user_id": "u1", "text": "надо починить рамку", "file_id": ""})
    assert out["status"] == "duplicate"
    assert saved == []


def test_search_all_truncated_flag(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_ALL_TOP_K", 3)
    out = g.grade({"intent": "search_all", "results": make_results([0.5, 0.4, 0.3])})
    assert out["truncated"] is True