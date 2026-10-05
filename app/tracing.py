import logging
import os

from app import config  # noqa: F401  (подгружает .env)
from app.journal import hash_user

logger = logging.getLogger("mindbox.tracing")

_connection_checked = False
_connection_ok = False


def tracing_configured():
    # Трейсинг включён, только если в .env есть оба ключа Langfuse
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    return public != "" and secret != ""


def check_connection():
    # SDK при неверных ключах молчит и трейсы теряются. Поэтому проверяем явно.
    global _connection_checked, _connection_ok
    if not tracing_configured():
        return False
    try:
        from langfuse import get_client
        _connection_ok = bool(get_client().auth_check())
    except Exception as error:
        logger.warning("Langfuse недоступен: %s", error)
        _connection_ok = False
    _connection_checked = True
    if _connection_ok:
        logger.info("Langfuse подключён")
    else:
        logger.warning("Langfuse настроен, но связи нет: трейсы не уйдут")
    return _connection_ok


def get_trace_config(user_id, session_id=None, tags=None):
    # Возвращает config для graph.invoke или None, если трейсинг выключен
    if not tracing_configured():
        return None
    if not _connection_checked:
        check_connection()
    if not _connection_ok:
        return None

    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
    except Exception as error:
        logger.warning("Не удалось создать обработчик Langfuse: %s", error)
        return None

    user = hash_user(user_id)
    return {
        "callbacks": [handler],
        "run_name": "mindbox_request",
        "metadata": {
            "langfuse_session_id": session_id or ("user-" + user),
            "langfuse_user_id": user,
            "langfuse_tags": tags or ["mindbox", os.getenv("APP_ENV", "dev")],
        },
    }





def fetch_observations(trace_id=None, session_id=None, minutes=60, limit=100,
                       wait=90, fields="core,basic,usage"):
    # Читает шаги трейсов через новый API Langfuse (v2 observations).
    # Старый /api/public/traces закрыт для организаций, созданных после 16.09.2026.
    # Шаги доходят до трекера не одновременно, поэтому опрашиваем,
    # пока их число не перестанет расти (или пока не выйдет время wait).
    import time
    import datetime as dt
    import httpx

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    auth = (os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", ""))

    def iso(t):
        return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def request():
        now = dt.datetime.now(dt.timezone.utc)
        params = {
            "fromStartTime": iso(now - dt.timedelta(minutes=minutes)),
            "toStartTime": iso(now + dt.timedelta(minutes=1)),
            "limit": limit,
        }
        if fields:
            params["fields"] = fields
        if trace_id:
            params["traceId"] = trace_id
        if session_id:
            params["sessionId"] = session_id
        response = httpx.get(host + "/api/public/v2/observations",
                             params=params, auth=auth, timeout=30)
        if response.status_code == 400 and fields:
            # Если сервер не принял группы полей, просим без них
            params.pop("fields")
            response = httpx.get(host + "/api/public/v2/observations",
                                 params=params, auth=auth, timeout=30)
        response.raise_for_status()
        return response.json().get("data", [])

    deadline = time.time() + wait
    previous = -1
    data = []
    while time.time() < deadline:
        data = request()
        has_root = any(o.get("isRootObservation") for o in data)
        if len(data) > 0 and len(data) == previous and (has_root or not trace_id):
            return data          # два опроса подряд дали одно и то же: всё дошло
        previous = len(data)
        time.sleep(8)
    return data