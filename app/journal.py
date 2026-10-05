import hashlib
import json
import os
from datetime import datetime

LOG_PATH = "data/logs/requests.jsonl"


def hash_user(user_id):
    # В журнале не храним настоящий id пользователя, только короткий хеш
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:8]


def write_event(event):
    # Одна строка = один запрос в формате JSON
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        event["time"] = datetime.now().isoformat(timespec="seconds")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as error:
        # Журнал не должен ронять бота
        print("Не удалось записать журнал:", error)