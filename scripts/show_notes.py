# Просмотр содержимого базы Chroma (только чтение, ничего не меняет).
# Запуск из корня проекта:
#   python scripts/show_notes.py                       база бота (data/chroma)
#   python scripts/show_notes.py data/chroma_notebook  база ноутбука
import os
import sys
from collections import Counter

import chromadb

sys.path.insert(0, os.getcwd())
from app.journal import hash_user

path = sys.argv[1] if len(sys.argv) > 1 else "data/chroma"
if not os.path.exists(path):
    print("папки нет:", path)
    sys.exit()

client = chromadb.PersistentClient(path=path)
collections = client.list_collections()
print("база:", path, "| коллекций:", len(collections))

for c in collections:
    col = client.get_collection(c.name)
    data = col.get(include=["documents", "metadatas"])
    print(f"\n=== коллекция {col.name}: заметок {len(data['ids'])}")

    users = Counter(m.get("user_id") for m in data["metadatas"])
    print("пользователей:", len(users), "| заметок у каждого:",
          {hash_user(u): n for u, n in users.items()})

    rows = sorted(zip(data["metadatas"], data["documents"]),
                  key=lambda x: x[0].get("created", ""))
    for m, text in rows:
        mark = "[" + m.get("file_type", "") + "] " if m.get("file_id") else ""
        print(f"  {m.get('created', '')[:16]}  {hash_user(m.get('user_id'))}  {mark}{text[:80]}")