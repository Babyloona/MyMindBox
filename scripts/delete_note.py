import sys

from app.database import get_collection

if len(sys.argv) < 2:
    print('Использование: python -m scripts.delete_note "текст заметки"')
    raise SystemExit

target = sys.argv[1].strip().lower()

collection = get_collection()
data = collection.get()

ids_to_delete = []
for i in range(len(data["ids"])):
    if data["documents"][i].strip().lower() == target:
        ids_to_delete.append(data["ids"][i])

if len(ids_to_delete) == 0:
    print("Заметок с таким текстом не найдено.")
else:
    collection.delete(ids=ids_to_delete)
    print("Удалено заметок:", len(ids_to_delete))