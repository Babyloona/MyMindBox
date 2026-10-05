import json

LOG_PATH = "data/logs/requests.jsonl"

events = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line != "":
            events.append(json.loads(line))

print("Всего запросов:", len(events))
print()

# Сколько запросов по каждому намерению
intents = {}
for e in events:
    name = e["intent"]
    intents[name] = intents.get(name, 0) + 1
print("По намерению:")
for name in intents:
    print("  ", name, ":", intents[name])
print()

# Чем закончились запросы
statuses = {}
for e in events:
    name = e["status"]
    statuses[name] = statuses.get(name, 0) + 1
print("По результату:")
for name in statuses:
    print("  ", name, ":", statuses[name])
print()

# Время ответа
times = []
for e in events:
    times.append(e["seconds"])
times.sort()
average = sum(times) / len(times)
median = times[len(times) // 2]
print("Время ответа, секунды: среднее", round(average, 2),
      ", медиана", median, ", максимум", times[-1])
print()

# Последние запросы без ответа: их стоит разобрать вручную
print("Запросы, на которые бот ответил «не записано»:")
for e in events:
    if e["status"] == "not_found":
        print("  ", e["text"], "| баллы:", e["scores"])