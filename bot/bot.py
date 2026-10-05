import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import Message
from aiohttp import web
from dotenv import load_dotenv

from app.embeddings import embed_text
from app.graph import run
from app.tracing import check_connection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mindbox")

dp = Dispatcher()

MAX_FILES_SENT = 15                      # максимум файлов за один ответ
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))

HELP_TEXT = (
    "MindBox: ваша внешняя память.\n\n"
    "Сохранить: напишите факт («провода в чёрном чемодане на балконе») "
    "или пришлите фото или файл С ПОДПИСЬЮ.\n"
    "Найти: спросите своими словами («где провода?», «скинь шаблон акта»).\n"
    "Всё сразу: «скинь все акты».\n\n"
    "Поиск идёт только по тексту подписей, содержимое фото и файлов не читается.\n\n"
    "Не сохраняйте пароли и номера документов: тексты заметок обрабатывает "
    "внешний сервис LLM."
)


@dp.message(Command("start", "help"))
async def start_command(message: Message):
    await message.answer(HELP_TEXT)


async def send_long_text(message: Message, text):
    # У сообщения в Telegram лимит 4096 символов, режем по строкам с запасом
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > 3800:
            await message.answer(chunk)
            chunk = ""
        chunk = chunk + line + "\n"
    if chunk.strip() != "":
        await message.answer(chunk)


async def send_one_file(message: Message, f):
    caption = f["text"][:1000]       # у подписи в Telegram есть лимит длины

    # До трёх попыток: если Telegram просит подождать, ждём и пробуем снова
    for attempt in range(3):
        try:
            if f["file_type"] == "photo":
                await message.answer_photo(photo=f["file_id"], caption=caption)
            else:
                await message.answer_document(document=f["file_id"], caption=caption)
            return
        except TelegramRetryAfter as error:
            await asyncio.sleep(error.retry_after + 1)
        except Exception as error:
            logger.warning("Файл не отправился: %s", error)
            await message.answer(
                "Заметка найдена, но файл больше недоступен: " + caption
            )
            return


async def send_files(message: Message, files):
    # Только заметки, у которых есть файл
    with_files = []
    for f in files:
        if f["file_id"] != "":
            with_files.append(f)

    to_send = with_files[:MAX_FILES_SENT]
    for f in to_send:
        await send_one_file(message, f)
        await asyncio.sleep(0.3)      # небольшая пауза между файлами

    if len(with_files) > len(to_send):
        await message.answer(
            "Прислано файлов: " + str(len(to_send)) + " из " + str(len(with_files))
            + ". Уточните вопрос, чтобы найти нужный."
        )


@dp.message(F.text | F.photo | F.document)
async def handle_message(message: Message):
    user_id = message.from_user.id

    text = ""
    file_id = ""
    file_type = "text"

    if message.photo:
        file_id = message.photo[-1].file_id      # последнее в списке: самое большое фото
        file_type = "photo"
        text = message.caption or ""
    elif message.document:
        file_id = message.document.file_id
        file_type = "document"
        text = message.caption or ""
    elif message.text:
        text = message.text

    # Пока граф думает, показываем «печатает...»
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        # Граф работает синхронно и долго, поэтому запускаем в отдельном потоке
        result = await asyncio.to_thread(run, user_id, text, file_id, file_type)
    except Exception:
        logger.exception("Ошибка в графе")
        await message.answer("Что-то пошло не так, попробуйте ещё раз.")
        return

    logger.info("user=%s путь: %s", user_id, " -> ".join(result["steps"]))

    await send_long_text(message, result["answer"])
    await send_files(message, result.get("files", []))


async def health(request):
    # Ручка здоровья: оркестратор (Docker) проверяет, что процесс жив
    return web.json_response({"status": "ok", "service": "mindbox-bot"})


async def start_health_server():
    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HEALTH_PORT)
    await site.start()
    print("Health: http://localhost:" + str(HEALTH_PORT) + "/health")
    return runner


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Не найден TELEGRAM_BOT_TOKEN в файле .env")
        return

    # Прогреваем модель эмбеддингов, чтобы первое сообщение не ждало загрузку
    print("Загрузка модели эмбеддингов...")
    await asyncio.to_thread(embed_text, "прогрев")

    # Проверяем связь с трекером при старте (если ключи Langfuse заданы)
    await asyncio.to_thread(check_connection)

    runner = await start_health_server()
    bot = Bot(token=token)
    print("Бот запущен. Остановить: Ctrl+C")
    try:
        await dp.start_polling(bot)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())