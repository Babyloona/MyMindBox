import uuid
from datetime import datetime
import chromadb

from app import config
from app.embeddings import embed_text

client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# У каждой модели эмбеддингов своя коллекция: размеры векторов разные (384 и 1536)
collection_name = "notes_" + config.EMBEDDING_PROVIDER


def get_collection():
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    # Удаляет все заметки. Нужно только для тестов.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def save_note(user_id, text, file_id="", file_type="text"):
    collection = get_collection()

    vector = embed_text(text)
    note_id = str(uuid.uuid4())

    collection.add(
        ids=[note_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "user_id": str(user_id),
            "file_id": file_id,
            "file_type": file_type,
            "created": datetime.now().isoformat(),
        }],
    )
    return note_id


def search_notes(user_id, question, top_k=3):
    collection = get_collection()

    vector = embed_text(question)
    result = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        where={"user_id": str(user_id)},
    )

    found = []
    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]
        found.append({
            "text": result["documents"][0][i],
            "file_id": result["metadatas"][0][i]["file_id"],
            "file_type": result["metadatas"][0][i]["file_type"],
            "created": result["metadatas"][0][i]["created"],
            "score": round(1 - distance, 3),
        })
    return found