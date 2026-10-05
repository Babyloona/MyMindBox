from app import config

_local_model = None
_openai_model = None


def embed_text(text):
    global _local_model, _openai_model

    if config.EMBEDDING_PROVIDER == "openai":
        if _openai_model is None:
            from langchain_openai import OpenAIEmbeddings
            _openai_model = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
        return _openai_model.embed_query(text)

    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_model.encode(text).tolist()