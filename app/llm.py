from langchain_core.callbacks import BaseCallbackHandler

from app import config


class UsageCounter(BaseCallbackHandler):
    # Считает вызовы и токены всех обращений к LLM. Нужен для расчёта стоимости.
    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def snapshot(self):
        return {"calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}

    def on_llm_end(self, response, **kwargs):
        self.calls = self.calls + 1
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    self.prompt_tokens = self.prompt_tokens + usage.get("input_tokens", 0)
                    self.completion_tokens = self.completion_tokens + usage.get("output_tokens", 0)


USAGE = UsageCounter()


def get_llm():
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, temperature=0, callbacks=[USAGE])

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, callbacks=[USAGE])