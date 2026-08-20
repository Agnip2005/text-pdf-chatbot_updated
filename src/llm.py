# llm.py

import os
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class FallbackLLM:
    """
    Used when no API key is configured, or when the real LLM fails to
    initialize. Its .invoke() must return an object shaped the same way
    app.py expects a real response to look (answer.content[0]["text"]),
    otherwise app.py crashes with a TypeError whenever we fall back.
    """

    def invoke(self, prompt):
        message = (
            "The chatbot is currently unavailable because no Groq API key is configured. "
            "Set GROQ_API_KEY in your environment to enable live responses."
        )
        return SimpleNamespace(content=[{"text": message}])


def load_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        try:
            return ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0,
            )
        except Exception:
            return FallbackLLM()

    return FallbackLLM()
