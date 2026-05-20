import os
from typing import Any, Optional, Protocol


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class LangChainChatProvider:
    def __init__(self, model: str, temperature: float = 0, chat_model: Optional[Any] = None):
        self.model = model
        self.temperature = temperature
        self._chat = chat_model

    def _get_chat_model(self):
        if self._chat is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError(
                    "LangChain support requires langchain-openai. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc
            self._chat = ChatOpenAI(model=self.model, temperature=self.temperature)
        return self._chat

    def complete(self, prompt: str) -> str:
        response = self._get_chat_model().invoke([{"role": "user", "content": prompt}])
        return getattr(response, "content", str(response))


class OpenAIChatProvider:
    def __init__(self, model: str, temperature: float = 0, client: Optional[Any] = None):
        self.model = model
        self.temperature = temperature
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def create_llm_provider(model: Optional[str] = None, provider: Optional[str] = None) -> LLMProvider:
    selected_provider = (provider or os.getenv("LLM_PROVIDER") or "langchain").strip().lower()
    selected_model = model or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"

    if selected_provider == "langchain":
        return LangChainChatProvider(model=selected_model)
    if selected_provider == "openai":
        return OpenAIChatProvider(model=selected_model)

    raise ValueError("Unsupported LLM_PROVIDER. Use 'langchain' or 'openai'.")
