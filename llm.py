"""LangChain integration module.

Exposes:
  build_chain(llm=None) -> RunnableSequence  – construct the chain without network calls.
  get_completion(text: str) -> str           – run the chain and return the model response.

The LLM_PROVIDER environment variable controls which backend is used:
  langchain   (default) – ChatOpenAI via langchain-openai
  huggingface           – HuggingFaceHub via langchain-community
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_TEMPLATE = """Proof read this text: ```{input_text}```

Summarise the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, \
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and a suggestion on how to improve it.
"""

PROMPT = PromptTemplate(
    input_variables=["input_text"],
    template=_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(llm=None):
    """Return a LangChain RunnableSequence (prompt | llm).

    If *llm* is None the appropriate LLM is instantiated based on the
    ``LLM_PROVIDER`` environment variable.  Passing a custom *llm* object
    (e.g. a mock) makes the function usable in offline unit tests without any
    network calls.
    """
    if llm is None:
        llm = _make_llm()
    # The LCEL pipe operator produces a RunnableSequence.
    return PROMPT | llm


def _make_llm():
    """Instantiate the backend LLM based on LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()

    if provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub  # lazy import

        repo_id = os.getenv("HUGGINGFACEHUB_REPO_ID", "google/flan-t5-xxl")
        return HuggingFaceHub(
            repo_id=repo_id,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={"temperature": 0, "max_new_tokens": 512},
        )

    # Default: langchain (ChatOpenAI)
    from langchain_openai import ChatOpenAI  # lazy import

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_completion(text: str) -> str:
    """Run the LangChain chain and return the model's text response.

    Parameters
    ----------
    text:
        The user's spoken-English transcript to analyse.

    Returns
    -------
    str
        A JSON-parseable string containing the analysis.
    """
    chain = build_chain()
    result = chain.invoke({"input_text": text})
    # ChatOpenAI returns an AIMessage; HuggingFaceHub returns a plain str.
    if hasattr(result, "content"):
        return result.content
    return str(result)
