"""Thin orchestration layer that delegates LLM calls to llm.py.

The public interface (get_completion) is unchanged so main.py requires no
modification.
"""

from dotenv import find_dotenv, load_dotenv
import os

import llm as _llm

load_dotenv(find_dotenv())


def get_completion(input_text: str, **_kwargs) -> str:  # noqa: ANN001
    """Analyse *input_text* and return a JSON-parseable analysis string.

    Delegates entirely to :func:`llm.get_completion` so that the backend
    (LangChain / HuggingFace) can be switched via the ``LLM_PROVIDER`` env
    var without touching this file.
    """
    return _llm.get_completion(input_text)
