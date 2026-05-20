import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

DEFAULT_MODEL = "gpt-3.5-turbo"

_PROMPT_TEMPLATE = """Proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it. Return only valid JSON with no markdown fences.
"""

PROMPT = PromptTemplate(
    input_variables=["input_text"],
    template=_PROMPT_TEMPLATE,
)


def get_prompt(input_text):
    return PROMPT.format(input_text=input_text)


def get_provider():
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()
    if provider == "huggingface":
        return "huggingface"
    return "langchain"


def build_chain(llm=None, model=None, provider=None):
    """Build the LangChain runnable without invoking a network request."""
    if llm is None:
        llm = _build_llm(model=model, provider=provider)
    return PROMPT | llm


def get_completion(text, model=None):
    chain = build_chain(model=model)
    result = chain.invoke({"input_text": text})
    return _response_to_text(result)


def _build_llm(model=None, provider=None):
    provider = provider or get_provider()

    if provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub

        return HuggingFaceHub(
            repo_id=os.getenv("HUGGINGFACEHUB_REPO_ID", "google/flan-t5-xxl"),
            huggingfacehub_api_token=_huggingface_token(),
            model_kwargs={"temperature": 0, "max_new_tokens": 512},
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _huggingface_token():
    return os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")


def _response_to_text(result):
    content = getattr(result, "content", None)
    if content is not None:
        return content

    if isinstance(result, dict):
        for key in ("text", "content", "output"):
            if key in result:
                return str(result[key])

    return str(result)
