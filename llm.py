import os

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

LANGCHAIN_PROVIDER = "langchain"
HUGGINGFACE_PROVIDER = "huggingface"
SUPPORTED_PROVIDERS = {LANGCHAIN_PROVIDER, HUGGINGFACE_PROVIDER}

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DEFAULT_HUGGINGFACE_REPO_ID = os.getenv(
    "HUGGINGFACE_REPO_ID",
    "mistralai/Mistral-7B-Instruct-v0.2",
)

PROMPT_TEMPLATE = """proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use,
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""


def get_prompt(input_text: str) -> str:
    return PROMPT_TEMPLATE.format(input_text=input_text)


def get_provider(provider: str | None = None) -> str:
    provider_name = (provider or os.getenv("LLM_PROVIDER") or LANGCHAIN_PROVIDER)
    provider_name = provider_name.strip().lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        return LANGCHAIN_PROVIDER
    return provider_name


def _sync_huggingface_token_alias() -> None:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


def build_chain(llm=None, model: str | None = None):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chat_model = llm or ChatOpenAI(
        model=model or DEFAULT_OPENAI_MODEL,
        temperature=0,
    )
    return prompt | chat_model | StrOutputParser()


def build_huggingface_chain(llm=None, repo_id: str | None = None):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_community.llms import HuggingFaceEndpoint

    _sync_huggingface_token_alias()

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    huggingface_model = llm or HuggingFaceEndpoint(
        repo_id=repo_id or DEFAULT_HUGGINGFACE_REPO_ID,
        temperature=0,
    )
    return prompt | huggingface_model | StrOutputParser()


def get_completion(
    input_text: str,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    provider = get_provider(provider)
    if provider == HUGGINGFACE_PROVIDER:
        return build_huggingface_chain().invoke({"input_text": input_text})
    return build_chain(model=model).invoke({"input_text": input_text})
