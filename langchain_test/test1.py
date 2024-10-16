from functools import cache

from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

USER_PROMPT = "Aufgabe:"


@cache
def get_chain():
    hf = HuggingFacePipeline.from_model_id(
        # model_id="gpt2",
        # model_id="benjamin/gerpt2",
        # model_id="benjamin/gerpt2-large",  # outputs grammatically correct German
        # model_id="ai-forever/mGPT",
        # model_id="meta-llama/Llama-3.2-1B",  # need to register
        model_id="meta-llama/Llama-3.2-3B",  # need to register
        # model_id="utter-project/EuroLLM-1.7B",
        task="text-generation",
        model_kwargs={
            "temperature": 0.05,  # Control randomness
            "do_sample": True,  # Sampling for more natural question generation
        },
        pipeline_kwargs={
            "max_new_tokens": 80,
        },
    )

    template = """Du bist ein Meinungsforscher und überlegst dir Fragen,
    mit denen du Nutzer nach ihren Präferenzen fragen kannst.

    Erzeuge für jede der folgenden Kategorien genau 1 Frage,
    die sich nur auf die entsprechende Kategorie bezieht.
    Kategorien: {question}.
    
    Frage 1: """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf
    return chain


def invoke(message: str) -> str:
    if not message:
        message = "1. Lieblingsessen, 2. Maximalpreis"
    chain = get_chain()
    return chain.invoke({"question": message})
