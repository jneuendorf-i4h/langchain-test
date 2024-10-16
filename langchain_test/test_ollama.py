from functools import cache

from langchain_ollama import ChatOllama

USER_PROMPT = "Kategorien:"


@cache
def get_llm():
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        # other params...
    )
    return llm


system_message = (
    "system",
    (
        "Du bist ein Interviewer und überlegst dir Fragen, "
        "mit denen du den Nutzer nach seinen Präferenzen interviewst. "
        "Der Nutzer gibt die Kategorien für die Fragen vor."
    ),
)


def invoke(message: str) -> str:
    if not message:
        message = "Lieblingsessen, Maximalpreis"

    llm = get_llm()
    ai_message = llm.invoke([
        system_message,
        (
            "human",
            f"Kategorien: {message}",
        )
    ])
    return str(ai_message.content)
