import sys
import time
from collections.abc import Callable

from langchain_test import test1, test_ollama

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"args: {sys.argv}")
    module = sys.argv[1]

    prompot: str
    invoke: Callable[[str], str]

    match module:
        case "test1":
            prompt = test1.USER_PROMPT
            invoke = test1.invoke
        case "test_ollama":
            prompt = test_ollama.USER_PROMPT
            invoke = test_ollama.invoke
        case _:
            raise ValueError(f"Invalid module argument '{module}'")

    while True:
        try:
            question = input(f"\n{prompt} ")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        start = time.time()
        answer = invoke(question)
        end = time.time()
        print(answer)
        print(f"  took: {end - start}s")

