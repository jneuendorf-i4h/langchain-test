run MODULE:
    poetry run python -m main {{ MODULE }}

module MODULE:
    poetry run python -m langchain_test.{{ MODULE }}

hf-login:
    poetry run huggingface-cli login