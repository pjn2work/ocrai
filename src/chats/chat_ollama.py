"""
Ollama local inference — Q&A chat script.

Usage:
    python chat_ollama.py                        # interactive mode
    python chat_ollama.py -q "What is OCR?"     # single question
    python chat_ollama.py -s "You are a security analyst."
    python chat_ollama.py --model llama3         # use a different model

Requires:
    Ollama running locally (https://ollama.com)
    pip install requests
"""

import argparse
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
DEFAULT_SYSTEM = "You are a helpful assistant, responding to a bash script shell output."


JSON_INSTRUCTION = "Respond only with valid JSON. Do not include any text or explanation outside the JSON."


def ask(history: list[dict], question: str, model: str, json_mode: bool = False) -> str:
    history.append({"role": "user", "content": question})
    payload = {"model": model, "messages": history, "stream": False}
    if json_mode:
        payload["format"] = "json"  # enforced at the tokenizer level by Ollama
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)
    answer = response.json()["message"]["content"]
    history.append({"role": "assistant", "content": answer})
    return answer


def interactive(system: str, model: str, json_mode: bool = False):
    history = [{"role": "system", "content": system}]
    print(f"Model : {model}")
    print(f"System: {system}")
    print("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer = ask(history, question, model, json_mode)
        print(f"\nAssistant: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model")
    parser.add_argument("-q", "--question", default=None, help="Single question (non-interactive)")
    parser.add_argument("-s", "--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--json", action="store_true", help="Force JSON-only output")
    args = parser.parse_args()

    system = args.system
    if args.json:
        system = f"{system} {JSON_INSTRUCTION}"

    if args.question:
        history = [{"role": "system", "content": system}]
        answer = ask(history, args.question, args.model, args.json)
        print(answer)
    else:
        interactive(system, args.model, args.json)


if __name__ == "__main__":
    main()
