"""
Hugging Face Serverless Inference — Q&A chat script.

Usage:
    python chat_hf.py                        # interactive mode
    python chat_hf.py -q "What is OCR?"     # single question
    python chat_hf.py --system "You are a security analyst."

Requires:
    pip install huggingface_hub
    HF_TOKEN env variable (from huggingface.co/settings/tokens)
"""

import argparse
import os
import sys

from huggingface_hub import InferenceClient

MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_SYSTEM = "You are a helpful assistant, responding to a bash script shell output."


def build_client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("Get a token at https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)
    #print(token[:5] + "..." + token[-5:])
    return InferenceClient(model=MODEL, token=token, provider="featherless-ai")


JSON_INSTRUCTION = "Respond only with valid JSON. Do not include any text or explanation outside the JSON."


def ask(client: InferenceClient, history: list[dict], question: str, json_mode: bool = False) -> str:
    history.append({"role": "user", "content": question})
    kwargs = {"messages": history, "max_tokens": 1024}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat_completion(**kwargs)
    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer


def interactive(client: InferenceClient, system: str, json_mode: bool = False):
    history = [{"role": "system", "content": system}]
    print(f"Model : {MODEL}")
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
        answer = ask(client, history, question, json_mode)
        print(f"\nAssistant: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM via Hugging Face Inference API")
    parser.add_argument("-q", "--question", default=None, help="Single question (non-interactive)")
    parser.add_argument("-s", "--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--json", action="store_true", help="Force JSON-only output")
    args = parser.parse_args()

    system = args.system
    if args.json:
        system = f"{system} {JSON_INSTRUCTION}"

    client = build_client()

    if args.question:
        history = [{"role": "system", "content": system}]
        answer = ask(client, history, args.question, args.json)
        print(answer)
    else:
        interactive(client, system, args.json)


if __name__ == "__main__":
    main()
