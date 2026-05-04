import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from flask import Flask, jsonify, request, send_from_directory
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL_ID = "google/gemma-2b-it"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
MAX_HISTORY_TURNS = 4


def ensure_adapter_path(adapter_path: Path) -> None:
    missing_files = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_path / name).exists()]
    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Adapter path is missing required files: {missing_str}. "
            f"Checked: {adapter_path.resolve()}"
        )


def get_hf_token(explicit_token: str | None) -> str | None:
    token = explicit_token or os.getenv("HF_TOKEN")
    if token:
        return token
    return None


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_base_model(base_model_id: str, hf_token: str | None, use_4bit: bool):
    if torch.cuda.is_available():
        if use_4bit:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            return AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token,
            )

        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=model_dtype,
            device_map="auto",
            token=hf_token,
        )

    return AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        token=hf_token,
    )


def load_model_and_tokenizer(
    adapter_path: Path,
    base_model_id: str,
    hf_token: str | None,
    use_adapter: bool,
):
    if use_adapter:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), token=hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = torch.cuda.is_available() and bitsandbytes_available()
    if torch.cuda.is_available() and not use_4bit:
        print("bitsandbytes is unavailable. Falling back to standard precision on GPU.", flush=True)
    if not torch.cuda.is_available():
        print("CUDA not detected. Loading on CPU (this can be slow).", flush=True)
    base_model = load_base_model(base_model_id=base_model_id, hf_token=hf_token, use_4bit=use_4bit)
    model = (
        PeftModel.from_pretrained(base_model, str(adapter_path))
        if use_adapter
        else base_model
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, history: List[Tuple[str, str]], user_message: str) -> str:
    messages = []
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": user_message})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    turns: List[str] = []
    for user_text, assistant_text in history:
        turns.append(f"<start_of_turn>user\n{user_text}<end_of_turn>")
        turns.append(f"<start_of_turn>model\n{assistant_text}<end_of_turn>")
    turns.append(f"<start_of_turn>user\n{user_message}<end_of_turn>")
    turns.append("<start_of_turn>model\n")
    return "\n".join(turns)


def generate_reply(
    tokenizer,
    model,
    history: List[Tuple[str, str]],
    user_message: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.3,
) -> str:
    prompt = build_prompt(tokenizer, history, user_message)
    tokenized = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
        )

    generated_ids = output_ids[0][tokenized["input_ids"].shape[-1] :]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return reply or "I could not generate a response."


def run_cli(tokenizer, model, max_new_tokens: int) -> None:
    history: List[Tuple[str, str]] = []
    print("Local AI assistant ready. Type 'quit' to exit.\n", flush=True)
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Exiting.", flush=True)
            return
        if not user_input:
            continue
        reply = generate_reply(
            tokenizer,
            model,
            history,
            user_input,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.3,
        )
        history.append((user_input, reply))
        print(f"Bot: {reply}\n", flush=True)


def run_react_server(tokenizer, model, max_new_tokens: int, port: int, web_dir: Path) -> None:
    app = Flask(__name__, static_folder=str(web_dir / "static"), static_url_path="/static")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/")
    def index():
        return send_from_directory(str(web_dir), "index.html")

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message", "")).strip()
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        history_payload = payload.get("history", [])
        history: List[Tuple[str, str]] = []
        if isinstance(history_payload, list):
            for item in history_payload:
                if not isinstance(item, dict):
                    continue
                user_text = str(item.get("user", "")).strip()
                assistant_text = str(item.get("assistant", "")).strip()
                if user_text and assistant_text:
                    history.append((user_text, assistant_text))
        history = history[-MAX_HISTORY_TURNS:]

        response_tokens = int(payload.get("max_new_tokens", max_new_tokens))
        response_tokens = max(32, min(1024, response_tokens))
        repetition_penalty = float(payload.get("repetition_penalty", 1.3))
        repetition_penalty = max(1.0, min(2.0, repetition_penalty))

        reply = generate_reply(
            tokenizer=tokenizer,
            model=model,
            history=history,
            user_message=user_message,
            max_new_tokens=response_tokens,
            repetition_penalty=repetition_penalty,
        )
        return jsonify({"reply": reply})

    @app.get("/api/config")
    def config():
        return jsonify({"defaultMaxNewTokens": max_new_tokens, "defaultRepetitionPenalty": 1.3})

    print(f"React UI running at http://127.0.0.1:{port}", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run your local Gemma assistant.")
    parser.add_argument("--adapter-path", default=".", help="Path to the LoRA adapter folder.")
    parser.add_argument(
        "--base-model-id",
        default=BASE_MODEL_ID,
        help="Base model ID to load from Hugging Face.",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN).")
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Run base model only without applying the local LoRA adapter.",
    )
    parser.add_argument(
        "--mode",
        choices=("react", "cli"),
        default="react",
        help="Choose chat interface mode.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port for local web UI.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum tokens per response.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate adapter files and exit without loading the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    use_adapter = not args.skip_adapter

    if use_adapter:
        try:
            ensure_adapter_path(adapter_path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        print("Running without adapter: base model only.", flush=True)

    if args.check_only:
        print(f"Adapter files found in: {adapter_path}")
        print("Validation complete.")
        return 0

    hf_token = get_hf_token(args.hf_token)
    if not hf_token:
        print(
            "HF token not provided. Attempting to load with local cache or public access.",
            flush=True,
        )

    print("Loading model and adapter. This can take a few minutes...", flush=True)
    try:
        tokenizer, model = load_model_and_tokenizer(
            adapter_path=adapter_path,
            base_model_id=args.base_model_id,
            hf_token=hf_token,
            use_adapter=use_adapter,
        )
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        if not hf_token:
            print(
                "If the base model is gated, set HF_TOKEN (or pass --hf-token) and retry.",
                file=sys.stderr,
            )
        return 1
    print("Model ready.", flush=True)

    if args.mode == "cli":
        run_cli(tokenizer, model, max_new_tokens=args.max_new_tokens)
        return 0

    web_dir = Path(__file__).resolve().parent / "web"
    if not web_dir.exists():
        print(f"Web directory not found: {web_dir}", file=sys.stderr)
        return 1
    run_react_server(tokenizer, model, max_new_tokens=args.max_new_tokens, port=args.port, web_dir=web_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
