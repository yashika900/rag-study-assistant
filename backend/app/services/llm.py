"""Gemini chat model factory and prompt helpers."""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.app.utils.helpers import BASE_DIR


ENV_PATH = BASE_DIR / "backend" / ".env"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]

MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "3"))


def _get_api_key(env_path: Path = ENV_PATH) -> str:
    """Load the Gemini API key from backend/.env or the process environment."""

    load_dotenv(env_path, override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is missing. Add it to {env_path}.")
    return api_key


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model used for final answers."""

    load_dotenv(ENV_PATH, override=True)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    print(f"Creating Gemini chat model: {model_name}")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=_get_api_key(),
        temperature=0.2,
    )


def invoke_llm_with_fallback(prompt: str) -> str:
    """Generate an answer and fall back if a model is unavailable or out of quota."""

    load_dotenv(ENV_PATH, override=True)
    configured_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    model_names = [configured_model]
    model_names.extend(model for model in FALLBACK_GEMINI_MODELS if model not in model_names)

    last_error: Exception | None = None
    for model_name in model_names:
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                print(f"Calling Gemini LLM model: {model_name} (attempt {attempt})")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=_get_api_key(),
                    temperature=0.2,
                )
                start_time = time.time()
                response = llm.invoke(prompt)
                end_time = time.time()
                print(f"Gemini response time: {end_time - start_time:.2f} seconds")
                return response.content.strip()
            except Exception as exc:
                last_error = exc
                wait_seconds = _retry_delay_seconds(exc, attempt)
                print("\n===== ERROR =====")
                print(str(exc))
                if attempt < MAX_LLM_RETRIES and _is_temporary_error(exc):
                    print(f"Gemini LLM call failed temporarily. Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
                print(f"Trying next Gemini model after failure: {model_name}")
                break

    raise RuntimeError(f"All Gemini chat models failed. Last error: {last_error}")


def _is_temporary_error(exc: Exception) -> bool:
    """Return True for rate-limit or temporary availability errors."""

    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message or "UNAVAILABLE" in message or "retry" in message.lower()


def _retry_delay_seconds(exc: Exception, attempt: int) -> int:
    """Read Gemini retry delay from the error, then fall back to small backoff."""

    message = str(exc)

    retry_delay_match = re.search(r"retryDelay': '(\d+)s'", message)
    if retry_delay_match:
        return max(int(retry_delay_match.group(1)) + 2, 5)

    retry_text_match = re.search(r"retry in ([\d.]+)s", message, flags=re.IGNORECASE)
    if retry_text_match:
        return max(int(float(retry_text_match.group(1))) + 2, 5)

    return min(30, 5 * attempt)
