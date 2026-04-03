"""
chatbot.py
----------
Ollama-powered mental health chatbot.
Exposes stream_response() and get_models() for use in app.py.

Standalone test:
    python chatbot.py
"""

import requests
import json

OLLAMA_BASE  = "http://localhost:11434"
CHAT_URL     = f"{OLLAMA_BASE}/api/chat"
TAGS_URL     = f"{OLLAMA_BASE}/api/tags"
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are Neuron, a compassionate mental health support assistant.

Your responsibilities:
- Listen with empathy and respond in a warm, non-judgmental tone
- Provide evidence-based coping strategies (CBT, mindfulness, sleep hygiene, etc.)
- Offer psychoeducation about anxiety, depression, stress, and related topics
- Recognise when someone may be in crisis and direct them to professional help

Hard boundaries:
- NEVER diagnose a medical or psychiatric condition
- NEVER prescribe or recommend specific medications
- NEVER claim to replace a therapist, psychologist, or psychiatrist
- Always keep responses concise (2–4 paragraphs) and actionable

If the user expresses suicidal thoughts, self-harm, or acute crisis:
  1. Validate their feelings without minimising them
  2. Firmly encourage them to contact a professional or helpline immediately
  3. Provide: iCall India — 9152987821 | Vandrevala Foundation — 1860-2662-345

Tone: warm, hopeful, grounded, professional."""


# ── Core streaming function ────────────────────────────────────────────────────
def stream_response(messages: list, model: str = DEFAULT_MODEL):
    """
    Generator — yields text chunks from Ollama as they arrive.
    `messages` should be a list of {"role": ..., "content": ...} dicts
    NOT including the system prompt (added automatically).

    Usage:
        for chunk in stream_response(messages):
            print(chunk, end="", flush=True)
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": True,
    }
    try:
        with requests.post(CHAT_URL, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done", False):
                        yield data["message"]["content"]
    except requests.exceptions.ConnectionError:
        yield (
            "⚠️ **Ollama is not running.**\n\n"
            "Please start it in a terminal:\n"
            "```\nollama serve\n```\n"
            "Then pull a model:\n"
            "```\nollama pull llama3.2\n```\n"
            "Refresh the page once it's running."
        )
    except requests.exceptions.Timeout:
        yield "⏳ The model is taking too long to respond. Please try again."
    except Exception as exc:
        yield f"❌ Unexpected error: {exc}"


# ── Non-streaming helper ───────────────────────────────────────────────────────
def get_response(messages: list, model: str = DEFAULT_MODEL) -> str:
    """Blocking — returns the full response string."""
    return "".join(stream_response(messages, model))


# ── Available models ───────────────────────────────────────────────────────────
def get_models() -> list[str]:
    """Returns list of locally available Ollama model names."""
    try:
        data = requests.get(TAGS_URL, timeout=4).json()
        names = [m["name"] for m in data.get("models", [])]
        return names if names else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


# ── Crisis keyword detector ────────────────────────────────────────────────────
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life",
    "self-harm", "self harm", "hurt myself", "don't want to live",
    "want to die", "no reason to live",
]

def is_crisis_message(text: str) -> bool:
    """Returns True if the message contains crisis-related language."""
    lower = text.lower()
    return any(kw in lower for kw in CRISIS_KEYWORDS)


CRISIS_ADDENDUM = """

---
🆘 **If you are in crisis, please reach out immediately:**
- **iCall (India):** 9152987821
- **Vandrevala Foundation:** 1860-2662-345
- **SNEHI:** 044-24640050
- **iCall WhatsApp:** +91 9152987821

You don't have to face this alone. Please contact a professional right away.
"""


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Neuron Chatbot — CLI Test")
    print("Type 'quit' to exit.\n")
    history = []
    while True:
        user = input("You: ").strip()
        if user.lower() in ("quit", "exit"):
            break
        history.append({"role": "user", "content": user})
        print("Neuron: ", end="", flush=True)
        full = ""
        for chunk in stream_response(history):
            print(chunk, end="", flush=True)
            full += chunk
        print()
        if is_crisis_message(user):
            print(CRISIS_ADDENDUM)
        history.append({"role": "assistant", "content": full})
