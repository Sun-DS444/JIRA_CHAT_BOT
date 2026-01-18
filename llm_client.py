# llm_client.py
import requests
import subprocess
import time

# Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def ensure_ollama_running():
    try:
        requests.get(OLLAMA_TAGS_URL, timeout=2)
    except Exception:
        subprocess.Popen(["ollama", "serve"], shell=True)
        time.sleep(5)

    # 🔥 Warm-up call (loads model)
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": "smollm2:360m",
                "prompt": "Hello",
                "stream": False
            },
            timeout=60
        )
    except Exception:
        pass


# Main LLM call used by the chatbot
def call_llm(context: str, question: str) -> str:
    try:
        # Ensure Ollama is up before calling
        ensure_ollama_running()

        prompt = f"""
You are a helpful, friendly AI assistant.
Answer clearly and simply.


Instructions:
- Answer like ChatGPT in simple, clear words
- Explain in short paragraphs
- Do NOT mention Jira, tickets, IDs, sections, or documents
- Use the given context only to understand the issue
- If information is unclear, explain it in a general way

Context (background only):
{context}

User Question:
{question}

Answer:
""".strip()

        payload = {
            "model": "smollm2:360m",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        return response.json().get("response", "").strip()

    except Exception:
        return (
            "⚠️ The AI engine is starting or temporarily unavailable.\n\n"
            "Please try again in a few seconds."
        )
