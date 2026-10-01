"""
Calls an NVIDIA NIM model (OpenAI-compatible chat API at
https://integrate.api.nvidia.com/v1/chat/completions) to explain a tab's data,
using a per-tab agent persona from agents.py.

Requires NVIDIA_API_KEY as an environment variable. Get one at
https://build.nvidia.com (create account -> Get API Key).
"""
import os
import requests

from src.agents import get_agent

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")


def explain_with_ai(tab: str, context: str = "") -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return (
            "NVIDIA_API_KEY is not set. Get a key at https://build.nvidia.com, "
            "then set it as an environment variable and restart the server:\n"
            "  export NVIDIA_API_KEY=\"your-key-here\"   (macOS/Linux)\n"
            "  $env:NVIDIA_API_KEY=\"your-key-here\"      (Windows PowerShell)"
        )

    agent = get_agent(tab)
    user_prompt = f"Explain this data from the '{tab}' tab:\n\n{context}" if context else \
        f"Explain the '{tab}' tab of this dashboard."

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": agent["system"]},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"[{agent['name']}] Request to NVIDIA API failed: {e}"
    except (KeyError, IndexError):
        return f"[{agent['name']}] Unexpected response format from NVIDIA API."
