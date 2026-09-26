import json
import time
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b"


@dataclass
class Reply:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class OllamaClient:
    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL, timeout: int = 600):
        self.model = model
        self.url = url
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> Reply:
        body = {
            "model": self.model,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return Reply(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            seconds=time.perf_counter() - start,
        )


def ollama_available(url: str = OLLAMA_URL) -> bool:
    try:
        urllib.request.urlopen(url.replace("/api/chat", "/api/tags"), timeout=3)
        return True
    except OSError:
        return False
