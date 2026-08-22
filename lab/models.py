"""Pluggable chat backend.

Two backends ship: a local Ollama server (whatever models the user has pulled)
and any OpenAI-compatible HTTP API. The model is chosen in the browser, not in
the source.

The model never reaches the lattice. It proposes commands by emitting lines of
the form

    @cmd {"cmd": "inject_density", "x": 512, "y": 512, "sigma": 16, "strength": 0.05}

which land in an approval queue. A human clicks approve before anything is
published on 5557.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

CMD_LINE = re.compile(r"^\s*@cmd\s+(\{.*\})\s*$", re.MULTILINE)


@dataclass
class Proposal:
    raw: str
    command: dict | None
    error: str | None


def extract_proposals(text: str) -> list[Proposal]:
    out: list[Proposal] = []
    for match in CMD_LINE.finditer(text or ""):
        raw = match.group(1)
        try:
            out.append(Proposal(raw, json.loads(raw), None))
        except json.JSONDecodeError as exc:
            out.append(Proposal(raw, None, f"not valid JSON: {exc}"))
    return out


class Backend:
    name = "none"

    async def list_models(self) -> list[str]:
        return []

    async def chat(self, messages: list[dict], model: str,
                   images: list[bytes] | None = None,
                   temperature: float = 0.8, timeout: int = 180) -> str:
        raise NotImplementedError

    async def health(self) -> dict:
        return {"backend": self.name, "ok": False, "detail": "no backend configured"}


class NoneBackend(Backend):
    name = "none"

    async def chat(self, messages, model, images=None, temperature=0.8, timeout=180):
        raise RuntimeError(
            "No model backend configured. Set MODEL_BACKEND=ollama (with Ollama "
            "running) or MODEL_BACKEND=openai with an API key."
        )


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.url}/api/tags")
                r.raise_for_status()
            return {"backend": self.name, "ok": True, "url": self.url}
        except Exception as exc:
            return {"backend": self.name, "ok": False, "url": self.url,
                    "detail": f"{type(exc).__name__}: {exc}"}

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{self.url}/api/tags")
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))

    async def chat(self, messages, model, images=None, temperature=0.8, timeout=180):
        if not model:
            raise RuntimeError("no Ollama model selected")
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = [base64.b64encode(i).decode() for i in images]
        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self.url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, base_url: str, api_key: str | None, default_model: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "No API key. Export it in the environment variable named by "
                "openai_api_key_env (default RESONANCE_LAB_API_KEY)."
            )
        return {"Authorization": f"Bearer {self.api_key}"}

    async def health(self) -> dict:
        if not self.api_key:
            return {"backend": self.name, "ok": False, "url": self.base_url,
                    "detail": "no API key in the environment"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}/models", headers=self._headers())
                r.raise_for_status()
            return {"backend": self.name, "ok": True, "url": self.base_url}
        except Exception as exc:
            return {"backend": self.name, "ok": False, "url": self.base_url,
                    "detail": f"{type(exc).__name__}: {exc}"}

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{self.base_url}/models", headers=self._headers())
                r.raise_for_status()
                return sorted(m["id"] for m in r.json().get("data", []))
        except Exception:
            return [self.default_model] if self.default_model else []

    async def chat(self, messages, model, images=None, temperature=0.8, timeout=180):
        model = model or self.default_model
        if not model:
            raise RuntimeError("no model selected")
        msgs = [dict(m) for m in messages]
        if images:
            parts: list[dict] = [{"type": "text", "text": msgs[-1].get("content", "")}]
            for img in images:
                b64 = base64.b64encode(img).decode()
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{b64}"}})
            msgs[-1] = {"role": msgs[-1].get("role", "user"), "content": parts}
        payload = {"model": model, "messages": msgs, "temperature": temperature}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions",
                                  headers=self._headers(), json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def make_backend(cfg) -> Backend:
    kind = (cfg.model_backend or "none").lower()
    if kind == "ollama":
        return OllamaBackend(cfg.ollama_url)
    if kind in ("openai", "openai-compatible", "api"):
        return OpenAIBackend(cfg.openai_base_url,
                             os.environ.get(cfg.openai_api_key_env),
                             cfg.openai_model)
    return NoneBackend()
