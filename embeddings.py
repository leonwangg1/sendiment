"""
embeddings.py — pluggable semantic-similarity for Sediment.

Why this matters: the deterministic lexical matcher can't tell "target model is
Claude" and "Anthropic model as target" are the same lesson. Embedding the text
into a vector space and using cosine similarity fixes this — paraphrases land
near each other. The lexical text_sim() stays as a zero-dependency fallback.

Design choices:
  - Provider is pluggable (OpenAI, Voyage; add more easily).
  - The vector is cached on the Heuristic (`_embedding` field) so we pay the API
    cost ONCE per entry, not on every recall.
  - Batching: when re-embedding many entries (e.g. after a bulk import), one API
    call handles up to 100 texts.
  - Failure mode is loud-but-safe: if the API fails, we log and fall back to
    text_sim() rather than crashing the agent's working loop. Memory > perfection.
"""

from __future__ import annotations
import os, math, time, hashlib, json
from typing import Optional, Protocol, Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Provider protocol — anything that turns strings into float vectors          #
# --------------------------------------------------------------------------- #
class EmbeddingProvider(Protocol):
    name: str
    dim: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- #
# OpenAI provider (text-embedding-3-small by default — cheap and good)        #
# --------------------------------------------------------------------------- #
class OpenAIEmbeddings:
    """Uses OPENAI_API_KEY from env. Default model is text-embedding-3-small
    (1536 dim, ~$0.02 per 1M tokens). Override with SEDIMENT_EMBED_MODEL."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAIEmbeddings needs `pip install openai`. "
                "Or use VoyageEmbeddings, or set provider=None to use lexical fallback."
            ) from e
        self.model = model or os.environ.get("SEDIMENT_EMBED_MODEL", "text-embedding-3-small")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.name = f"openai:{self.model}"
        self.dim = 1536 if "small" in self.model else 3072

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # API caps; batch in chunks of 100 to be safe
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            chunk = list(texts[i:i + 100])
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            out.extend(d.embedding for d in resp.data)
        return out


# --------------------------------------------------------------------------- #
# Voyage provider (the other strong choice — better quality, slightly pricier)#
# --------------------------------------------------------------------------- #
class VoyageEmbeddings:
    """Uses VOYAGE_API_KEY from env. Default model voyage-3-lite."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import voyageai
        except ImportError as e:
            raise RuntimeError(
                "VoyageEmbeddings needs `pip install voyageai`."
            ) from e
        self.model = model or os.environ.get("SEDIMENT_EMBED_MODEL", "voyage-3-lite")
        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        self.name = f"voyage:{self.model}"
        # voyage-3-lite is 512-dim; voyage-3 is 1024
        self.dim = 512 if "lite" in self.model else 1024

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            chunk = list(texts[i:i + 100])
            resp = self.client.embed(chunk, model=self.model, input_type="document")
            out.extend(resp.embeddings)
        return out


# --------------------------------------------------------------------------- #
# Similarity                                                                   #
# --------------------------------------------------------------------------- #
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard cosine similarity in [-1, 1] — but for normalized embeddings
    from these providers, effectively in [0, 1] for similar texts."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def text_for_embedding(when: str, prefer: str) -> str:
    """Compose the text we embed. Including both fields lets cosine pick up
    semantic similarity in EITHER the trigger or the action."""
    return f"WHEN {when} -> PREFER {prefer}"


# --------------------------------------------------------------------------- #
# Provider factory + safe ensure-embeddings helper                            #
# --------------------------------------------------------------------------- #
def make_provider(name: Optional[str] = None) -> Optional[EmbeddingProvider]:
    """Resolve provider by name or SEDIMENT_EMBED_PROVIDER env var.
    Returns None if disabled or unavailable (caller should fall back to lexical)."""
    name = (name or os.environ.get("SEDIMENT_EMBED_PROVIDER", "")).lower().strip()
    if name in ("", "none", "off", "lexical"):
        return None
    try:
        if name == "openai":
            return OpenAIEmbeddings()
        if name == "voyage":
            return VoyageEmbeddings()
    except Exception as e:
        # honest about the failure — don't silently hide it from the user
        print(f"[sediment.embeddings] provider '{name}' unavailable: {e}. "
              f"Falling back to lexical similarity.")
        return None
    raise ValueError(f"Unknown embedding provider: {name!r}")
