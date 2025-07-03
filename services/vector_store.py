"""Vector store helper for Qdrant (shared local server).

This module centralises all interaction with the local Qdrant instance so the
rest of the codebase does not need to know implementation details.

Key points:

• `get_client()` – returns a singleton *network* `QdrantClient` that talks to a
  local Qdrant server running on ``http://127.0.0.1:6333``.  If the server is
  not yet running, the first call automatically spawns it (pointing at
  ``qdrant_db`` on disk) and waits until it is ready.  All processes therefore
  share one server and can use the database concurrently without file-lock
  conflicts.
• `recreate_collection(client, name, dim)` – (re)creates a collection with the
  desired dimensionality using cosine distance.
• `embed(texts)` – returns ``list[Vector]`` using the same sentence-transformer
  model everywhere so that embeddings are consistent across ingestion and
  querying.

The embedding model is loaded lazily at first call to avoid slowing down import
at cold start.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import subprocess
import time
import atexit
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from pathlib import Path
_QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
_QDRANT_CONFIG = str((Path(__file__).resolve().parent.parent / "qdrant.yaml"))
# Helpful diagnostics – printed once at import time.
print("[vector_store] QDRANT_URL:", _QDRANT_URL)
print("[vector_store] QDRANT_CONFIG:", _QDRANT_CONFIG)
# We use the widely-available MiniLM model. (384-dimensional embeddings.)
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 1536


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it."""
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


def embed(texts: Iterable[str]) -> List[List[float]]:
    """Return list of embedding vectors for *texts*."""
    model = _get_embedding_model()
    return model.encode(list(texts)).tolist()


from functools import lru_cache

_server_proc: subprocess.Popen | None = None

def shutdown_server(timeout: float = 10.0) -> None:
    """Terminate the Qdrant server started by this module (if any).

    If the server was not spawned by :func:`get_client` in the current Python
    interpreter (i.e. ``_server_proc`` is *None*), this is a no-op so that the
    caller can safely invoke it regardless of who owns the server.

    Parameters
    ----------
    timeout:
        Seconds to wait for the server process to exit after sending
        ``SIGTERM``.  If the process is still alive after the timeout it is
        force-killed with ``SIGKILL``.
    """
    global _server_proc

    if _server_proc is None:
        return  # nothing we own is running

    if _server_proc.poll() is not None:
        # Process already exited – clear handle so that a new server can be
        # started later if needed.
        _server_proc = None
        return

    # Ask the server to terminate gracefully first.
    _server_proc.terminate()
    try:
        _server_proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Escalate if it refuses to die.
        _server_proc.kill()
        _server_proc.wait()
    finally:
        _server_proc = None  # Allow a fresh server to spawn next time.
        # Clear cached client so that a new one is created if requested later.
        try:
            get_client.cache_clear()
        except Exception:
            pass


def _start_local_server() -> None:
    """Spawn a standalone Qdrant server (if not already running)."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        return  # already running

    cmd = [
        "qdrant",
        "--config-path",
        _QDRANT_CONFIG 
    ]
    _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Ensure graceful shutdown on interpreter exit
    atexit.register(lambda: _server_proc.terminate() if _server_proc and _server_proc.poll() is None else None)

    # Wait until server responds
    # Give the server up to ~60 s to come online (120 × 0.5 s).
    max_attempts = 120
    for attempt in range(max_attempts):
        try:
            QdrantClient(url=_QDRANT_URL, timeout=60).get_collections()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(
        "Failed to start local Qdrant server after waiting 60s. "
        "Check that the 'qdrant' binary can access the storage directory and that port 6333 is free."
    )


def _ensure_qdrant_ready(client: QdrantClient) -> None:
    """Raise RuntimeError if server is not reachable via *client*."""
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to reach Qdrant server at {client.url}. Is it running and accessible?"
        ) from exc


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a singleton Qdrant client.

    • Tries to connect to an existing server (``_QDRANT_URL``).
    • If that fails, spins up a local server via ``_start_local_server`` and
      retries.
    """
    try:
        client = QdrantClient(url=_QDRANT_URL, timeout=60)
        _ensure_qdrant_ready(client)
        return client
    except Exception:
        # server not running yet – spawn our own
        _start_local_server()
        client = QdrantClient(url=_QDRANT_URL, timeout=60)
        _ensure_qdrant_ready(client)
        return client


def recreate_collection(client: QdrantClient, name: str, dim: int = _EMBEDDING_DIM) -> None:
    """(Re)create *name* collection with given dimensionality using cosine distance."""
    client.recreate_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )


import uuid

def str2uuid(value: str) -> str:
    """Return a deterministic UUID (v5) derived from arbitrary *value* string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))

__all__ = [
    "shutdown_server",
    "embed",
    "get_client",
    "recreate_collection",
    "str2uuid",
]
