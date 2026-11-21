"""
Downloading pinned models into the local models/ directory.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from quantbench.config import ModelSpec

MODELS_DIR = Path("models")

# Only what a transformers backend and the tokenizer need; skips other formats.
_PATTERNS = ["*.json", "*.safetensors", "*.txt", "LICENSE"]


def fetch_model(spec: ModelSpec, models_dir: Path = MODELS_DIR) -> Path:
    """
    Download one model at its pinned revision and return its local path.
    """
    target = models_dir / spec.name

    snapshot_download(
        repo_id=spec.repo,
        revision=spec.revision,
        local_dir=target,
        allow_patterns=_PATTERNS,
    )

    return target
