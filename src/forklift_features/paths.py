from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or script working directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root from {start}")


def safe_path_part(value: str) -> str:
    """Make a short path component safe for local output filenames."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("._")
    return safe or "unknown"
