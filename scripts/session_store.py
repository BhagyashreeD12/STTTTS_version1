"""
session_store.py — Lightweight per-session data persistence.

Writes one JSON file per session to:
    <project_root>/data/sessions/<session_id>.json

The file is updated after every validated answer so you can watch it grow
in real time during testing. Each write is atomic (write-to-.tmp then
rename) to prevent partial / corrupt files.

Usage
-----
    from session_store import SessionStore

    store = SessionStore("abc12345")
    store.update("whatsapp_number", "+14165550199")
    store.update("policy_start_date", "2026-06-01")

    data = store.get_all()   # dict of everything collected so far
    print(store.path)        # Path — easy to open in an editor during testing
    store.reset()            # wipe in-memory + delete file (between test runs)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default base directory — resolved relative to this file so it works
# regardless of the working directory when the script is launched.
_DEFAULT_BASE_DIR: Path = Path(__file__).parent.parent / "data" / "sessions"


class SessionStore:
    """Thin wrapper that persists validated session data to a JSON file.

    Parameters
    ----------
    session_id : str
        Unique identifier for this session (used as the filename stem).
    base_dir   : Path | None
        Directory to write session files to.
        Defaults to ``<project_root>/data/sessions/``.
        The directory (and any parents) is created automatically on first use.
    """

    def __init__(
        self,
        session_id: str,
        base_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self._base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR
        self._data: dict[str, Any] = {}
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Path to the session JSON file."""
        return self._base_dir / f"{self.session_id}.json"

    def update(self, field_name: str, value: Any) -> None:
        """Store a validated field value and immediately persist to disk.

        Only validated values should be passed here — the flow engine ensures
        this by only calling update() after extract_structured_answer returns
        is_valid=True.
        """
        self._data[field_name] = value
        self.save()
        logger.debug("[SessionStore] %s ← %r", field_name, value)

    def get_all(self) -> dict[str, Any]:
        """Return a shallow copy of all collected data."""
        return dict(self._data)

    def save(self) -> None:
        """Write current data to disk atomically.

        Writes to a ``.tmp`` file first, then renames it over the target file.
        This prevents half-written JSON if the process is interrupted mid-write.
        """
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(self._data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as exc:
            logger.error(
                "[SessionStore] Failed to save session %s: %s",
                self.session_id,
                exc,
            )

    def reset(self) -> None:
        """Clear in-memory data and delete the session file.

        Useful between test runs so stale data from a previous session does
        not accumulate.
        """
        self._data.clear()
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "[SessionStore] Could not delete %s: %s",
                self.path,
                exc,
            )
