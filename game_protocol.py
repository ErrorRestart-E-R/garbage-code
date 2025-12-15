"""
Game protocol (shared): CONTROL line spec + safe parsing helpers.

Why this file exists:
- We want the main core to stay clean even as games grow.
- Games can request mode transitions (start/stop) via a machine-readable
  CONTROL line that the main core can parse deterministically.
- The CONTROL line MUST NEVER be spoken (TTS) or stored as the assistant's
  visible chat content.

CONTROL line spec:
- The model MAY emit a single CONTROL line as the very first line.
- Grammar:
  - CONTROL: START <game_id>
  - CONTROL: STOP
  - CONTROL: NONE
- The rest of the output (after the first newline) is the user-facing text.

Safety:
- Only whitelisted game_ids are accepted.
- Anything else is treated as CONTROL: NONE.

Session scope:
- session_id is the Discord voice channel ID (per-voice-channel session).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Set, Tuple


class ControlAction(str, Enum):
    START = "START"
    STOP = "STOP"
    NONE = "NONE"


@dataclass(frozen=True)
class ControlCommand:
    action: ControlAction
    game_id: Optional[str] = None  # required only for START


# "game_id" format: conservative ascii slug (no spaces)
_GAME_ID_RE = r"[a-z0-9][a-z0-9_\-]{0,63}"

# Accept one control line only, as the first line.
_CONTROL_LINE_RE = re.compile(
    rf"^\s*CONTROL:\s*(START\s+(?P<gid>{_GAME_ID_RE})|STOP|NONE)\s*$",
    re.IGNORECASE,
)


def parse_control_line(
    line: str,
    allowed_game_ids: Optional[Set[str]] = None,
) -> Optional[ControlCommand]:
    """
    Parse a single line. Returns None if the line is not a valid CONTROL line.

    allowed_game_ids:
    - If provided, START is accepted only if gid is in this set.
    - STOP/NONE are always accepted.
    """
    if not line:
        return None
    m = _CONTROL_LINE_RE.match(line)
    if not m:
        return None

    raw = (m.group(1) or "").strip()
    raw_upper = raw.upper()

    if raw_upper.startswith("START"):
        gid = (m.group("gid") or "").strip().lower()
        if not gid:
            return None
        if allowed_game_ids is not None and gid not in allowed_game_ids:
            return ControlCommand(action=ControlAction.NONE, game_id=None)
        return ControlCommand(action=ControlAction.START, game_id=gid)

    if raw_upper == "STOP":
        return ControlCommand(action=ControlAction.STOP, game_id=None)

    return ControlCommand(action=ControlAction.NONE, game_id=None)


def strip_control_from_text(
    text: str,
    allowed_game_ids: Optional[Set[str]] = None,
) -> Tuple[Optional[ControlCommand], str]:
    """
    If the first line is a CONTROL line, remove it and return (cmd, remaining_text).
    Otherwise return (None, original_text).
    """
    if not text:
        return None, ""

    # Split only once to keep the rest intact.
    first_line, sep, rest = text.partition("\n")
    cmd = parse_control_line(first_line, allowed_game_ids=allowed_game_ids)
    if cmd is None:
        return None, text

    # Remove the CONTROL line. Trim only leading newlines/spaces of the remainder.
    remaining = (rest if sep else "").lstrip("\n").lstrip()
    return cmd, remaining


