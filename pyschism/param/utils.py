
import re
import os
from enum import Enum
import warnings
from datetime import timedelta
"""Helper functions used in core / opt / schout"""

def _to_fortran_scalar(val):
    """Format a scalar as Fortran namelist value."""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, str):
        # safest: single-quote and escape single quotes by doubling them
        return "'" + val.replace("'", "''") + "'"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, Enum):
        return str(val.value)
    if isinstance(val,timedelta):
        out = str(val / timedelta(days=1))
        warnings.warn(f'setting timedelta {val} to {out} in .nml')
        return out
    raise TypeError(f"Unsupported value type for namelist: {type(val)} ({val!r})")

_OPT_START_RE = re.compile(r"^\s*&\s*opt\b", re.IGNORECASE)
_OPT_END_RE   = re.compile(r"^\s*/\s*$")
_ASSIGN_RE    = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\(\s*\d+\s*\))?\s*=")

def _extract_group_key_order(nml_path: os.PathLike, group: str = "opt") -> list[str]:
    """Return keys in the order they first appear inside &GROUP ... /."""
    start_re = re.compile(rf"^\s*&\s*{re.escape(group)}\b", re.IGNORECASE)
    keys: list[str] = []
    seen = set()

    in_group = False
    with open(nml_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # strip inline comments (! ...) for key detection
            line_nocomment = line.split("!", 1)[0]

            if not in_group:
                if start_re.match(line_nocomment):
                    in_group = True
                continue

            if _OPT_END_RE.match(line_nocomment):
                break

            m = _ASSIGN_RE.match(line_nocomment)
            if m:
                k = m.group(1)
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

    return keys
