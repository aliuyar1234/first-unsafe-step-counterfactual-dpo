"""Parse one ODCV agent turn into a normalized action record.

Behavior requirements:
- support the benchmark's thought/action/final output pattern
- preserve the raw assistant text
- extract `parsed_action_type` and `parsed_action`
- never drop unparseable text silently; return explicit parse errors
"""
from __future__ import annotations

import re
from typing import Dict


def parse_agent_output(assistant_text: str) -> Dict[str, str]:
    """Return a normalized dict with raw text, action type, action payload, and parse status."""
    action_type = "final"
    action_payload = assistant_text.strip()
    parse_error = ""

    action_match = re.search(r"^Action:\s*(.+?)\s*$", assistant_text, flags=re.MULTILINE)
    input_match = re.search(r"^Action Input:\s*([\s\S]+)$", assistant_text, flags=re.MULTILINE)
    if action_match:
        action_type = action_match.group(1).strip().lower()
        if input_match:
            action_payload = input_match.group(1).strip()
        else:
            action_payload = ""
            parse_error = "missing_action_input"
    elif assistant_text.strip():
        parse_error = "missing_action_marker"
    else:
        parse_error = "empty_assistant_text"

    return {
        "raw_text": assistant_text,
        "parsed_action_type": action_type or "unknown",
        "parsed_action": action_payload,
        "parse_error": parse_error,
    }
