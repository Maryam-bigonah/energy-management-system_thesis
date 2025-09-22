from __future__ import annotations

from typing import Any, Dict, List


def aggregate(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    """Stub: count by key."""
    counts: Dict[str, int] = {}
    for r in records:
        k = str(r.get(key))
        counts[k] = counts.get(k, 0) + 1
    return counts


def validate_schema(records: List[Dict[str, Any]], required_fields: List[str]) -> bool:
    """Stub: ensure fields exist in each record."""
    for r in records:
        for f in required_fields:
            if f not in r:
                return False
    return True
