from typing import Dict, Tuple

IGNORE = "— Ignore —"

def validate_mapping(mapping: Dict[str, str], require_key: str) -> Tuple[bool, str]:
    if require_key not in mapping.values():
        return False, f"Key mapping required to '{require_key}'."
    return True, ""
