import re
from typing import List, Dict


def split_medicines(raw_text: str) -> List[str]:
    """
    Split the entire corpus into medicine-wise blocks.
    Medicine names are FULL CAPS at line start.
    """
    pattern = r"\n(?=[A-Z][A-Z\s\-]{3,}\n)"
    blocks = re.split(pattern, raw_text.strip())

    return [b.strip() for b in blocks if b.strip()]


def extract_name_and_synonyms(block: str) -> Dict:
    """
    Extract:
      - Medicine name (first ALL CAPS line)
      - Synonyms (short non-sentence lines after it)
      - Remaining content
    """
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    if not lines:
        raise ValueError("Empty medicine block")

    name = lines[0]

    synonyms = []
    content_start = 1

    for i in range(1, len(lines)):
        line = lines[i]

        # Stop if real sentence begins
        if "." in line or "––" in line or ":" in line:
            break

        # Avoid false synonyms like "Dose"
        if len(line.split()) <= 6:
            synonyms.append(line)
            content_start = i + 1
        else:
            break

    content = "\n".join(lines[content_start:])

    return {
        "name": name,
        "synonyms": synonyms,
        "content": content
    }
