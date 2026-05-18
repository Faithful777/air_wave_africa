"""Knowledge base data structures + parser."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DOMAIN_CHUNK_PREFIX


@dataclass
class KBChunk:
    chunk_id: str
    heading:  str
    body:     str
    domain:   str
    keywords: list = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.body}"


class KnowledgeBaseParser:
    CHUNK_RE = re.compile(r"^## \(CHUNK [A-Z]+-\d+\)\s*\|(.+)$", re.MULTILINE)
    KW_RE    = re.compile(r"\*\*Keywords:\*\*\s*(.+)$",          re.MULTILINE)

    def parse(self, doc_path):
        text = Path(doc_path).read_text(encoding="utf-8")
        matches = list(self.CHUNK_RE.finditer(text))
        if not matches:
            raise ValueError("No CHUNK headings found in document.")
        chunks = []
        for i, m in enumerate(matches):
            chunk_id = m.group(1).strip()
            heading  = m.group(2).strip() if m.lastindex >= 2 else ""
            start    = m.start()
            end      = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body     = text[start:end].strip()
            kw_m     = self.KW_RE.search(body)
            keywords = [k.strip() for k in kw_m.group(1).split(",")] if kw_m else []
            domain   = self._infer_domain(chunk_id)
            chunks.append(KBChunk(chunk_id, heading, body, domain, keywords))
        print(f"   ✅ Parsed {len(chunks)} KB chunks")
        return chunks

    @staticmethod
    def _infer_domain(chunk_id):
        for domain, prefix in DOMAIN_CHUNK_PREFIX.items():
            if f"CHUNK {prefix}" in chunk_id:
                return domain
        return "general"
