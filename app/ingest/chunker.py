"""Heading-aware Markdown chunking.

Each chunk keeps the heading path it came from ("Payments > Webhooks > Retries"),
so answers can cite a precise section instead of a whole page.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from app.ingest.loader import Document

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    id: str
    product: str
    title: str
    section: str
    source: str
    text: str
    embedding: list[float] = field(default_factory=list)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading_path, body) pairs, ignoring '#' inside code fences."""
    path: list[str] = []
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2)]
            sections.append((" > ".join(path), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > size:
            out.append(buf)
            buf = buf[-overlap:] if overlap else ""
        if len(p) > size:  # hard-split very long paragraphs
            for i in range(0, len(p), size - overlap):
                piece = p[i : i + size]
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(piece)
            continue
        buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        out.append(buf)
    return out


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks = []
    for section, body in _sections(doc.text):
        for i, piece in enumerate(_windows(body, size, overlap)):
            digest = hashlib.sha1(f"{doc.source}|{section}|{i}|{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, doc.product, doc.title, section or doc.title, doc.source, piece))
    return chunks
