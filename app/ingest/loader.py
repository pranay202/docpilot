"""Load documentation from a local Markdown folder or a sitemap URL."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}


@dataclass
class Document:
    product: str
    title: str
    source: str  # file path or URL
    text: str


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def load_markdown_folder(path: str | Path, product: str) -> list[Document]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    docs = []
    for file in sorted(root.rglob("*")):
        if file.suffix.lower() in MARKDOWN_SUFFIXES and file.is_file():
            text = file.read_text(encoding="utf-8")
            if text.strip():
                docs.append(Document(product, _title_from_markdown(text, file.stem), str(file), text))
    return docs


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"h1", "h2", "h3"} and not self._skip:
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag in {"p", "li", "br", "div", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()
    return parser.title.strip(), text


def load_sitemap(url: str, product: str, limit: int = 200, client: httpx.Client | None = None) -> list[Document]:
    own = client is None
    client = client or httpx.Client(timeout=15, follow_redirects=True)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = [el.text.strip() for el in ET.fromstring(resp.content).findall(".//sm:loc", ns) if el.text]
        docs = []
        for loc in locs[:limit]:
            page = client.get(loc)
            if page.status_code != 200:
                continue
            title, text = html_to_text(page.text)
            if text:
                docs.append(Document(product, title or loc, loc, text))
        return docs
    finally:
        if own:
            client.close()


def load_source(source: str, product: str) -> list[Document]:
    if source.startswith(("http://", "https://")):
        return load_sitemap(source, product)
    return load_markdown_folder(source, product)
