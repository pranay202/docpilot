"""Load documentation from a local Markdown folder or a sitemap URL."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

from app.ingest.xhp import xhp_to_markdown

MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown", ".xhp"}


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


FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
JSX_LINE = re.compile(r"^\s*</?\$?[A-Z][\w.$]*(\s[^>]*)?/?>\s*$")
JSX_TAG = re.compile(r"</?\$?[A-Z][\w.$]*(\s[^<>]*)?/?>")
MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def clean_mdx(text: str) -> tuple[str | None, str]:
    """Strip MDX-only syntax (frontmatter, imports, JSX components, link targets), keep prose and code.

    Returns (frontmatter title or None, cleaned Markdown). Code fences are left untouched.
    """
    title = None
    fm = FRONTMATTER.match(text)
    if fm:
        m = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", fm.group(1), re.M)
        title = m.group(1).strip() if m else None
        text = text[fm.end():]
    out, in_fence, in_jsx_block = [], False, 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if re.match(r"^(import|export)\s", stripped):
            continue
        # multi-line JSX opening tag: <Component\n  prop="x"\n>
        if in_jsx_block:
            if stripped.endswith(">"):
                in_jsx_block = 0
            continue
        if re.match(r"^<\$?[A-Z][\w.$]*\s*$", stripped) or (re.match(r"^<\$?[A-Z]", stripped) and ">" not in stripped):
            in_jsx_block = 1
            continue
        if JSX_LINE.match(line):
            continue
        line = JSX_TAG.sub("", line)
        line = MD_LINK.sub(r"\1", line)
        out.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return title, body


def load_markdown_folder(path: str | Path, product: str) -> list[Document]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    docs = []
    for file in sorted(root.rglob("*")):
        if file.suffix.lower() in MARKDOWN_SUFFIXES and file.is_file():
            raw = file.read_text(encoding="utf-8")
            if file.suffix.lower() == ".xhp":
                try:
                    fm_title, text = xhp_to_markdown(raw)
                except ET.ParseError:
                    continue
            else:
                fm_title, text = clean_mdx(raw)
            if not text.strip():
                continue
            title = fm_title or _title_from_markdown(text, file.stem)
            if not re.search(r"^#\s", text, re.M):
                text = f"# {title}\n\n{text}"
            docs.append(Document(product, title, str(file), text))
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
