"""Convert LibreOffice help pages (.xhp XML) to Markdown.

Headings become Markdown headings, ordered lists become numbered steps, tables become
pipe-free row text, and platform switches keep the default (non-macOS) variant.
Index bookmarks, comments, images and embedded fragments are dropped.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

SKIP = {"bookmark", "comment", "image", "embed", "embedvar", "ahelp", "object", "video", "audio", "meta"}
BLOCK = {"paragraph", "tip", "note", "warning"}


def _inline(el: ET.Element) -> str:
    """Flatten inline markup (menuitem, emph, link, keycode, switchinline) to plain text."""
    if el.tag in SKIP:
        return el.tail or ""
    if el.tag == "switchinline":
        default = el.find("defaultinline")
        text = _inline_children(default) if default is not None else ""
        return text + (el.tail or "")
    if el.tag == "br":
        return " " + (el.tail or "")
    return _inline_children(el) + (el.tail or "")


def _inline_children(el: ET.Element) -> str:
    return (el.text or "") + "".join(_inline(child) for child in el)


PLACEHOLDERS = {"$[officename]": "LibreOffice", "%PRODUCTNAME": "LibreOffice", "%PRODUCTVERSION": ""}


def _clean(text: str) -> str:
    for key, value in PLACEHOLDERS.items():
        text = text.replace(key, value)
    return re.sub(r"\s+", " ", text).strip()


def _blocks(el: ET.Element, out: list[str]) -> None:
    for child in el:
        tag = child.tag
        if tag in SKIP:
            continue
        if re.fullmatch(r"h[1-6]", tag):
            text = _clean(_inline_children(child))
            if text:
                out.append(f"\n{'#' * int(tag[1])} {text}\n")
        elif tag in BLOCK:
            text = _clean(_inline_children(child))
            if text:
                prefix = {"tip": "Tip: ", "note": "Note: ", "warning": "Warning: "}.get(tag, "")
                out.append(prefix + text + "\n")
        elif tag == "list":
            ordered = child.get("type") == "ordered"
            for n, item in enumerate(child.findall("listitem"), 1):
                parts: list[str] = []
                _blocks(item, parts)
                text = _clean(" ".join(parts))
                if text:
                    out.append(f"{n}. {text}" if ordered else f"- {text}")
            out.append("")
        elif tag == "table":
            for row in child.iter("tablerow"):
                cells = []
                for cell in row.findall("tablecell"):
                    parts = []
                    _blocks(cell, parts)
                    cells.append(_clean(" ".join(parts)))
                line = " — ".join(c for c in cells if c)
                if line:
                    out.append(line)
            out.append("")
        elif tag == "switch":
            default = child.find("default") or child.find("defaultinline")
            if default is not None:
                _blocks(default, out)
        else:  # section, variable, sort, and other containers
            text = _clean(child.text or "")
            if text and tag not in {"section", "sort"}:
                out.append(text + "\n")
            _blocks(child, out)


def xhp_to_markdown(xml_text: str) -> tuple[str | None, str]:
    """Return (title, markdown body). Raises ET.ParseError on malformed XML."""
    root = ET.fromstring(xml_text)
    title_el = root.find("./meta/topic/title")
    title = _clean(title_el.text or "") if title_el is not None else None
    body = root.find("body")
    out: list[str] = []
    if body is not None:
        _blocks(body, out)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    if title and not text.startswith("# "):
        text = f"# {title}\n\n{text}"
    return title, text
