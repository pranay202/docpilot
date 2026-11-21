from pathlib import Path

import pytest

from app.agent import _steps_answer
from app.ingest.chunker import Chunk
from app.ingest.loader import clean_mdx, load_markdown_folder
from app.ingest.xhp import xhp_to_markdown
from app.retrieval import rerank
from app.store import Hit

XHP = """<?xml version="1.0" encoding="UTF-8"?>
<helpdocument version="1.0">
<meta><topic id="t"><title id="tit">Page Numbers</title></topic></meta>
<body>
<bookmark branch="index" id="b"><bookmark_value>page numbers;inserting</bookmark_value></bookmark>
<comment>internal note</comment>
<h1 id="h1"><variable id="v"><link href="x.xhp">Page Numbers</link></variable></h1>
<paragraph role="paragraph">In $[officename] Writer, a page number is a field.</paragraph>
<h2 id="h2">To Start With a Defined Page Number</h2>
<list type="ordered">
  <listitem><paragraph role="listitem">Click into the first paragraph.</paragraph></listitem>
  <listitem><paragraph role="listitem">Choose <menuitem>Format - Paragraph - Text flow</menuitem> tab.</paragraph></listitem>
</list>
<tip id="t1">Press <switchinline select="sys"><caseinline select="MAC"><keycode>Command+F9</keycode></caseinline><defaultinline><keycode>Ctrl+F9</keycode></defaultinline></switchinline> to toggle.</tip>
<table id="tb"><tablerow><tablecell><paragraph role="tablecontent">Key</paragraph></tablecell><tablecell><paragraph role="tablecontent">Effect</paragraph></tablecell></tablerow></table>
<image src="x.png" id="img"><alt id="a">icon</alt></image>
</body>
</helpdocument>"""


def test_xhp_to_markdown():
    title, md = xhp_to_markdown(XHP)
    assert title == "Page Numbers"
    assert md.startswith("# Page Numbers")
    assert "In LibreOffice Writer, a page number is a field." in md
    assert "## To Start With a Defined Page Number" in md
    assert "1. Click into the first paragraph." in md
    assert "2. Choose Format - Paragraph - Text flow tab." in md
    assert "Tip: Press Ctrl+F9 to toggle." in md and "Command+F9" not in md
    assert "Key — Effect" in md
    assert "internal note" not in md and "page numbers;inserting" not in md and "icon" not in md


def test_loader_reads_xhp_and_skips_malformed(tmp_path: Path):
    (tmp_path / "good.xhp").write_text(XHP)
    (tmp_path / "bad.xhp").write_text("<helpdocument><body>")
    docs = load_markdown_folder(tmp_path, "writer")
    assert [d.title for d in docs] == ["Page Numbers"]


def test_clean_mdx():
    raw = "---\ntitle: 'Guide'\n---\nimport X from 'y'\n\n<Admonition type=\"note\">\n\nSee [the docs](/x).\n\n</Admonition>\n\n```js\n<Keep />\n```"
    title, body = clean_mdx(raw)
    assert title == "Guide"
    assert "import" not in body and "Admonition" not in body
    assert "See the docs." in body and "<Keep />" in body


def _hit(section: str, text: str, score: float, source: str = "a.xhp") -> Hit:
    return Hit(Chunk(section, "p", "T", section, source, text), score)


def test_rerank_prefers_matching_heading():
    hits = [
        _hit("Guide > Formatting an Index", "index styles and levels", 0.50),
        _hit("Guide > To Insert a Table of Contents", "Choose Insert", 0.40),
    ]
    assert rerank("insert a table of contents", hits)[0].chunk.section.endswith("To Insert a Table of Contents")


def test_steps_stitched_across_sibling_chunks():
    hits = [
        _hit("Guide > To create a form letter", "5. Pick fields.\n6. Finish.", 0.9),
        _hit("Guide > To create a form letter", "1. Choose Tools - Mail Merge Wizard.\n2. Select a template.", 0.8),
    ]
    answer = _steps_answer(hits)
    assert answer is not None
    assert answer.answer.splitlines()[0].startswith("1. Choose Tools - Mail Merge Wizard")
    assert answer.answer.splitlines()[-1] == "6. Finish."


def test_imperative_bullets_count_as_steps():
    hits = [_hit("Guide > Activation", "To enable it:\n- Choose Slide Show - Slide Show Settings.\n- Select Windowed.", 0.9)]
    answer = _steps_answer(hits)
    assert answer is not None and "Slide Show Settings" in answer.answer and answer.answer.startswith("To enable it:")


BRAND = Path(__file__).resolve().parent.parent / "brand_docs" / "libreoffice"


@pytest.mark.skipif(not (BRAND / ".sha").exists(), reason="run scripts/fetch_libreoffice_help.sh first")
def test_libreoffice_brand_eval_meets_thresholds():
    from evals.run_eval import build_service, evaluate, load_rows

    rows = load_rows(Path(__file__).resolve().parent.parent / "evals" / "libreoffice.jsonl")
    svc = build_service(BRAND, sorted({r["product"] for r in rows}))
    report = evaluate(svc, rows)
    assert report["hit@5"] >= 0.85 and report["answer_accuracy"] >= 0.7
