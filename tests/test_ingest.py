import httpx

from app.embeddings import HashingEmbedder, cosine, tokenize
from app.ingest.chunker import chunk_document
from app.ingest.loader import Document, html_to_text, load_markdown_folder, load_sitemap
from tests.conftest import DOCS


def test_markdown_loader_reads_titles():
    docs = load_markdown_folder(DOCS / "payments", "payments")
    assert {d.title for d in docs} >= {"Webhooks", "Refunds"}
    assert all(d.product == "payments" for d in docs)


def test_chunker_keeps_heading_path_and_ignores_code_fence_hashes():
    text = "# Guide\n\n## Setup\n\nStep one.\n\n```bash\n# not a heading\necho hi\n```\n\n## Usage\n\nCall it."
    chunks = chunk_document(Document("p", "Guide", "g.md", text))
    sections = [c.section for c in chunks]
    assert "Guide > Setup" in sections and "Guide > Usage" in sections
    assert any("# not a heading" in c.text for c in chunks)


def test_chunker_splits_long_sections_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_document(Document("p", "T", "t.md", f"# T\n\n{body}"), size=400, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 450 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_hashing_embedder_is_deterministic_and_semantic_enough():
    e = HashingEmbedder(256)
    a, b, c = e.embed(["verify webhook signature", "webhook signatures verification", "refund timelines"])
    assert e.embed(["verify webhook signature"])[0] == a
    assert cosine(a, b) > cosine(a, c)
    assert tokenize("Duplicates retried") == ["duplicate", "retry"]


def test_sitemap_loader_with_mock_transport():
    sitemap = (
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://docs.example.com/a</loc></url></urlset>"
    )
    page = "<html><head><title>Page A</title></head><body><nav>menu</nav><h2>Intro</h2><p>Hello docs.</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=sitemap)
        return httpx.Response(200, text=page)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        docs = load_sitemap("https://docs.example.com/sitemap.xml", "x", client=client)
    assert len(docs) == 1 and docs[0].title == "Page A"
    assert "Hello docs." in docs[0].text and "menu" not in docs[0].text


def test_html_to_text_marks_headings():
    title, text = html_to_text("<title>T</title><h1>Main</h1><p>Body</p>")
    assert title == "T" and "# Main" in text


def test_ingest_counts(service):
    assert service.store.count() > 10
    assert service.store.products() == ["notifications", "payments"]
