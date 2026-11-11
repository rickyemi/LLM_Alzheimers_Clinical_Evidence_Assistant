from ad_rag.chunking import chunk_documents, chunk_section, split_sections
from ad_rag.ingestion import Document
from ad_rag.phi import has_phi, redact, scan_text


def test_phi_detects_structured_identifiers():
    text = "Patient: Jane Doe, MRN 1234567, DOB: 04/05/1950, call 555-123-4567 or jane@example.com, SSN 123-45-6789"
    kinds = {f.kind for f in scan_text(text)}
    assert {"PATIENT_NAME", "MRN", "DOB", "PHONE", "EMAIL", "SSN"} <= kinds


def test_redact_removes_identifiers():
    out = redact("Email jane@example.com about SSN 123-45-6789")
    assert "jane@example.com" not in out and "123-45-6789" not in out
    assert "[REDACTED:EMAIL]" in out


def test_clean_text_has_no_phi():
    assert not has_phi("Plasma p-tau217 is a marker of amyloid pathology.")


def test_sections_split_on_headings():
    secs = split_sections("# A\nalpha\n\n# B\nbeta")
    assert [h for h, _ in secs] == ["A", "B"]


def test_chunks_respect_size_and_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
    chunks = chunk_section(body, size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 + 10 for c in chunks)       # size + carried overlap


def test_oversize_paragraph_is_windowed():
    chunks = chunk_section("x " * 1000, size=200, overlap=40)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)


def test_chunk_metadata_and_embed_text():
    doc = Document("d1", "Title", "d1.md", "# Sec\nSome content here that is long enough.")
    ch = chunk_documents([doc], 500, 50, "v1")[0]
    assert ch.source == "d1.md" and ch.corpus_version == "v1"
    assert ch.embed_text.startswith("Title | Sec")
