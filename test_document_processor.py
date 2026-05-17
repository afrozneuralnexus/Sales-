"""Tests for DocumentProcessor."""

import io
import pytest
import pandas as pd

from src.document_processor import DocumentProcessor, DocumentChunk


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=100, chunk_overlap=10)


def make_csv_file(content: str, name: str = "test.csv"):
    class FakeFile:
        def __init__(self, data, filename):
            self._buf = io.BytesIO(data.encode())
            self.name = filename
            self.size = len(data)

        def read(self):
            return self._buf.read()

        def seek(self, pos):
            self._buf.seek(pos)

    return FakeFile(content, name)


def test_process_csv_basic(processor):
    csv_data = "product,revenue,units\nWidget A,5000,100\nWidget B,8000,200\nWidget C,3000,75"
    f = make_csv_file(csv_data)
    chunks = processor.process_file(f)
    assert len(chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    full_text = " ".join(c.text for c in chunks)
    assert "Widget A" in full_text
    assert "revenue" in full_text.lower()


def test_process_csv_source_name(processor):
    csv_data = "a,b\n1,2"
    f = make_csv_file(csv_data, "sales_q1.csv")
    chunks = processor.process_file(f)
    assert all(c.source == "sales_q1.csv" for c in chunks)


def test_chunk_size_respected(processor):
    # Large CSV → multiple chunks
    rows = "\n".join(f"Product{i},{i*100},{i*10}" for i in range(200))
    csv_data = "product,revenue,units\n" + rows
    f = make_csv_file(csv_data)
    chunks = processor.process_file(f)
    # Each chunk should be at most chunk_size words
    for c in chunks:
        assert len(c.text.split()) <= processor.chunk_size + 5  # small tolerance


def test_split_text_overlap(processor):
    text = " ".join([f"word{i}" for i in range(250)])
    chunks = processor._split_text(text)
    assert len(chunks) > 1
    # Check overlap: last words of chunk N appear at start of chunk N+1
    for i in range(len(chunks) - 1):
        end_words = set(chunks[i].split()[-processor.chunk_overlap:])
        start_words = set(chunks[i + 1].split()[:processor.chunk_overlap])
        assert len(end_words & start_words) > 0


def test_unsupported_extension(processor):
    class FakeFile:
        name = "file.json"
        size = 10

        def read(self):
            return b"{}"

        def seek(self, _):
            pass

    with pytest.raises(ValueError, match="Unsupported"):
        processor.process_file(FakeFile())


def test_empty_csv(processor):
    f = make_csv_file("col1,col2\n")
    chunks = processor.process_file(f)
    # Should handle gracefully (may return 0 or more chunks)
    assert isinstance(chunks, list)
