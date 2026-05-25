from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterable

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime when PDF extraction is used
    fitz = None


WATERMARK_PATTERNS = [
    re.compile(r"\bqueplan(?:\.cl)?\b", re.IGNORECASE),
    re.compile(r"\bconfidencial\b", re.IGNORECASE),
]
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:p(?:á|a)?g(?:ina)?|page)?\s*\d+\s*(?:de\s*\d+)?\s*$", re.IGNORECASE)
INDEX_LINE_PATTERN = re.compile(r"^\s*[^\n]{3,}\.\.+\s*\d+\s*$")
MULTISPACE_PATTERN = re.compile(r"[ \t]+")
TOKEN_PATTERN = re.compile(r"[\wáéíóúñ]+", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    start_char: int
    end_char: int


class SimpleHashEmbedder:
    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _normalize_line(line: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", line).strip()


def _is_noise_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_NUMBER_PATTERN.match(line) or INDEX_LINE_PATTERN.match(line):
        return True
    return any(pattern.search(line) for pattern in WATERMARK_PATTERNS)


def clean_policy_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r", "").split("\n"):
        line = _normalize_line(raw_line)
        if _is_noise_line(line):
            continue
        cleaned_lines.append(line)

    deduped_lines: list[str] = []
    for line in cleaned_lines:
        if deduped_lines and deduped_lines[-1] == line:
            continue
        deduped_lines.append(line)

    compact = "\n".join(deduped_lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def _extract_pages_with_pymupdf(pdf_path: str | Path) -> list[list[str]]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. Install it with `pip install pymupdf`."
        )

    pages: list[list[str]] = []
    with fitz.open(str(pdf_path)) as document:
        for page in document:
            lines = [_normalize_line(line) for line in page.get_text("text").splitlines()]
            pages.append([line for line in lines if line])
    return pages


def _identify_repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    edge_lines: list[str] = []
    for page_lines in pages:
        edge_lines.extend(page_lines[:2])
        edge_lines.extend(page_lines[-2:])

    counts = Counter(edge_lines)
    repeated = {
        line
        for line, count in counts.items()
        if count >= 2 and len(line) <= 120 and not _is_noise_line(line)
    }
    return repeated


def extract_and_clean_pdf_text(pdf_path: str | Path, output_path: str | Path | None = None) -> str:
    pages = _extract_pages_with_pymupdf(pdf_path)
    repeated_edge_lines = _identify_repeated_edge_lines(pages)

    flattened_lines: list[str] = []
    for page_lines in pages:
        filtered = [line for line in page_lines if line not in repeated_edge_lines]
        flattened_lines.extend(filtered)
        flattened_lines.append("")

    cleaned_text = clean_policy_text("\n".join(flattened_lines))

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned_text, encoding="utf-8")

    return cleaned_text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    source = text.strip()
    if not source:
        return []

    parts = [part.strip() for part in re.split(r"\n\s*\n", source) if part.strip()]
    if not parts:
        return []

    chunks: list[Chunk] = []
    current_text = ""
    current_start = 0
    cursor = 0

    def flush_chunk() -> None:
        nonlocal current_text, current_start
        chunk_body = current_text.strip()
        if not chunk_body:
            current_text = ""
            return
        start = source.find(chunk_body, current_start)
        if start < 0:
            start = max(current_start, 0)
        end = start + len(chunk_body)
        chunks.append(Chunk(text=chunk_body, start_char=start, end_char=end))
        current_text = ""
        current_start = max(end - overlap, 0)

    for part in parts:
        if len(part) > chunk_size:
            sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(part) if sentence.strip()]
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    words = sentence.split()
                    piece = ""
                    for word in words:
                        candidate = f"{piece} {word}".strip()
                        if len(candidate) <= chunk_size:
                            piece = candidate
                            continue
                        if piece:
                            if current_text and len(f"{current_text}\n\n{piece}") > chunk_size:
                                flush_chunk()
                            current_text = f"{current_text}\n\n{piece}".strip()
                            cursor += len(piece)
                            piece = word
                    if piece:
                        if current_text and len(f"{current_text}\n\n{piece}") > chunk_size:
                            flush_chunk()
                        current_text = f"{current_text}\n\n{piece}".strip()
                        cursor += len(piece)
                    continue

                candidate = f"{current_text}\n\n{sentence}".strip()
                if current_text and len(candidate) > chunk_size:
                    flush_chunk()
                    candidate = sentence
                current_text = candidate
                cursor += len(sentence)
            continue

        candidate = f"{current_text}\n\n{part}".strip()
        if current_text and len(candidate) > chunk_size:
            flush_chunk()
            candidate = part
        current_text = candidate
        cursor += len(part)

    flush_chunk()

    if not chunks:
        return [Chunk(text=source[:chunk_size], start_char=0, end_char=min(len(source), chunk_size))]

    return chunks


def build_chroma_payload(
    chunks: list[Chunk],
    source_id: str,
    embedder: SimpleHashEmbedder | None = None,
) -> dict[str, list]:
    embedder = embedder or SimpleHashEmbedder()
    chunk_texts = [chunk.text for chunk in chunks]

    return {
        "ids": [f"{source_id}-chunk-{index}" for index, _ in enumerate(chunks)],
        "documents": chunk_texts,
        "metadatas": [
            {
                "source": source_id,
                "chunk_index": index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for index, chunk in enumerate(chunks)
        ],
        "embeddings": embedder.embed_many(chunk_texts),
    }


def process_policy_pdf_for_chromadb(
    pdf_path: str | Path,
    output_directory: str | Path,
    source_id: str,
    chunk_size: int = 900,
    overlap: int = 120,
    embedding_dimension: int = 128,
) -> dict[str, list]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    cleaned_text_path = output_directory / f"{source_id}_cleaned.txt"
    cleaned_text = extract_and_clean_pdf_text(pdf_path, cleaned_text_path)

    chunks = chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
    payload = build_chroma_payload(
        chunks,
        source_id=source_id,
        embedder=SimpleHashEmbedder(dimension=embedding_dimension),
    )

    payload_path = output_directory / f"{source_id}_chroma_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload
