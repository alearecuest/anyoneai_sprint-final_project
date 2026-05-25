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
MAX_REPEATED_EDGE_LINE_LENGTH = 120


@dataclass(frozen=True)
class Chunk:
    text: str
    start_char: int
    end_char: int


class SimpleHashEmbedder:
    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
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
        if count >= 2 and len(line) <= MAX_REPEATED_EDGE_LINE_LENGTH and not _is_noise_line(line)
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
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    source = text.strip()
    if not source:
        return []

    raw_parts = [part.strip() for part in re.split(r"\n\s*\n", source) if part.strip()]
    if not raw_parts:
        return []

    segments: list[str] = []
    for part in raw_parts:
        if len(part) <= chunk_size:
            segments.append(part)
            continue

        sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(part) if sentence.strip()]
        for sentence in sentences:
            if len(sentence) <= chunk_size:
                segments.append(sentence)
                continue

            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) <= chunk_size:
                    piece = candidate
                else:
                    if piece:
                        segments.append(piece)
                    piece = word
            if piece:
                segments.append(piece)

    chunk_texts: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n\n{segment}".strip() if current else segment
        if current and len(candidate) > chunk_size:
            chunk_texts.append(current)
            current = segment
        else:
            current = candidate
    if current:
        chunk_texts.append(current)

    if overlap > 0:
        with_overlap: list[str] = [chunk_texts[0]]
        for index in range(1, len(chunk_texts)):
            requested_tail = chunk_texts[index - 1][-overlap:].strip()
            available_tail_size = max(chunk_size - len(chunk_texts[index]) - 2, 0)
            overlap_tail = requested_tail[-available_tail_size:] if available_tail_size > 0 else ""
            candidate = f"{overlap_tail}\n\n{chunk_texts[index]}".strip() if overlap_tail else chunk_texts[index]
            with_overlap.append(candidate)
        chunk_texts = with_overlap

    chunks: list[Chunk] = []
    search_start = 0
    for chunk_body in chunk_texts:
        start = source.find(chunk_body, max(search_start - overlap, 0))
        if start < 0:
            start = source.find(chunk_body)
        if start < 0:
            raise ValueError(
                f"Could not map chunk boundaries back to source text for chunk of length {len(chunk_body)}."
            )
        end = start + len(chunk_body)
        chunks.append(Chunk(text=chunk_body, start_char=start, end_char=end))
        search_start = end

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
