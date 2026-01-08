#!/usr/bin/env python3
"""Generate a synthetic document corpus for RAG performance testing.

Why this exists:
- The repo's sample files in data/ are intentionally tiny.
- For throughput/latency benchmarks you typically need MBs–GBs of text.

Output:
- Creates either:
    - many per-doc text documents (doc_000001.txt, ...), OR
    - a sharded corpus (shard_000001.txt, ...) sized for GiB-scale benchmarking
- Optionally creates a combined file (combined.txt)
- Always writes a manifest.json and optionally corpus.jsonl

All content is original, synthetic, and deterministic given a seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO


_TOPICS: list[dict[str, list[str]]] = [
    {
        "name": "RAG",
        "keywords": [
            "retrieval",
            "ranking",
            "fusion",
            "chunking",
            "token budget",
            "grounding",
            "citations",
            "context window",
            "overlap",
            "BM25",
            "hybrid search",
            "reranker",
        ],
        "entities": [
            "query encoder",
            "document store",
            "vector index",
            "sparse index",
            "ingestion pipeline",
            "evaluation harness",
        ],
    },
    {
        "name": "FastAPI",
        "keywords": [
            "dependency injection",
            "pydantic models",
            "async endpoints",
            "middleware",
            "rate limits",
            "timeouts",
            "OpenAPI",
            "health checks",
            "streaming responses",
        ],
        "entities": [
            "router",
            "lifespan hook",
            "request context",
            "background task",
            "exception handler",
        ],
    },
    {
        "name": "VectorDB",
        "keywords": [
            "HNSW",
            "payload filters",
            "collections",
            "upsert",
            "snapshot",
            "replication",
            "consistency",
            "index build",
            "segment",
        ],
        "entities": [
            "point id",
            "payload schema",
            "distance metric",
            "shard key",
            "index optimizer",
        ],
    },
    {
        "name": "ML-Ops",
        "keywords": [
            "monitoring",
            "drift",
            "A/B testing",
            "SLO",
            "incident review",
            "feature store",
            "batch scoring",
            "online inference",
            "caching",
        ],
        "entities": [
            "pipeline run",
            "model registry",
            "deployment roll-back",
            "metrics dashboard",
        ],
    },
]

_STYLE_SNIPPETS: list[str] = [
    "Checklist: define objective, constrain inputs, measure outcomes.",
    "Rule of thumb: optimize the slowest stage first.",
    "Failure mode: silent truncation reduces answer quality.",
    "Tip: include stable identifiers in metadata for debugging.",
    "Anti-pattern: treating cache misses as errors.",
    "Observation: latency tails often dominate perceived performance.",
]


@dataclass(frozen=True)
class Manifest:
    seed: int
    docs: int
    min_words: int
    max_words: int
    format: str
    created_utc: str
    target_bytes: int | None
    total_bytes: int
    total_words_est: int
    out_dir: str
    mode: str
    shards: int | None
    combined_path: str | None
    jsonl_path: str | None


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "doc"


def _pick_topic(rng: random.Random) -> dict[str, list[str]]:
    return rng.choice(_TOPICS)


def _make_paragraph(rng: random.Random, topic: dict[str, list[str]], target_words: int) -> str:
    keywords = topic["keywords"]
    entities = topic["entities"]

    sentences: list[str] = []
    while sum(len(s.split()) for s in sentences) < target_words:
        kw1, kw2 = rng.sample(keywords, 2)
        ent = rng.choice(entities)
        snippet = rng.choice(_STYLE_SNIPPETS)

        template = rng.choice(
            [
                "In {area}, {kw1} interacts with {kw2} via the {ent} and the trade-off is usually latency vs accuracy.",
                "When tuning {kw1}, measure p50/p95 latency and track regression causes in the {ent}.",
                "A practical note: combine {kw1} with {kw2} but keep the {ent} metadata small and consistent.",
                "This section focuses on {kw1}: it sounds simple, but the {ent} often becomes the hidden bottleneck.",
                "We observed that {kw1} improves recall, while {kw2} can stabilize precision if the {ent} is configured carefully.",
            ]
        )

        sentences.append(template.format(area=topic["name"], kw1=kw1, kw2=kw2, ent=ent))
        if rng.random() < 0.35:
            sentences.append(snippet)

    # Trim to approximately target_words
    words = " ".join(sentences).split()
    if len(words) > target_words:
        words = words[:target_words]
    return " ".join(words)


def _make_doc(rng: random.Random, doc_id: int, min_words: int, max_words: int) -> tuple[str, dict[str, object]]:
    topic = _pick_topic(rng)
    words = rng.randint(min_words, max_words)

    title = f"Synthetic {topic['name']} Note {doc_id:06d}"
    doc_slug = _slug(title)

    sections = rng.randint(3, 7)
    remaining = words
    paras: list[str] = []

    for s in range(sections):
        # Allocate words roughly evenly with noise
        base = max(30, remaining // (sections - s))
        target = max(30, int(base * rng.uniform(0.7, 1.2)))
        remaining = max(0, remaining - target)
        paras.append(_make_paragraph(rng, topic, target))

    tags = rng.sample(topic["keywords"], k=min(5, len(topic["keywords"])))
    metadata = {
        "id": doc_id,
        "title": title,
        "slug": doc_slug,
        "topic": topic["name"],
        "tags": tags,
        "created_utc": _utc_iso(),
    }

    body = "\n\n".join(
        [
            f"# {title}",
            f"Topic: {topic['name']}",
            f"Tags: {', '.join(tags)}",
            "",
            *paras,
        ]
    ).strip() + "\n"

    return body, metadata


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _append_utf8(fp: IO[str], text: str) -> int:
    fp.write(text)
    # Return the UTF-8 byte count for accounting.
    return len(text.encode("utf-8"))


def generate(
    *,
    out_dir: Path,
    docs: int,
    seed: int,
    min_words: int,
    max_words: int,
    write_jsonl: bool,
    write_combined: bool,
    target_bytes: int | None = None,
    shards: int | None = None,
) -> Manifest:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    combined_path: Path | None = None
    combined_fp: IO[str] | None = None
    if write_combined:
        combined_path = out_dir / "combined.txt"
        combined_fp = combined_path.open("w", encoding="utf-8")

    jsonl_path: Path | None = None
    jsonl_fp: IO[str] | None = None
    if write_jsonl:
        jsonl_path = out_dir / "corpus.jsonl"
        jsonl_fp = jsonl_path.open("w", encoding="utf-8")

    total_bytes = 0
    total_words_est = 0

    mode = "docs"
    used_shards: int | None = None

    try:
        if target_bytes is not None:
            mode = "shards"
            shard_count = int(shards or 64)
            if shard_count <= 0:
                raise ValueError("--shards must be > 0 when using --target-* options")

            used_shards = shard_count
            per_shard_target = int(math.ceil(target_bytes / shard_count))

            doc_id = 0
            for shard_idx in range(1, shard_count + 1):
                if total_bytes >= target_bytes:
                    break

                shard_path = out_dir / f"shard_{shard_idx:06d}.txt"
                shard_written = 0

                with shard_path.open("w", encoding="utf-8") as shard_fp:
                    while shard_written < per_shard_target and total_bytes < target_bytes:
                        doc_id += 1
                        body, meta = _make_doc(rng, doc_id, min_words, max_words)

                        if shard_written > 0:
                            shard_written += _append_utf8(shard_fp, "\n\n---\n\n")
                        shard_written += _append_utf8(shard_fp, body)

                        if combined_fp is not None:
                            if total_bytes > 0:
                                _append_utf8(combined_fp, "\n\n---\n\n")
                            _append_utf8(combined_fp, body)

                        if jsonl_fp is not None:
                            jsonl_line = json.dumps({"text": body, "metadata": meta}, ensure_ascii=False)
                            jsonl_fp.write(jsonl_line + "\n")

                        total_bytes += len(body.encode("utf-8"))
                        total_words_est += len(body.split())

                # Ensure the shard is included in accounting even if delimiters skewed the estimate
                # (this is the actual on-disk size).
                total_bytes += max(0, shard_path.stat().st_size - shard_written)

        else:
            for i in range(1, docs + 1):
                body, meta = _make_doc(rng, i, min_words, max_words)
                filename = f"doc_{i:06d}.txt"
                path = out_dir / filename
                _write_atomic(path, body)

                if combined_fp is not None:
                    if i > 1:
                        _append_utf8(combined_fp, "\n\n---\n\n")
                    _append_utf8(combined_fp, body)

                if jsonl_fp is not None:
                    jsonl_fp.write(json.dumps({"text": body, "metadata": meta}, ensure_ascii=False) + "\n")

                total_bytes += path.stat().st_size
                total_words_est += len(body.split())
    finally:
        if combined_fp is not None:
            combined_fp.write("\n")
            combined_fp.close()
        if jsonl_fp is not None:
            jsonl_fp.close()

    manifest = Manifest(
        seed=seed,
        docs=docs,
        min_words=min_words,
        max_words=max_words,
        format="both" if (write_jsonl and True) else ("jsonl" if write_jsonl else "txt"),
        created_utc=_utc_iso(),
        target_bytes=target_bytes,
        total_bytes=total_bytes,
        total_words_est=total_words_est,
        out_dir=str(out_dir),
        mode=mode,
        shards=used_shards,
        combined_path=str(combined_path) if combined_path else None,
        jsonl_path=str(jsonl_path) if jsonl_path else None,
    )

    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="data/generated/benchmark_corpus",
        help="Output directory for generated docs (default: data/generated/benchmark_corpus)",
    )
    p.add_argument("--docs", type=int, default=2000, help="Number of documents to generate (docs mode)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (deterministic output)")
    p.add_argument("--min-words", type=int, default=120, help="Minimum words per document")
    p.add_argument("--max-words", type=int, default=320, help="Maximum words per document")
    p.add_argument(
        "--target-mib",
        type=float,
        default=None,
        help="Target corpus size in MiB (enables sharded generation mode)",
    )
    p.add_argument(
        "--target-gib",
        type=float,
        default=None,
        help="Target corpus size in GiB (enables sharded generation mode)",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Number of shard files to write when using --target-* (default: 64)",
    )
    p.add_argument("--jsonl", action="store_true", help="Also write corpus.jsonl")
    p.add_argument("--combined", action="store_true", help="Also write combined.txt")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)

    if args.docs <= 0:
        raise SystemExit("--docs must be > 0")
    if args.min_words <= 0 or args.max_words <= 0 or args.min_words > args.max_words:
        raise SystemExit("--min-words/--max-words must be positive and min<=max")

    if args.target_mib is not None and args.target_gib is not None:
        raise SystemExit("Use only one of --target-mib or --target-gib")

    target_bytes: int | None = None
    if args.target_mib is not None:
        if args.target_mib <= 0:
            raise SystemExit("--target-mib must be > 0")
        target_bytes = int(args.target_mib * 1024 * 1024)
    if args.target_gib is not None:
        if args.target_gib <= 0:
            raise SystemExit("--target-gib must be > 0")
        target_bytes = int(args.target_gib * 1024 * 1024 * 1024)

    manifest = generate(
        out_dir=out_dir,
        docs=args.docs,
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
        write_jsonl=bool(args.jsonl),
        write_combined=bool(args.combined),
        target_bytes=target_bytes,
        shards=args.shards,
    )

    mb = manifest.total_bytes / (1024 * 1024)
    print(
        "Generated corpus:\n"
        f"- out_dir: {manifest.out_dir}\n"
        f"- mode: {manifest.mode}\n"
        f"- docs: {manifest.docs}\n"
        f"- shards: {manifest.shards or '-'}\n"
        f"- target_bytes: {manifest.target_bytes or '-'}\n"
        f"- approx size: {mb:.2f} MiB\n"
        f"- combined: {manifest.combined_path or '-'}\n"
        f"- jsonl: {manifest.jsonl_path or '-'}\n"
    )


if __name__ == "__main__":
    main()
