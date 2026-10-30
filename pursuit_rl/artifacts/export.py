#!/usr/bin/env python3
"""Export training summary and per-episode metrics without .pt checkpoints.

Preserves JSON/JSONL/CSV/configs/README; omits model checkpoints, tensors,
large trajectory archives, TensorBoard binary files and arbitrary logs.
"""
import hashlib, json, zipfile
from pathlib import Path

INCLUDED = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".md", ".txt"}


def export(source, out, max_file_mb=15, exclude_paths=()):
    source = Path(source).resolve()
    out = Path(out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if max_file_mb <= 0:
        raise ValueError("max_file_mb must be positive")
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    excluded = [Path(path).resolve() for path in exclude_paths]
    count = 0
    manifest = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(source.rglob("*")):
            if any(f.resolve().is_relative_to(path) for path in excluded):
                continue
            if not f.is_file() or f.suffix.lower() not in INCLUDED:
                continue
            if any(
                part in ("tensorboard", "__pycache__", "traces")
                for part in f.relative_to(source).parts
            ):
                continue
            if f.stat().st_size > max_file_mb * 1024 * 1024:
                continue
            payload = f.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            arc = f.relative_to(source).as_posix()
            z.writestr(arc, payload)
            manifest.append({"name": arc, "size_bytes": len(payload), "sha256": sha})
            count += 1
        z.writestr(
            "LIGHTWEIGHT_MANIFEST.json", json.dumps({"files": manifest}, indent=2)
        )
    return count
