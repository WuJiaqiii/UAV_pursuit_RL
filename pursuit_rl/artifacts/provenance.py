"""Fingerprint package code and authoritative configuration resources."""

import hashlib
from pathlib import Path


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts
    )
    entry = root.parent / "main.py"
    if entry.is_file():
        digest.update(b"main.py")
        digest.update(entry.read_bytes())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rescue_source_fingerprint():
    """Include the new entry points and JSON profiles in rescue provenance."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(source_fingerprint().encode())
    files = list((root / "configs" / "rescue").glob("*"))
    files.extend((root / "configs" / "curriculum_v681").glob("*"))
    files.extend(
        (
            root / "scripts" / name
            for name in (
                "rescue.py",
                "run_rescue.sh",
                "challenge.py",
                "curriculum.py",
                "run_curriculum.sh",
            )
        )
    )
    for path in sorted(p for p in files if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
