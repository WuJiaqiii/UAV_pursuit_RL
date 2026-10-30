"""Strict JSON serialization for logs, reports and protocol metadata."""

import json
from pathlib import Path
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data, indent=2, ensure_ascii=False, allow_nan=False, default=json_default
        ),
        encoding="utf8",
    )


def log_jsonl(path, record):
    with Path(path).open("a", encoding="utf8") as stream:
        stream.write(
            json.dumps(
                record, ensure_ascii=False, allow_nan=False, default=json_default
            )
            + "\n"
        )
