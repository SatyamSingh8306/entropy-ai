"""Dataset container + loaders (json / jsonl / yaml / csv / hf) — spec §12."""
from __future__ import annotations

import csv
import json
import pathlib
from typing import Iterable, List

from .cases import Case, make_case


class Dataset(list):
    """A list of :class:`Case` with multi-format loaders."""

    @classmethod
    def from_json(cls, path: str) -> "Dataset":
        data = json.loads(pathlib.Path(path).read_text())
        return cls(_from_payload(data))

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        cases = []
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                cases.append(make_case(json.loads(line)))
        return cls(cases)

    @classmethod
    def from_yaml(cls, path: str) -> "Dataset":
        import yaml  # pyyaml
        data = yaml.safe_load(pathlib.Path(path).read_text())
        return cls(_from_payload(data))

    @classmethod
    def from_csv(cls, path: str) -> "Dataset":
        cases = []
        with pathlib.Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cases.append(make_case({
                    "input": row.get("input"),
                    "expected": row.get("expected"),
                    "type": row.get("type", "case"),
                    "metadata": {"row": dict(row)},
                }))
        return cls(cases)

    @classmethod
    def from_hf(cls, name: str, split: str = "train") -> "Dataset":
        """Load a HuggingFace dataset (requires the ``hf`` extra)."""
        from datasets import load_dataset  # datasets
        ds = load_dataset(name, split=split)
        cases = [make_case({"input": row.get("input"), "expected": row.get("expected")})
                 for row in ds]
        return cls(cases)

    def save(self, path: str) -> None:
        payload = [{"type": "case", "input": c.input, "expected": c.expected,
                    "metadata": c.metadata} for c in self]
        pathlib.Path(path).write_text(json.dumps(payload, indent=2, default=str))


def _from_payload(data: object) -> List[Case]:
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    return [make_case(i) for i in data]
