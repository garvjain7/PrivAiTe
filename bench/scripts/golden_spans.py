"""Capture the default detector spans on the pinned structured corpus.

The temporary corpus checkout is removed when this command exits. The snapshot
contains IDs and span metadata only; it never serializes request text or values.
Run from the repository root with ``python bench/scripts/golden_spans.py``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

from pbench import sources, structured  # noqa: E402
from pbench.corpus import write_jsonl  # noqa: E402
from privaite.config.schema import PIIConfig  # noqa: E402
from privaite.pii.engine import PIIEngine  # noqa: E402

_SCRUB_SPEC = importlib.util.spec_from_file_location(
    "privaite_gateway_scrub", ROOT / "privaite" / "gateway" / "scrub.py"
)
assert _SCRUB_SPEC is not None and _SCRUB_SPEC.loader is not None
_SCRUB_MODULE = importlib.util.module_from_spec(_SCRUB_SPEC)
_SCRUB_SPEC.loader.exec_module(_SCRUB_MODULE)
scrub_anthropic_request = _SCRUB_MODULE.scrub_anthropic_request
scrub_responses_request = _SCRUB_MODULE.scrub_responses_request


def _prepare_documents(cache: Path) -> list[dict[str, Any]]:
    sources.RAW = cache / "raw"
    sources.CORPORA = cache / "corpora"
    structured.CORPORA = sources.CORPORA
    needed = {name for name, _, _ in structured.PLAN}
    for name in sorted(needed):
        builder = sources.BUILDERS[name]
        built = builder()
        docs = built[0] if isinstance(built, tuple) else built
        write_jsonl(sources.CORPORA / f"{name}.jsonl", docs)
    user_items = [item for item in structured.build() if item["carrier"] == "user_text"]
    return sorted(user_items, key=lambda item: (item["corpus"], item["doc"]))


async def _capture(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engine = PIIEngine(PIIConfig())
    await engine.initialize()
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    leaf = 0
    original_with_known_values = engine._with_known_values

    def record_spans(text: str, entities: list[Any]) -> None:
        nonlocal leaf
        for entity in entities:
            records.append(
                {
                    "route": current["route"],
                    "corpus": current["corpus"],
                    "doc": current["doc"],
                    "leaf": leaf,
                    "start": entity.start,
                    "end": entity.end,
                    "type": entity.entity_type,
                    "score": entity.score,
                    "source": entity.source,
                }
            )
        leaf += 1

    def capture_with_known_values(text: str, entities: list[Any], mapping: Any) -> list[Any]:
        result = original_with_known_values(text, entities, mapping)
        record_spans(text, result)
        return result

    engine._with_known_values = capture_with_known_values  # type: ignore[method-assign]
    try:
        for item in items:
            text = item["messages"][0]["content"]
            for route in ("core_chat", "gateway_messages", "gateway_responses"):
                current.update(route=route, corpus=item["corpus"], doc=item["doc"])
                leaf = 0
                if route == "core_chat":
                    await engine.process_request([{"role": "user", "content": text}])
                elif route == "gateway_messages":
                    body = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": text}],
                            }
                        ]
                    }
                    await scrub_anthropic_request(engine, body)
                else:
                    body = {
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            }
                        ]
                    }
                    await scrub_responses_request(engine, body)
    finally:
        await engine.shutdown()
    return records


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="privaite-golden-", dir=ROOT) as temp:
        items = _prepare_documents(Path(temp))
        records = asyncio.run(_capture(items))
    snapshot = {
        "schema": 1,
        "preset": "onnx",
        "documents": len(items),
        "routes": ["core_chat", "gateway_messages", "gateway_responses"],
        "spans": sorted(
            records,
            key=lambda span: (
                span["route"], span["corpus"], span["doc"], span["leaf"],
                span["start"], span["end"], span["type"], span["source"],
            ),
        ),
    }
    destination = ROOT / "bench" / "golden_spans.json"
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(snapshot['spans'])} metadata-only spans for {len(items)} documents.")


if __name__ == "__main__":
    main()
