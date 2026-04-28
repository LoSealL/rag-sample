"""Eval API: upload evaluation JSON and visualize diff results."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()

# In-memory store for evaluations
_evaluations: dict[str, dict[str, Any]] = {}


class EvalItem(BaseModel):
    file: str
    name: str
    expected_type: str
    expected_start: int
    expected_end: int
    predicted_type: str | None = None
    predicted_start: int | None = None
    predicted_end: int | None = None
    match: bool = False


class EvalSummary(BaseModel):
    eval_id: str
    total: int
    matched: int
    accuracy: float
    items: list[EvalItem]


def _compute_diff(expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, Any]:
    """Compute diff between expected and predicted chunk."""
    if predicted is None:
        return {
            "type_match": False,
            "name_match": False,
            "start_diff": None,
            "end_diff": None,
            "exact_match": False,
        }

    type_match = expected.get("type") == predicted.get("type")
    name_match = expected.get("name") == predicted.get("name")
    exp_start = expected.get("line_start", 0)
    exp_end = expected.get("line_end", 0)
    pred_start = predicted.get("line_start", 0)
    pred_end = predicted.get("line_end", 0)

    return {
        "type_match": type_match,
        "name_match": name_match,
        "start_diff": abs(exp_start - pred_start) if pred_start else None,
        "end_diff": abs(exp_end - pred_end) if pred_end else None,
        "exact_match": type_match and name_match and exp_start == pred_start and exp_end == pred_end,
    }


@router.post("/upload")
async def upload_eval(file: UploadFile = File(...)) -> EvalSummary:
    """Upload a reference_labels.json and compute diff against predictions."""
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    eval_id = str(uuid4())[:8]
    items: list[EvalItem] = []
    matched = 0

    for entry in data if isinstance(data, list) else data.get("labels", []):
        expected = {
            "file": entry.get("file", ""),
            "name": entry.get("name", ""),
            "type": entry.get("type", ""),
            "line_start": entry.get("line_start"),
            "line_end": entry.get("line_end"),
        }

        # Try to find predicted match (simple name-based lookup)
        predicted = entry.get("predicted") or entry.get("best_match")
        diff = _compute_diff(expected, predicted)

        is_match = diff["exact_match"]
        if is_match:
            matched += 1

        items.append(
            EvalItem(
                file=expected["file"],
                name=expected["name"],
                expected_type=expected["type"],
                expected_start=expected["line_start"] or 0,
                expected_end=expected["line_end"] or 0,
                predicted_type=predicted.get("type") if predicted else None,
                predicted_start=predicted.get("line_start") if predicted else None,
                predicted_end=predicted.get("line_end") if predicted else None,
                match=is_match,
            )
        )

    total = len(items)
    accuracy = matched / total if total > 0 else 0.0

    summary = EvalSummary(
        eval_id=eval_id,
        total=total,
        matched=matched,
        accuracy=accuracy,
        items=items,
    )

    _evaluations[eval_id] = summary.model_dump()
    return summary


@router.get("/{eval_id}")
def get_eval(eval_id: str) -> dict[str, Any]:
    if eval_id not in _evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _evaluations[eval_id]


@router.get("/")
def list_evals() -> list[dict[str, Any]]:
    return [{"eval_id": k, "summary": v} for k, v in _evaluations.items()]
