import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "create_annotation_template.py"
SPEC = importlib.util.spec_from_file_location("create_annotation_template", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_annotation_template_is_blinded_and_stable(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "q1", "query": "Claim", "evidence": ["Passage"]}) + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    assert module.build_template(source, first) == 1
    assert module.build_template(source, second) == 1
    row = json.loads(first.read_text(encoding="utf-8").strip())
    assert row["claim"] == "Claim"
    assert row["evidence"] == "Passage"
    assert row["annotator_a"]["label"] is None
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_annotation_validation_reports_completion_and_agreement(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    rows = [
        {"annotation_id": "a", "annotator_a": {"label": "entailed"}, "annotator_b": {"label": "entailed"}},
        {"annotation_id": "b", "annotator_a": {"label": "entailed"}, "annotator_b": {"label": "insufficient"}},
        {"annotation_id": "c", "annotator_a": {"label": None}, "annotator_b": {"label": None}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = module.validate_annotations(path)

    assert result["completed_pairs"] == 2
    assert result["raw_agreement"] == 0.5
    assert result["invalid_or_incomplete_ids"] == ["c"]
