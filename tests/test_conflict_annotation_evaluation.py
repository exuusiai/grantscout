import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_conflict_annotations.py"
SPEC = importlib.util.spec_from_file_location("evaluate_conflict_annotations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_silver_pilot_does_not_invent_recall(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "annotation_id": "a",
                "source_run_id": "r1",
                "papers": [{"paper_id": "p1"}, {"paper_id": "p2"}],
                "model_annotation": {"label": "not_comparable"},
                "annotator_a": {"label": None},
                "annotator_b": {"label": None},
                "adjudication": {"label": None},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"run_id": "r1", "conflicts": []}), encoding="utf-8")

    result = module.evaluate(annotations, [run])

    assert result["observed_false_positive_rate"] == 0.0
    assert result["recall"] is None
    assert result["recall_identifiable"] is False
    assert result["label_source"] == "silver_or_mixed"
