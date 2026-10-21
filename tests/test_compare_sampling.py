import json

import pytest

from core.model_backend import ModelBackend
from core.types import Candidate
from evaluation import compare_sampling as sampling


class GenerationBackend(ModelBackend):
    name = "fixture_generation"

    def __init__(self, stop_after_first=False):
        self.stop_after_first = stop_after_first
        self.sample_calls = 0

    def start_request(self, seed):
        self.seed = seed

    def generate(self, prompt, n, parameters, deadline=None):
        if parameters["temperature"] == 0:
            sqls = ["SELECT COUNT(*) FROM students"]
        else:
            self.sample_calls += 1
            sqls = ["SELECT COUNT(*) FROM classes"] * n
            if self.seed % 1000 == 43:
                sqls[-1] = "SELECT COUNT(*) FROM students"
            if self.stop_after_first and self.sample_calls == 1:
                sampling.STOP = True
        return [Candidate(i, raw_text=f"<answer>{sql}</answer>") for i, sql in enumerate(sqls)]


class MergeBackend(ModelBackend):
    name = "fixture_merge"

    def start_request(self, seed):
        pass

    def generate(self, prompt, n, parameters, deadline=None):
        assert "Execution result" in prompt
        return [
            Candidate(i, raw_text="<answer>SELECT COUNT(*) FROM students</answer>")
            for i in range(n)
        ]


def test_checkpoint_partial_resume_and_curves(db, config, tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "KS", (4, 8))
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "question": "Có bao nhiêu sinh viên?",
                    "gold_sql": "SELECT COUNT(*) FROM students",
                    "db_path": str(db),
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    first_backend = GenerationBackend(stop_after_first=True)
    partial = sampling.compare(
        config,
        dataset,
        output,
        backend=first_backend,
        merge_backend=MergeBackend(),
        merge_samples=2,
    )
    assert partial["checkpoints"] == 1
    assert next(x for x in partial["metrics"] if x["metric"] == "greedy@1")["accuracy"] == 1
    assert next(x for x in partial["metrics"] if x["metric"] == "pass@4")["completed"] == 0
    state = json.loads((output / "checkpoints/00000.json").read_text())
    assert len(state["sampled"]) == 4

    resumed_backend = GenerationBackend()
    complete = sampling.compare(
        config,
        dataset,
        output,
        backend=resumed_backend,
        merge_backend=MergeBackend(),
        merge_samples=2,
    )
    assert resumed_backend.sample_calls == 1  # Only candidates 5-8 are regenerated.
    assert {x["metric"]: x["accuracy"] for x in complete["metrics"]} == {
        "greedy@1": 1,
        "self_consistency@4": 0,
        "slm_sql@4": 1,
        "pass@4": 1,
        "self_consistency@8": 0,
        "slm_sql@8": 1,
        "pass@8": 1,
    }
    assert len(json.loads((output / "checkpoints/00000.json").read_text())["sampled"]) == 8
    assert (output / "summary.csv").exists()
    with pytest.raises(ValueError, match="Checkpoint khac"):
        sampling.compare(
            config,
            dataset,
            output,
            backend=resumed_backend,
            merge_backend=MergeBackend(),
            merge_samples=3,
        )


def test_report_ignores_unfinished_stages(tmp_path):
    output = tmp_path / "run"
    sampling.atomic_json(
        output / "checkpoints/00000.json",
        {
            "index": 0,
            "id": "one",
            "metrics": {"greedy@1": False, "pass@4": True},
            "latency_ms": {"greedy@1": 1, "pass@4": 20},
        },
    )
    report = sampling.aggregate(
        output, {"questions": 2, "dataset_sha256": "hash", "model_id": "a", "merge_model_id": "b"}
    )
    assert next(x for x in report["metrics"] if x["metric"] == "pass@4")["accuracy"] == 1
    assert (
        next(x for x in report["metrics"] if x["metric"] == "pass@4")["average_total_generations"]
        == 4
    )
    assert next(x for x in report["metrics"] if x["metric"] == "slm_sql@4")["accuracy"] is None


def test_adaptive_uses_pipeline_and_is_checkpointed(db, config, tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "KS", (4,))
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "question": "Có bao nhiêu sinh viên?",
                    "gold_sql": "SELECT COUNT(*) FROM students",
                    "db_path": str(db),
                }
            ]
        ),
        encoding="utf-8",
    )
    backend = GenerationBackend()
    report = sampling.compare(
        config,
        dataset,
        tmp_path / "run",
        backend=backend,
        merge_backend=MergeBackend(),
        with_adaptive=True,
    )
    assert next(x for x in report["metrics"] if x["metric"] == "adaptive_sc")["accuracy"] == 0
    state = json.loads((tmp_path / "run/checkpoints/00000.json").read_text())
    assert state["adaptive"]["candidates"] == 4
    assert state["adaptive"]["stop_reason"] == "consensus"
    assert report["metrics"][-1]["metric"] == "adaptive_sc"
    assert report["metrics"][-1]["average_total_generations"] == 4
    assert (
        next(x for x in report["metrics"] if x["metric"] == "self_consistency@4")[
            "average_total_generations"
        ]
        == 4
    )
    assert (
        next(x for x in report["metrics"] if x["metric"] == "slm_sql@4")[
            "average_total_generations"
        ]
        == 12
    )


def test_resumed_prefix_latency_is_unchanged(db, config, tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "KS", (4, 8))
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "question": "Có bao nhiêu sinh viên?",
                    "gold_sql": "SELECT COUNT(*) FROM students",
                    "db_path": str(db),
                }
            ]
        ),
        encoding="utf-8",
    )

    class StopOnSecondChunk(GenerationBackend):
        def generate(self, prompt, n, parameters, deadline=None):
            result = super().generate(prompt, n, parameters, deadline)
            if self.sample_calls == 2:
                sampling.STOP = True
            return result

    output = tmp_path / "run"
    sampling.compare(
        config, dataset, output, backend=StopOnSecondChunk(), merge_backend=MergeBackend()
    )
    checkpoint = output / "checkpoints/00000.json"
    before = json.loads(checkpoint.read_text())["latency_ms"]["self_consistency@4"]
    sampling.compare(
        config, dataset, output, backend=GenerationBackend(), merge_backend=MergeBackend()
    )
    after = json.loads(checkpoint.read_text())["latency_ms"]["self_consistency@4"]
    assert after == before


def test_merge_model_failure_keeps_samples_and_explains_cause(db, config, tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "KS", (4,))
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "question": "Có bao nhiêu sinh viên?",
                    "gold_sql": "SELECT COUNT(*) FROM students",
                    "db_path": str(db),
                }
            ]
        ),
        encoding="utf-8",
    )

    def unavailable(_config):
        try:
            raise FileNotFoundError("merge weights missing")
        except FileNotFoundError as error:
            raise RuntimeError("Không tải được model") from error

    monkeypatch.setattr(sampling, "create_backend", unavailable)
    output = tmp_path / "run"
    report = sampling.compare(config, dataset, output, backend=GenerationBackend())
    state = json.loads((output / "checkpoints/00000.json").read_text())
    assert len(state["sampled"]) == 4
    assert "CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct" in state["error"]
    assert "merge weights missing" in state["error"]
    assert report["metrics"][0]["completed"] == 1

    resumed = sampling.compare(
        config, dataset, output, backend=GenerationBackend(), merge_backend=MergeBackend()
    )
    assert next(row for row in resumed["metrics"] if row["metric"] == "slm_sql@4")["completed"] == 1
