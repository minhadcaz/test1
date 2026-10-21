"""Checkpointed BIRD sampling curves; local EX is not the official BIRD score."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from core.config import load_config
from core.model_backend import create_backend
from core.pipeline import Pipeline
from core.prompt_builder import build_prompt
from core.schema_inspector import SchemaInspector
from core.sql_executor import SQLExecutor
from core.sql_generator import generate_candidates
from core.sql_validator import SQLValidator
from databases.create_student_db import DEFAULT_DB, create_database
from evaluation.metrics import execution_match

KS = (4, 8, 16, 32, 64)
MERGE_MODEL = "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct"
STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def error_details(error: Exception) -> str:
    root = error
    while root.__cause__ is not None:
        root = root.__cause__
    summary = f"{type(error).__name__}: {error}"
    if root is not error:
        summary += f" | Root cause: {type(root).__name__}: {root}"
    return summary[:1200]


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def signature(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def group_votes(candidates: list[dict]) -> list[list[dict]]:
    groups = {}
    for candidate in candidates:
        key = candidate.get("fingerprint")
        if key is not None:
            groups.setdefault(key, []).append(candidate)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]["index"]))


def choice(candidates: list[dict]):
    groups = group_votes(candidates)
    return groups[0][0] if groups else None


def scores(candidates: list[dict]):
    selected = choice(candidates)
    return {
        "self_consistency": bool(selected and selected["correct"]),
        "pass": any(c["correct"] for c in candidates),
        "valid": sum(c.get("fingerprint") is not None for c in candidates),
    }


def merge_prompt(question: str, schema: dict, evidence: str, groups: list[list[dict]]) -> str:
    drafts = []
    for index, group in enumerate(groups[:2], 1):
        candidate = group[0]
        drafts.append(
            f"{index}. {candidate['sql']}\nExecution result (preview): "
            + json.dumps(candidate["preview"], ensure_ascii=False)
        )
    return (
        "You are a data science expert. Compare the two draft SQLite queries and their "
        "execution results against the question. Select the correct draft query; do not "
        "combine them. Reason in <think> tags and give one final SELECT or WITH query "
        "inside <answer> tags.\nDatabase Schema:\n"
        + SchemaInspector.format(schema)
        + "\nQuestion: "
        + json.dumps({"evidence": evidence, "question": question}, ensure_ascii=False)
        + "\nDraft SQL and execution results:\n"
        + "\n".join(drafts)
    )


def checked(
    candidate,
    database: Path,
    validator: SQLValidator,
    executor: SQLExecutor,
    gold,
    gold_sql: str,
    index: int,
) -> dict:
    record = {
        "index": index,
        "sql": candidate.sql,
        "error": candidate.error,
        "fingerprint": None,
        "correct": False,
        "preview": [],
        "generation_ms": candidate.generation_ms,
    }
    if candidate.error or not candidate.sql:
        return record
    validation = validator.validate(candidate.sql, database)
    if not (validation.is_safe and validation.is_valid):
        record["error"] = "; ".join(validation.errors)
        return record
    result = executor.execute(database, candidate.sql)
    if result.success:
        record["fingerprint"] = result.normalized_result
        record["correct"] = execution_match(result, gold, gold_sql)
        record["preview"] = result.rows[:5]
    else:
        record["error"] = result.error
    return record


def aggregate(output: Path, manifest: dict) -> dict:
    """Write partial results based only on completed stages and explicit denominators."""
    all_states = []
    records = output / "checkpoints"
    for index in range(manifest["questions"]):
        path = records / f"{index:05}.json"
        if path.exists():
            all_states.append(json.loads(path.read_text(encoding="utf-8")))
    keys = ["greedy@1"] + [
        f"{name}@{k}" for k in KS for name in ("self_consistency", "slm_sql", "pass")
    ]
    if manifest.get("with_adaptive"):
        keys.append("adaptive_sc")
    rows = []
    for key in keys:
        completed = [s for s in all_states if key in s.get("metrics", {})]
        correct = sum(s["metrics"][key] for s in completed)
        elapsed = sum(s.get("latency_ms", {}).get(key, 0) for s in completed)
        if key == "adaptive_sc":
            generations = sum(
                s["adaptive"]["candidates"] + s["adaptive"]["correction_attempts"]
                for s in completed
            )
        elif key == "greedy@1":
            generations = len(completed)
        else:
            k = int(key.rsplit("@", 1)[1])
            generations = sum(
                k
                + (
                    len(s["merges"][str(k)]["candidates"])
                    if key.startswith("slm_sql@") and s["merges"][str(k)]["status"] == "revised"
                    else 0
                )
                for s in completed
            )
        rows.append(
            {
                "metric": key,
                "completed": len(completed),
                "total": manifest["questions"],
                "correct": correct,
                "accuracy": correct / len(completed) if completed else None,
                "average_latency_ms": elapsed / len(completed) if completed else None,
                "average_total_generations": generations / len(completed) if completed else None,
            }
        )
    errors = [
        {"index": s["index"], "id": s["id"], "error": s["error"]}
        for s in all_states
        if s.get("error")
    ]
    report = {
        "dataset_sha256": manifest["dataset_sha256"],
        "model_id": manifest["model_id"],
        "merge_model_id": manifest["merge_model_id"],
        "questions": manifest["questions"],
        "checkpoints": len(all_states),
        "errors": errors,
        "metrics": rows,
        "note": "Local EX; slm_sql uses a separate merge model, not official BIRD evaluation.",
    }
    atomic_json(output / "summary.json", report)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output / "summary.csv")
    return report


def compare(
    config: dict,
    dataset: Path,
    output: Path,
    *,
    limit: int | None = None,
    database: Path = DEFAULT_DB,
    merge_model_id: str = MERGE_MODEL,
    merge_samples: int = 8,
    with_adaptive: bool = False,
    backend=None,
    merge_backend=None,
) -> dict:
    global STOP
    STOP = False
    dataset = dataset.resolve(strict=True)
    raw = dataset.read_bytes()
    items = json.loads(raw)
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset phai la JSON list khong rong.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit phai duong.")
        if limit > len(items):
            print(
                f"Dataset chi co {len(items)} cau; --limit {limit} khong tao them cau.", flush=True
            )
        items = items[:limit]
    if database.resolve() == DEFAULT_DB.resolve():
        create_database()
    if merge_samples < 1:
        raise ValueError("--merge-samples phai duong.")
    run_config = deepcopy(config)
    run_config["model"].update(
        backend="huggingface" if backend is None else config["model"]["backend"]
    )
    run_config["generation"].update(temperature=0.8)
    run_config["correction"]["enabled"] = False
    manifest = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "questions": len(items),
        "database": str(database.resolve()),
        "config": run_config,
        "model_id": run_config["model"]["model_id"],
        "merge_model_id": merge_model_id,
        "merge_samples": merge_samples,
        "with_adaptive": with_adaptive,
        "format_version": 1,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if signature(existing) != signature(manifest):
            raise ValueError("Checkpoint khac dataset/config/model; dung --output moi.")
    else:
        if any((output / "checkpoints").glob("*.json")):
            raise ValueError("Co checkpoint nhung thieu manifest; dung output moi.")
        atomic_json(manifest_path, manifest)
    aggregate(output, manifest)
    validator = SQLValidator()
    executor = SQLExecutor(run_config["execution"])
    generator = backend  # Load only when a stage needs generation.
    merger = merge_backend
    for index, item in enumerate(items):
        if STOP:
            break
        path = output / "checkpoints" / f"{index:05}.json"
        state = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {
                "index": index,
                "id": item["id"],
                "metrics": {},
                "latency_ms": {},
                "sampled": [],
                "merges": {},
            }
        )
        if state["id"] != item["id"]:
            raise ValueError(f"Checkpoint ID khac dataset tai vi tri {index}.")
        if state.get("retryable"):
            state.pop("error", None)
            state.pop("retryable", None)
        if state.get("error") or (
            f"slm_sql@{KS[-1]}" in state["metrics"]
            and (not with_adaptive or "adaptive_sc" in state["metrics"])
        ):
            continue
        try:
            db = (dataset.parent / item["db_path"]).resolve() if item.get("db_path") else database
            gold = executor.execute(db, item["gold_sql"])
            if not gold.success:
                raise ValueError(f"Gold SQL loi: {gold.error}")
            schema_start = time.perf_counter()
            schema = SchemaInspector(**run_config["schema"]).inspect(db)
            prompt = build_prompt(item["question"], schema, item.get("evidence", ""))
            state.setdefault("schema_ms", (time.perf_counter() - schema_start) * 1000)
            if generator is None:
                generator = create_backend(run_config)
            if "greedy@1" not in state["metrics"]:
                generator.start_request(run_config["seed"] + index * 1000)
                params = dict(run_config["generation"], temperature=0)
                started = time.perf_counter()
                result = generate_candidates(generator, prompt, 1, params)
                predicted = (
                    checked(result[0], db, validator, executor, gold, item["gold_sql"], 0)
                    if result
                    else None
                )
                state["greedy"] = predicted
                state["metrics"]["greedy@1"] = bool(predicted and predicted["correct"])
                state["latency_ms"]["greedy@1"] = (
                    state["schema_ms"] + (time.perf_counter() - started) * 1000
                )
                atomic_json(path, state)
            if with_adaptive and "adaptive_sc" not in state["metrics"] and not STOP:
                adaptive_config = deepcopy(run_config)
                adaptive_config["seed"] = run_config["seed"] + index * 1000
                adaptive_config["sampling"]["mode"] = "adaptive_sc"
                adaptive_config["correction"]["enabled"] = config["correction"]["enabled"]
                result = Pipeline(adaptive_config, generator).run(
                    db, item["question"], item.get("evidence", "")
                )
                selected = result.selected
                state["metrics"]["adaptive_sc"] = execution_match(
                    selected.execution if selected else None, gold, item["gold_sql"]
                )
                state["latency_ms"]["adaptive_sc"] = result.elapsed_ms
                state["adaptive"] = {
                    "selected_sql": selected.sql if selected else None,
                    "candidates": len(result.candidates),
                    "correction_attempts": result.correction_attempts,
                    "stop_reason": result.decision.reason,
                }
                atomic_json(path, state)
            for k in KS:
                if STOP:
                    break
                while len(state["sampled"]) < k and not STOP:
                    offset = len(state["sampled"])
                    generator.start_request(run_config["seed"] + index * 1000 + offset + 1)
                    requested = min(4, k - offset)
                    started = time.perf_counter()
                    generated = generate_candidates(
                        generator, prompt, requested, run_config["generation"], offset
                    )
                    if len(generated) != requested:
                        raise RuntimeError(
                            f"Sinh {len(generated)}/{requested} ung vien tai {item['id']}."
                        )
                    state["sampled"].extend(
                        checked(c, db, validator, executor, gold, item["gold_sql"], offset + j)
                        for j, c in enumerate(generated)
                    )
                    state["sample_ms"] = (
                        state.get("sample_ms", 0) + (time.perf_counter() - started) * 1000
                    )
                    atomic_json(path, state)
                if STOP:
                    break
                prefix = state["sampled"][:k]
                prefix_scores = scores(prefix)
                for name in ("self_consistency", "pass"):
                    key = f"{name}@{k}"
                    if key not in state["metrics"]:
                        state["metrics"][key] = prefix_scores[name]
                        state["latency_ms"][key] = state["schema_ms"] + state["sample_ms"]
                        atomic_json(path, state)
                if f"slm_sql@{k}" not in state["metrics"]:
                    groups = group_votes(prefix)
                    if len(groups) < 2:
                        selected = choice(prefix)
                        state["merges"][str(k)] = {"status": "not_needed", "selected": selected}
                        merge_ms = 0
                    else:
                        if merger is None:
                            merge_config = deepcopy(run_config)
                            merge_config["model"]["model_id"] = merge_model_id
                            try:
                                merger = create_backend(merge_config)
                            except Exception as error:
                                raise RuntimeError(
                                    f"Khong tai duoc merge model {merge_model_id}; "
                                    "kiem tra weights va VRAM."
                                ) from error
                        merger.start_request(run_config["seed"] + index * 1000 + 500 + k)
                        merge_params = dict(run_config["generation"], temperature=0.8)
                        started = time.perf_counter()
                        revisions = generate_candidates(
                            merger,
                            merge_prompt(
                                item["question"], schema, item.get("evidence", ""), groups
                            ),
                            merge_samples,
                            merge_params,
                        )
                        if len(revisions) != merge_samples:
                            raise RuntimeError(
                                f"Merge sinh {len(revisions)}/{merge_samples} tai {item['id']}."
                            )
                        merge_records = [
                            checked(c, db, validator, executor, gold, item["gold_sql"], j)
                            for j, c in enumerate(revisions)
                        ]
                        selected = choice(merge_records) or choice(prefix)
                        merge_ms = (time.perf_counter() - started) * 1000
                        state["merges"][str(k)] = {
                            "status": "revised",
                            "candidates": merge_records,
                            "selected": selected,
                            "elapsed_ms": merge_ms,
                        }
                    state["metrics"][f"slm_sql@{k}"] = bool(selected and selected["correct"])
                    state["latency_ms"][f"slm_sql@{k}"] = (
                        state["schema_ms"] + state["sample_ms"] + merge_ms
                    )
                    atomic_json(path, state)
            aggregate(output, manifest)
            print(
                f"{index + 1}/{len(items)} {item['id']} sampled={len(state['sampled'])} "
                f"completed={f'slm_sql@{KS[-1]}' in state['metrics']}",
                flush=True,
            )
        except KeyboardInterrupt:
            STOP = True
        except Exception as error:
            state["error"] = error_details(error)
            state["retryable"] = isinstance(error, (RuntimeError, ImportError))
            atomic_json(path, state)
            aggregate(output, manifest)
            print(f"{index + 1}/{len(items)} {item['id']} ERROR: {state['error']}", flush=True)
            # An unavailable model or incompatible environment should not mark all questions bad.
            if isinstance(error, (RuntimeError, ImportError)):
                break
    return aggregate(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild partial summary from checkpoints after forced termination.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-id", default="cycloneboy/SLM-SQL-1.5B")
    parser.add_argument("--merge-model-id", default=MERGE_MODEL)
    parser.add_argument("--merge-samples", type=int, default=8)
    parser.add_argument(
        "--with-adaptive",
        action="store_true",
        help="Also run the repository's adaptive_sc Pipeline on each question.",
    )
    args = parser.parse_args()
    if args.report_only:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps(aggregate(args.output, manifest), ensure_ascii=False, indent=2))
        return
    if args.dataset is None:
        parser.error("--dataset is required unless --report-only is set")
    config = load_config(args.config)
    config["model"].update(backend="huggingface", model_id=args.model_id, device=args.device)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    summary = compare(
        config,
        args.dataset,
        args.output,
        limit=args.limit,
        database=args.db,
        merge_model_id=args.merge_model_id,
        merge_samples=args.merge_samples,
        with_adaptive=args.with_adaptive,
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"Partial/final report: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
