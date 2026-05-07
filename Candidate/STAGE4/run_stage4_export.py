#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 S3R export CLI.

This pilot implementation is intentionally deterministic and local:
- `export-all` reads a JSON config and writes the seven required JSONL files.
- validation checks references and character offsets.
- quality checks release-critical structural coverage.
- manifest generation writes checksums and dataset stats.
- `release` runs all gates and writes a RELEASED marker only when they pass.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_FILES = {
    "nodes": "nodes.jsonl",
    "evidence_chunks": "evidence_chunks.jsonl",
    "span_supervision": "span_supervision.jsonl",
    "retrieval_pairs": "retrieval_pairs.jsonl",
    "rerank_pairs": "rerank_pairs.jsonl",
    "abstention_samples": "abstention_samples.jsonl",
    "graph_edges": "graph_edges.jsonl",
}

RETRIEVAL_NEGATIVE_TYPES = {
    "ontology_sibling",
    "ontology_parent",
    "ontology_child",
    "embedding_neighbor",
    "random",
}

ABSTENTION_REASONS = {
    "span_is_not_phenotype",
    "phenotype_but_all_candidates_wrong",
    "ambiguous_or_incomplete_span",
    "negated_phenotype",
}


@dataclass
class JsonlRow:
    path: str
    line_no: int
    obj: Dict[str, Any]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(os.path.dirname(path))
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterable[JsonlRow]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield JsonlRow(path=path, line_no=line_no, obj=obj)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def as_root_dir(path: str) -> str:
    if os.path.basename(os.path.normpath(path)) == "processed":
        return os.path.dirname(os.path.normpath(path))
    return path


def processed_dir(root_or_processed: str) -> str:
    if os.path.basename(os.path.normpath(root_or_processed)) == "processed":
        return root_or_processed
    return os.path.join(root_or_processed, "processed")


def processed_path(data_dir: str, key: str) -> str:
    return os.path.join(processed_dir(data_dir), REQUIRED_FILES[key])


def manifest_dir(root_or_processed: str) -> str:
    return os.path.join(as_root_dir(root_or_processed), "manifests")


def validation_dir(root_or_processed: str) -> str:
    return os.path.join(as_root_dir(root_or_processed), "validation")


def load_ids(data_dir: str) -> Tuple[Set[str], Set[str]]:
    node_ids: Set[str] = set()
    evidence_ids: Set[str] = set()

    for row in read_jsonl(processed_path(data_dir, "nodes")):
        node_id = row.obj.get("node_id")
        if isinstance(node_id, str) and node_id:
            node_ids.add(node_id)

    for row in read_jsonl(processed_path(data_dir, "evidence_chunks")):
        evidence_id = row.obj.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            evidence_ids.add(evidence_id)

    return node_ids, evidence_ids


def add_error(errors: List[Dict[str, Any]], row: JsonlRow, field: str, value: Any, message: str) -> None:
    errors.append({
        "file": os.path.basename(row.path),
        "line": row.line_no,
        "field": field,
        "value": value,
        "error": message,
    })


def require_node(
    errors: List[Dict[str, Any]],
    row: JsonlRow,
    node_ids: Set[str],
    field: str,
    value: Any,
    *,
    allow_null: bool = False,
) -> int:
    if value is None and allow_null:
        return 0
    if not isinstance(value, str) or value not in node_ids:
        add_error(errors, row, field, value, "node_id not found in nodes.jsonl")
        return 1
    return 0


def require_evidence(
    errors: List[Dict[str, Any]],
    row: JsonlRow,
    evidence_ids: Set[str],
    field: str,
    value: Any,
    *,
    allow_synthetic_abs: bool = True,
) -> int:
    if allow_synthetic_abs and isinstance(value, str) and value.startswith("synthetic_abstention_"):
        return 0
    if not isinstance(value, str) or value not in evidence_ids:
        add_error(errors, row, field, value, "evidence_id not found in evidence_chunks.jsonl")
        return 1
    return 0


def validate_references(data_dir: str) -> Dict[str, Any]:
    node_ids, evidence_ids = load_ids(data_dir)
    errors: List[Dict[str, Any]] = []
    total = 0

    for row in read_jsonl(processed_path(data_dir, "nodes")):
        total += 1
        require_node(errors, row, node_ids, "inherited_from", row.obj.get("inherited_from"), allow_null=True)

    for row in read_jsonl(processed_path(data_dir, "evidence_chunks")):
        for i, node_id in enumerate(row.obj.get("linked_hpo") or []):
            total += 1
            require_node(errors, row, node_ids, f"linked_hpo[{i}]", node_id)
        for i, span in enumerate(row.obj.get("span_offsets") or []):
            total += 1
            require_node(errors, row, node_ids, f"span_offsets[{i}].hpo_id", span.get("hpo_id"))

    for row in read_jsonl(processed_path(data_dir, "span_supervision")):
        total += 1
        require_evidence(errors, row, evidence_ids, "source_evidence_id", row.obj.get("source_evidence_id"))
        for i, span in enumerate(row.obj.get("spans") or []):
            total += 1
            require_node(errors, row, node_ids, f"spans[{i}].label", span.get("label"))

    for row in read_jsonl(processed_path(data_dir, "retrieval_pairs")):
        total += 1
        require_evidence(errors, row, evidence_ids, "source_evidence_id", row.obj.get("source_evidence_id"))
        total += 1
        require_node(errors, row, node_ids, "positive_node_id", row.obj.get("positive_node_id"))
        for i, evidence_id in enumerate(row.obj.get("positive_evidence_ids") or []):
            total += 1
            require_evidence(errors, row, evidence_ids, f"positive_evidence_ids[{i}]", evidence_id)
        for i, neg in enumerate(row.obj.get("hard_negatives") or []):
            total += 1
            require_node(errors, row, node_ids, f"hard_negatives[{i}].node_id", neg.get("node_id"))

    for row in read_jsonl(processed_path(data_dir, "rerank_pairs")):
        total += 1
        require_evidence(errors, row, evidence_ids, "source_evidence_id", row.obj.get("source_evidence_id"))
        total += 1
        require_node(errors, row, node_ids, "candidate_node_id", row.obj.get("candidate_node_id"))
        for i, evidence_id in enumerate(row.obj.get("evidence_ids") or []):
            total += 1
            require_evidence(errors, row, evidence_ids, f"evidence_ids[{i}]", evidence_id)

    for row in read_jsonl(processed_path(data_dir, "abstention_samples")):
        total += 1
        require_evidence(errors, row, evidence_ids, "source_evidence_id", row.obj.get("source_evidence_id"))
        for i, node_id in enumerate(row.obj.get("candidate_node_ids") or []):
            total += 1
            require_node(errors, row, node_ids, f"candidate_node_ids[{i}]", node_id)

    for row in read_jsonl(processed_path(data_dir, "graph_edges")):
        total += 1
        require_node(errors, row, node_ids, "source_id", row.obj.get("source_id"))
        total += 1
        require_node(errors, row, node_ids, "target_id", row.obj.get("target_id"))

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "stats": {
            "total_references_checked": total,
            "broken_references": len(errors),
            "n_nodes": len(node_ids),
            "n_evidence_chunks": len(evidence_ids),
        },
    }


def check_offset(
    errors: List[Dict[str, Any]],
    row: JsonlRow,
    text: Any,
    start: Any,
    end: Any,
    expected: Any,
    field_prefix: str,
) -> int:
    if not isinstance(text, str) or not isinstance(expected, str):
        add_error(errors, row, field_prefix, expected, "text or span text is not string")
        return 1
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(text):
        add_error(errors, row, field_prefix, {"start": start, "end": end}, "invalid offset range")
        return 1
    actual = text[start:end]
    if actual != expected:
        add_error(errors, row, field_prefix, {"expected": expected, "actual": actual, "start": start, "end": end}, "offset text mismatch")
        return 1
    return 0


def validate_offsets(data_dir: str) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    total = 0

    for row in read_jsonl(processed_path(data_dir, "evidence_chunks")):
        text = row.obj.get("text")
        if isinstance(text, str) and row.obj.get("char_length") != len(text):
            add_error(errors, row, "char_length", row.obj.get("char_length"), f"expected {len(text)}")
        for i, span in enumerate(row.obj.get("span_offsets") or []):
            total += 1
            check_offset(errors, row, text, span.get("start"), span.get("end"), span.get("text"), f"span_offsets[{i}]")

    for row in read_jsonl(processed_path(data_dir, "span_supervision")):
        text = row.obj.get("text")
        if isinstance(text, str) and row.obj.get("char_length") != len(text):
            add_error(errors, row, "char_length", row.obj.get("char_length"), f"expected {len(text)}")
        for i, span in enumerate(row.obj.get("spans") or []):
            total += 1
            check_offset(errors, row, text, span.get("start"), span.get("end"), span.get("text"), f"spans[{i}]")

    for row in read_jsonl(processed_path(data_dir, "retrieval_pairs")):
        total += 1
        check_offset(errors, row, row.obj.get("query_context"), row.obj.get("span_start_in_context"), row.obj.get("span_end_in_context"), row.obj.get("query_span"), "query_span")

    for row in read_jsonl(processed_path(data_dir, "rerank_pairs")):
        total += 1
        check_offset(errors, row, row.obj.get("query"), row.obj.get("span_start"), row.obj.get("span_end"), row.obj.get("span"), "span")

    for row in read_jsonl(processed_path(data_dir, "abstention_samples")):
        total += 1
        check_offset(errors, row, row.obj.get("text"), row.obj.get("span_start"), row.obj.get("span_end"), row.obj.get("span"), "span")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "stats": {
            "total_offsets_checked": total,
            "offset_errors": len(errors),
        },
    }


def export_all(config_path: str, output_root: str) -> Dict[str, Any]:
    cfg = read_json(config_path)
    processed = os.path.join(output_root, "processed")
    manifests = os.path.join(output_root, "manifests")
    validation = os.path.join(output_root, "validation")
    indexes = os.path.join(output_root, "indexes")
    raw = os.path.join(output_root, "raw_baseline_outputs")
    for path in [processed, manifests, validation, indexes, raw]:
        ensure_dir(path)

    rows_cfg = cfg.get("processed") or {}
    counts: Dict[str, int] = {}
    for key, filename in REQUIRED_FILES.items():
        rows = rows_cfg.get(key)
        if rows is None:
            rows = rows_cfg.get(filename)
        if rows is None:
            raise ValueError(f"config missing processed rows for {key}")
        if not isinstance(rows, list):
            raise ValueError(f"config field {key} must be a list")
        counts[filename] = write_jsonl(os.path.join(processed, filename), rows)

    baseline_manifest = {
        "created_at": utc_now(),
        "source": "stage4_pilot_export",
        "config_path": os.path.abspath(config_path),
        "notes": cfg.get("notes", []),
        "stage_versions": cfg.get("stage_versions", {}),
    }
    write_json(os.path.join(manifests, "baseline_manifest.json"), baseline_manifest)
    write_json(os.path.join(manifests, "pilot_export_config_snapshot.json"), cfg)

    splits_manifest = build_splits_manifest(processed, cfg)
    write_json(os.path.join(manifests, "splits_manifest.json"), splits_manifest)

    manifest = build_s3r_manifest(output_root, cfg, validation_summary=None)
    write_json(os.path.join(manifests, "s3r_data_manifest.json"), manifest)

    return {
        "passed": True,
        "output_root": output_root,
        "counts": counts,
        "manifests": {
            "baseline_manifest": os.path.join(manifests, "baseline_manifest.json"),
            "splits_manifest": os.path.join(manifests, "splits_manifest.json"),
            "s3r_data_manifest": os.path.join(manifests, "s3r_data_manifest.json"),
        },
    }


def build_splits_manifest(processed: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    stats_per_file: Dict[str, Dict[str, int]] = {}
    for key, filename in REQUIRED_FILES.items():
        split_counts: Dict[str, int] = {}
        path = os.path.join(processed, filename)
        for row in read_jsonl(path):
            split = row.obj.get("split")
            if isinstance(split, str):
                split_counts[split] = split_counts.get(split, 0) + 1
        stats_per_file[filename] = split_counts
    return {
        "split_version": cfg.get("split_version", "pilot_v1.0"),
        "main_split": cfg.get("main_split", {
            "strategy": "by_pmid_hash",
            "seed": 42,
            "ratios": {"train": 1.0, "dev": 0.0, "test": 0.0},
        }),
        "special_evals": cfg.get("special_evals", []),
        "stats_per_file": stats_per_file,
    }


def file_checksums(processed: str) -> Dict[str, str]:
    return {
        filename: sha256_file(os.path.join(processed, filename))
        for filename in REQUIRED_FILES.values()
    }


def collect_stats(processed: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    stats["n_nodes"] = count_jsonl(os.path.join(processed, "nodes.jsonl"))
    stats["n_evidence_chunks"] = count_jsonl(os.path.join(processed, "evidence_chunks.jsonl"))
    stats["n_span_supervision"] = count_jsonl(os.path.join(processed, "span_supervision.jsonl"))
    stats["n_retrieval_pairs"] = count_jsonl(os.path.join(processed, "retrieval_pairs.jsonl"))
    stats["n_rerank_pairs"] = count_jsonl(os.path.join(processed, "rerank_pairs.jsonl"))
    stats["n_abstention_samples"] = count_jsonl(os.path.join(processed, "abstention_samples.jsonl"))
    stats["n_graph_edges"] = count_jsonl(os.path.join(processed, "graph_edges.jsonl"))

    tiers: Dict[str, int] = {}
    for row in read_jsonl(os.path.join(processed, "nodes.jsonl")):
        tier = row.obj.get("tier")
        if isinstance(tier, str):
            tiers[tier] = tiers.get(tier, 0) + 1
    stats["n_nodes_by_tier"] = tiers

    styles: Dict[str, int] = {}
    for row in read_jsonl(os.path.join(processed, "evidence_chunks.jsonl")):
        style = row.obj.get("style")
        if isinstance(style, str):
            styles[style] = styles.get(style, 0) + 1
    stats["n_evidence_by_style"] = styles

    implicit = 0
    for row in read_jsonl(os.path.join(processed, "span_supervision.jsonl")):
        for span in row.obj.get("spans") or []:
            if span.get("expression_type") == "implicit":
                implicit += 1
    stats["n_implicit_expression"] = implicit

    abs_reasons: Dict[str, int] = {}
    for row in read_jsonl(os.path.join(processed, "abstention_samples.jsonl")):
        reason = row.obj.get("abstention_reason")
        if isinstance(reason, str):
            abs_reasons[reason] = abs_reasons.get(reason, 0) + 1
    stats["abstention_reason_distribution"] = abs_reasons

    edge_layers: Dict[str, int] = {}
    for row in read_jsonl(os.path.join(processed, "graph_edges.jsonl")):
        layer = row.obj.get("edge_layer")
        if isinstance(layer, str):
            edge_layers[layer] = edge_layers.get(layer, 0) + 1
    stats["graph_edges_by_layer"] = edge_layers
    return stats


def build_s3r_manifest(output_root: str, cfg: Optional[Dict[str, Any]], validation_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    processed = os.path.join(output_root, "processed")
    return {
        "data_version": cfg.get("data_version", "s3r_data_v1.0-pilot"),
        "created_at": utc_now(),
        "creator": cfg.get("creator", "stage4_pilot_export"),
        "stage_versions": cfg.get("stage_versions", {"stage4_export": "v1.0.0-pilot"}),
        "ontology_version": cfg.get("ontology_version", {"hpo_release": "pilot"}),
        "model_versions": cfg.get("model_versions", {}),
        "file_checksums": file_checksums(processed),
        "stats": collect_stats(processed),
        "validation": validation_summary or {
            "referential_integrity_passed": None,
            "offset_validation_passed": None,
            "data_quality_passed": None,
            "report_path": "validation/data_quality_report.json",
        },
    }


def check_quality(data_dir: str) -> Dict[str, Any]:
    processed = processed_dir(data_dir)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for filename in REQUIRED_FILES.values():
        path = os.path.join(processed, filename)
        if not os.path.exists(path):
            errors.append({"check": "required_file_exists", "file": filename, "error": "missing"})
        elif count_jsonl(path) == 0:
            errors.append({"check": "required_file_nonempty", "file": filename, "error": "empty"})

    seen_neg_types: Set[str] = set()
    for row in read_jsonl(os.path.join(processed, "retrieval_pairs.jsonl")):
        for neg in row.obj.get("hard_negatives") or []:
            typ = neg.get("negative_type")
            if isinstance(typ, str):
                seen_neg_types.add(typ)
    missing_neg_types = sorted(RETRIEVAL_NEGATIVE_TYPES - seen_neg_types)
    if missing_neg_types:
        errors.append({"check": "retrieval_negative_type_coverage", "missing": missing_neg_types})

    seen_abs_reasons: Set[str] = set()
    for row in read_jsonl(os.path.join(processed, "abstention_samples.jsonl")):
        reason = row.obj.get("abstention_reason")
        if isinstance(reason, str):
            seen_abs_reasons.add(reason)
    missing_abs_reasons = sorted(ABSTENTION_REASONS - seen_abs_reasons)
    if missing_abs_reasons:
        errors.append({"check": "abstention_reason_coverage", "missing": missing_abs_reasons})

    node_ids = {row.obj.get("node_id") for row in read_jsonl(os.path.join(processed, "nodes.jsonl"))}
    for row in read_jsonl(os.path.join(processed, "nodes.jsonl")):
        obj = row.obj
        if obj.get("tier") == "parent_inherited":
            inherited = obj.get("inherited_from")
            if not inherited or inherited not in node_ids:
                errors.append({"check": "parent_inherited_has_valid_source", "node_id": obj.get("node_id"), "inherited_from": inherited})
        if obj.get("evidence_count") == 0 and not obj.get("uses_parent_evidence"):
            warnings.append({"check": "zero_evidence_without_parent_inheritance", "node_id": obj.get("node_id")})

    checksums = file_checksums(processed)
    manifest_path = os.path.join(manifest_dir(data_dir), "s3r_data_manifest.json")
    checksum_match = None
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        checksum_match = manifest.get("file_checksums") == checksums
        if not checksum_match:
            errors.append({"check": "manifest_checksums_match", "error": "manifest checksums differ from current files"})

    stats = collect_stats(processed)
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "required_files_nonempty": not any(e.get("check") in {"required_file_exists", "required_file_nonempty"} for e in errors),
            "retrieval_negative_type_coverage": len(missing_neg_types) == 0,
            "abstention_reason_coverage": len(missing_abs_reasons) == 0,
            "manifest_checksums_match": checksum_match,
        },
        "stats": stats,
    }


def run_validate_all(root_or_processed: str) -> Dict[str, Any]:
    root = as_root_dir(root_or_processed)
    vdir = validation_dir(root)
    ensure_dir(vdir)
    ref_report = validate_references(root)
    offset_report = validate_offsets(root)
    write_json(os.path.join(vdir, "referential_integrity_report.json"), ref_report)
    write_json(os.path.join(vdir, "offset_validation_report.json"), offset_report)
    quality = {
        "passed": ref_report["passed"] and offset_report["passed"],
        "checks": {
            "referential_integrity_passed": ref_report["passed"],
            "offset_validation_passed": offset_report["passed"],
        },
        "stats": {
            **ref_report["stats"],
            **offset_report["stats"],
        },
    }
    write_json(os.path.join(vdir, "data_quality_report.json"), quality)
    return quality


def cmd_export_all(args: argparse.Namespace) -> int:
    result = export_all(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_validate_references(args: argparse.Namespace) -> int:
    report = validate_references(args.data_dir)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def cmd_validate_offsets(args: argparse.Namespace) -> int:
    report = validate_offsets(args.data_dir)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def cmd_validate_all(args: argparse.Namespace) -> int:
    quality = run_validate_all(args.data_dir if not hasattr(args, "validation_dir") else args.data_dir)
    if hasattr(args, "validation_dir") and args.validation_dir != validation_dir(args.data_dir):
        ensure_dir(args.validation_dir)
        for name in ["referential_integrity_report.json", "offset_validation_report.json", "data_quality_report.json"]:
            src = os.path.join(validation_dir(args.data_dir), name)
            dst = os.path.join(args.validation_dir, name)
            if os.path.abspath(src) != os.path.abspath(dst):
                write_json(dst, read_json(src))
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0 if quality["passed"] else 1


def cmd_check_quality(args: argparse.Namespace) -> int:
    report = check_quality(args.data_dir)
    out = args.report or os.path.join(validation_dir(args.data_dir), "quality_gate_report.json")
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def cmd_write_manifest(args: argparse.Namespace) -> int:
    root = as_root_dir(args.data_dir)
    cfg = read_json(args.config) if args.config else {}
    ref = read_json(os.path.join(validation_dir(root), "referential_integrity_report.json")) if os.path.exists(os.path.join(validation_dir(root), "referential_integrity_report.json")) else {}
    off = read_json(os.path.join(validation_dir(root), "offset_validation_report.json")) if os.path.exists(os.path.join(validation_dir(root), "offset_validation_report.json")) else {}
    q = read_json(os.path.join(validation_dir(root), "quality_gate_report.json")) if os.path.exists(os.path.join(validation_dir(root), "quality_gate_report.json")) else {}
    validation_summary = {
        "referential_integrity_passed": ref.get("passed"),
        "offset_validation_passed": off.get("passed"),
        "data_quality_passed": q.get("passed"),
        "report_path": "validation/quality_gate_report.json",
    }
    manifest = build_s3r_manifest(root, cfg, validation_summary)
    write_json(os.path.join(manifest_dir(root), "s3r_data_manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    root = as_root_dir(args.data_dir)
    cfg_path = args.config or os.path.join(manifest_dir(root), "pilot_export_config_snapshot.json")
    cfg = read_json(cfg_path) if os.path.exists(cfg_path) else {}

    validation_summary = run_validate_all(root)
    quality_report = check_quality(root)
    write_json(os.path.join(validation_dir(root), "quality_gate_report.json"), quality_report)

    manifest_validation = {
        "referential_integrity_passed": validation_summary["checks"]["referential_integrity_passed"],
        "offset_validation_passed": validation_summary["checks"]["offset_validation_passed"],
        "data_quality_passed": quality_report["passed"],
        "report_path": "validation/quality_gate_report.json",
    }
    manifest = build_s3r_manifest(root, cfg, manifest_validation)
    write_json(os.path.join(manifest_dir(root), "s3r_data_manifest.json"), manifest)

    passed = validation_summary["passed"] and quality_report["passed"]
    release_report = {
        "passed": passed,
        "created_at": utc_now(),
        "data_dir": os.path.abspath(root),
        "validation": manifest_validation,
    }
    write_json(os.path.join(validation_dir(root), "release_report.json"), release_report)
    if passed:
        with open(os.path.join(root, "RELEASED"), "w", encoding="utf-8") as f:
            f.write(json.dumps(release_report, ensure_ascii=False, indent=2))
            f.write("\n")
    print(json.dumps(release_report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def cmd_placeholder(args: argparse.Namespace) -> int:
    raise SystemExit(f"{args.command} is registered but not implemented in this Stage 4 pilot.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("run_stage4_export")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export-all")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_export_all)

    p = sub.add_parser("validate-references")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(func=cmd_validate_references)

    p = sub.add_parser("validate-offsets")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(func=cmd_validate_offsets)

    p = sub.add_parser("validate-all")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--validation-dir", required=False, default="")
    p.set_defaults(func=cmd_validate_all)

    p = sub.add_parser("check-quality")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(func=cmd_check_quality)

    p = sub.add_parser("write-manifest")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", default="")
    p.set_defaults(func=cmd_write_manifest)

    p = sub.add_parser("release")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", default="")
    p.set_defaults(func=cmd_release)

    for name in [
        "export-nodes",
        "export-evidence",
        "export-span-supervision",
        "export-retrieval-pairs",
        "export-rerank-pairs",
        "export-abstention-samples",
        "export-graph-edges",
        "build-indexes",
    ]:
        p = sub.add_parser(name)
        p.set_defaults(func=cmd_placeholder)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
