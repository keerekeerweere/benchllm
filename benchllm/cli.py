from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchllm.catalog import build_run_matrix, load_catalog
from benchllm.prepare import prepare_runtime_bundle
from benchllm.reporting import load_results, summarize_results
from benchllm.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchllm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print the benchmark run matrix.")
    plan_parser.add_argument("--catalog", required=True)

    run_parser = subparsers.add_parser("run", help="Execute the benchmark matrix.")
    run_parser.add_argument("--catalog", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--profile")
    run_parser.add_argument("--workload")

    prepare_parser = subparsers.add_parser("prepare", help="Write runtime launch scripts and directories.")
    prepare_parser.add_argument("--catalog", required=True)
    prepare_parser.add_argument("--output-dir", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize benchmark results.")
    summarize_parser.add_argument("--results", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        return _plan(args.catalog)
    if args.command == "run":
        return _run(args.catalog, args.output, args.profile, args.workload)
    if args.command == "prepare":
        return _prepare(args.catalog, args.output_dir)
    if args.command == "summarize":
        return _summarize(args.results)
    raise AssertionError(f"Unhandled command: {args.command}")


def run_cli() -> int:
    return main()


def _plan(catalog_path: str) -> int:
    catalog = load_catalog(catalog_path)
    for spec in build_run_matrix(catalog):
        profile = catalog.profiles[spec.profile_id]
        print(
            f"{spec.run_id}\tbackend={profile.backend}\tmodel={profile.model}\t"
            f"concurrency={spec.concurrency}"
        )
    return 0


def _run(catalog_path: str, output_path: str, profile_filter: str | None, workload_filter: str | None) -> int:
    catalog = load_catalog(catalog_path)
    runner = BenchmarkRunner()
    results = []
    for spec in build_run_matrix(catalog):
        if profile_filter and spec.profile_id != profile_filter:
            continue
        if workload_filter and spec.workload_id != workload_filter:
            continue
        profile = catalog.profiles[spec.profile_id]
        workload = catalog.workloads[spec.workload_id]
        results.extend(result.to_dict() for result in runner.run_group(spec, profile, workload))
    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {output_path}")
    return 0


def _prepare(catalog_path: str, output_dir: str) -> int:
    root = prepare_runtime_bundle(catalog_path, output_dir)
    print(f"Wrote runtime bundle to {root}")
    return 0


def _summarize(results_path: str) -> int:
    rows = summarize_results(load_results(results_path))
    for row in rows:
        print(
            f"{row.profile_id}\t{row.workload_id}\tsamples={row.samples}\t"
            f"median_ttft_ms={row.median_ttft_ms}\t"
            f"median_decode_tps={row.median_decode_tokens_per_second}\t"
            f"validation_pass_rate={row.validation_pass_rate}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
