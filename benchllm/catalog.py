from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Defaults:
    api_base: str
    context_size: int = 8192


@dataclass(frozen=True)
class LaunchSpec:
    command: list[str]
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    id: str
    backend: str
    kind: str
    model: str
    api_base: str
    launch: LaunchSpec | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationRules:
    expect_json: bool = False


@dataclass(frozen=True)
class Workload:
    id: str
    request: dict[str, Any]
    validations: ValidationRules = field(default_factory=ValidationRules)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatrixSpec:
    profiles: list[str]
    workloads: list[str]
    concurrencies: list[int]
    repetitions: int


@dataclass(frozen=True)
class Catalog:
    defaults: Defaults
    profiles: dict[str, Profile]
    workloads: dict[str, Workload]
    matrix: MatrixSpec


@dataclass(frozen=True)
class BenchmarkRunSpec:
    run_id: str
    profile_id: str
    workload_id: str
    concurrency: int
    repetition: int


def load_catalog(path: str | Path) -> Catalog:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults_raw = raw.get("defaults") or {}
    defaults = Defaults(
        api_base=_require_text(defaults_raw, "api_base"),
        context_size=int(defaults_raw.get("context_size", 8192)),
    )
    profiles = {
        profile.id: profile
        for profile in (_parse_profile(item, defaults) for item in raw.get("profiles", []))
    }
    workloads = {
        workload.id: workload
        for workload in (_parse_workload(item) for item in raw.get("workloads", []))
    }
    matrix_raw = raw.get("matrix") or {}
    matrix = MatrixSpec(
        profiles=list(matrix_raw.get("profiles", [])),
        workloads=list(matrix_raw.get("workloads", [])),
        concurrencies=[int(value) for value in matrix_raw.get("concurrencies", [1])],
        repetitions=int(matrix_raw.get("repetitions", 1)),
    )
    return Catalog(defaults=defaults, profiles=profiles, workloads=workloads, matrix=matrix)


def build_run_matrix(catalog: Catalog) -> list[BenchmarkRunSpec]:
    runs: list[BenchmarkRunSpec] = []
    for profile_id in catalog.matrix.profiles:
        _ensure_key(profile_id, catalog.profiles, "profile")
        for workload_id in catalog.matrix.workloads:
            _ensure_key(workload_id, catalog.workloads, "workload")
            for concurrency in catalog.matrix.concurrencies:
                for repetition in range(1, catalog.matrix.repetitions + 1):
                    run_id = f"{profile_id}__{workload_id}__c{concurrency}__r{repetition}"
                    runs.append(
                        BenchmarkRunSpec(
                            run_id=run_id,
                            profile_id=profile_id,
                            workload_id=workload_id,
                            concurrency=concurrency,
                            repetition=repetition,
                        )
                    )
    return runs


def _parse_profile(raw: dict[str, Any], defaults: Defaults) -> Profile:
    launch_raw = raw.get("launch")
    launch = None
    if launch_raw:
        launch = LaunchSpec(
            command=[str(item) for item in launch_raw.get("command", [])],
            args=[str(item) for item in launch_raw.get("args", [])],
            env={str(key): str(value) for key, value in (launch_raw.get("env") or {}).items()},
        )
    profile_fields = {"id", "backend", "kind", "model", "api_base", "launch"}
    return Profile(
        id=_require_text(raw, "id"),
        backend=_require_text(raw, "backend"),
        kind=_require_text(raw, "kind"),
        model=_require_text(raw, "model"),
        api_base=str(raw.get("api_base", defaults.api_base)),
        launch=launch,
        metadata={key: value for key, value in raw.items() if key not in profile_fields},
    )


def _parse_workload(raw: dict[str, Any]) -> Workload:
    validations_raw = raw.get("validations") or {}
    workload_fields = {"id", "request", "validations"}
    return Workload(
        id=_require_text(raw, "id"),
        request=dict(raw.get("request") or {}),
        validations=ValidationRules(expect_json=bool(validations_raw.get("expect_json", False))),
        metadata={key: value for key, value in raw.items() if key not in workload_fields},
    )


def _require_text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string field: {key}")
    return value


def _ensure_key(key: str, mapping: dict[str, Any], label: str) -> None:
    if key not in mapping:
        raise KeyError(f"Unknown {label}: {key}")
