from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Callable

import httpx

from benchllm.catalog import BenchmarkRunSpec, Profile, Workload


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    profile_id: str
    workload_id: str
    worker_index: int
    status_code: int
    ttft_ms: float
    total_duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    decode_tokens_per_second: float
    validation_passed: bool
    validation_error: str | None
    response_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    def __init__(
        self,
        client: httpx.Client | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self._clock = clock or time.perf_counter

    def run_case(
        self,
        spec: BenchmarkRunSpec,
        profile: Profile,
        workload: Workload,
        *,
        worker_index: int = 0,
    ) -> BenchmarkResult:
        request_payload = dict(workload.request)
        request_payload["model"] = profile.model
        request_payload["stream"] = True
        request_payload["stream_options"] = {"include_usage": True}
        url = f"{profile.api_base.rstrip('/')}/chat/completions"
        started = self._clock()
        first_token_at: float | None = None
        response_text_parts: list[str] = []
        usage: dict[str, int] = {}

        with self._client.stream("POST", url, json=request_payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                usage.update(event.get("usage") or {})
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        response_text_parts.append(content)
                        if first_token_at is None:
                            first_token_at = self._clock()

        finished = self._clock()
        response_text = "".join(response_text_parts)
        ttft_ms = round(((first_token_at or finished) - started) * 1000, 3)
        total_duration_ms = round((finished - started) * 1000, 3)
        completion_tokens = int(usage.get("completion_tokens", 0))
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        decode_seconds = max((finished - (first_token_at or started)), 1e-9)
        validation_passed, validation_error = _validate_response(workload, response_text)
        return BenchmarkResult(
            run_id=spec.run_id,
            profile_id=spec.profile_id,
            workload_id=spec.workload_id,
            worker_index=worker_index,
            status_code=response.status_code,
            ttft_ms=ttft_ms,
            total_duration_ms=total_duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            decode_tokens_per_second=round(completion_tokens / decode_seconds, 3),
            validation_passed=validation_passed,
            validation_error=validation_error,
            response_text=response_text,
        )

    def run_group(self, spec: BenchmarkRunSpec, profile: Profile, workload: Workload) -> list[BenchmarkResult]:
        with ThreadPoolExecutor(max_workers=spec.concurrency) as executor:
            futures = [
                executor.submit(self.run_case, spec, profile, workload, worker_index=index)
                for index in range(spec.concurrency)
            ]
        return [future.result() for future in futures]


def _validate_response(workload: Workload, response_text: str) -> tuple[bool, str | None]:
    if workload.validations.expect_json:
        try:
            json.loads(response_text)
        except json.JSONDecodeError as exc:
            return False, str(exc)
    return True, None
