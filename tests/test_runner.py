import json
import unittest

import httpx

from benchllm.catalog import BenchmarkRunSpec, Profile, ValidationRules, Workload
from benchllm.runner import BenchmarkRunner


class IteratorStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


class RunnerTest(unittest.TestCase):
    def test_executes_streaming_chat_completion_and_captures_metrics(self) -> None:
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"{\\"status\\":\\"ok\\"}"}}]}\n\n',
            b'data: {"usage":{"prompt_tokens":42,"completion_tokens":5,"total_tokens":47}}\n\n',
            b"data: [DONE]\n\n",
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, httpx.URL("http://localhost:8000/v1/chat/completions"))
            payload = json.loads(request.content.decode("utf-8"))
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["model"], "Qwen/Qwen3-Coder-Next-FP8")
            self.assertEqual(payload["messages"][0]["role"], "user")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=IteratorStream(chunks),
            )

        clock_values = iter([0.0, 0.05, 0.20])
        runner = BenchmarkRunner(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            clock=lambda: next(clock_values),
        )
        spec = BenchmarkRunSpec(
            run_id="vllm-qwen__json-small__c1__r1",
            profile_id="vllm-qwen",
            workload_id="json-small",
            concurrency=1,
            repetition=1,
        )
        profile = Profile(
            id="vllm-qwen",
            backend="vllm",
            kind="inference",
            model="Qwen/Qwen3-Coder-Next-FP8",
            api_base="http://localhost:8000/v1",
            launch=None,
            metadata={},
        )
        workload = Workload(
            id="json-small",
            request={"messages": [{"role": "user", "content": "Return JSON"}], "max_tokens": 64},
            validations=ValidationRules(expect_json=True),
            metadata={},
        )

        result = runner.run_case(spec, profile, workload)

        self.assertEqual(result.run_id, spec.run_id)
        self.assertEqual(result.prompt_tokens, 42)
        self.assertEqual(result.completion_tokens, 5)
        self.assertEqual(result.total_tokens, 47)
        self.assertEqual(result.ttft_ms, 50.0)
        self.assertEqual(result.total_duration_ms, 200.0)
        self.assertTrue(result.validation_passed)
        self.assertGreater(result.decode_tokens_per_second, 0.0)


if __name__ == "__main__":
    unittest.main()
