from datetime import datetime
import unittest
from uuid import UUID

from core.usage import (
    USAGE_EVENT_SCHEMA_VERSION,
    build_usage_event,
    new_usage_event_id,
    utc_now,
)


def _base_event_kwargs(**overrides):
    kwargs = {
        "event_id": "evt-123",
        "started_at": "2026-11-03T12:00:00+00:00",
        "method": "POST",
        "path": "/v1/chat/completions",
        "requested_model": "assistant-main",
        "actual_model": "Qwen2.5-7B-Instruct-AWQ",
        "logical_model": "assistant-main",
        "provider": "vllm",
        "service_id": "inference",
        "status_code": 200,
        "latency_ms": 42.1256,
    }
    kwargs.update(overrides)
    return kwargs


class UsageEventBuilderTests(unittest.TestCase):
    def test_build_usage_event_populates_nested_sections_and_defaults(self):
        event = build_usage_event(**_base_event_kwargs())

        self.assertEqual(USAGE_EVENT_SCHEMA_VERSION, event["schema_version"])
        self.assertEqual("evt-123", event["event_id"])
        self.assertEqual("2026-11-03T12:00:00+00:00", event["started_at"])
        self.assertEqual(
            {
                "method": "POST",
                "path": "/v1/chat/completions",
                "requested_model": "assistant-main",
                "streaming": False,
            },
            event["request"],
        )
        self.assertEqual(
            {
                "logical_model": "assistant-main",
                "actual_model": "Qwen2.5-7B-Instruct-AWQ",
                "provider": "vllm",
                "service_id": "inference",
                "route_reason": "single_local_upstream",
            },
            event["route"],
        )
        self.assertEqual(
            {
                "status_code": 200,
                "success": True,
                "error": None,
            },
            event["response"],
        )
        self.assertEqual(
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
            event["usage"],
        )
        self.assertEqual({"latency_ms": 42.126}, event["performance"])
        self.assertEqual(
            {
                "user_id": None,
                "team_id": None,
                "application_id": None,
            },
            event["identity"],
        )
        self.assertEqual(
            {
                "currency": None,
                "amount": None,
                "method": None,
            },
            event["cost"],
        )

    def test_build_usage_event_marks_success_only_for_2xx_and_3xx_statuses(self):
        for status_code, expected in (
            (200, True),
            (204, True),
            (400, False),
            (500, False),
            (None, False),
        ):
            with self.subTest(status_code=status_code):
                event = build_usage_event(
                    **_base_event_kwargs(
                        status_code=status_code,
                        error="upstream_error" if not expected else None,
                    )
                )
                self.assertEqual(expected, event["response"]["success"])

    def test_build_usage_event_preserves_explicit_tokens_streaming_and_rounds_latency(self):
        event = build_usage_event(
            **_base_event_kwargs(
                latency_ms=19.987654,
                input_tokens=120,
                output_tokens=45,
                total_tokens=165,
                streaming=True,
            )
        )

        self.assertTrue(event["request"]["streaming"])
        self.assertEqual(
            {
                "input_tokens": 120,
                "output_tokens": 45,
                "total_tokens": 165,
            },
            event["usage"],
        )
        self.assertEqual(19.988, event["performance"]["latency_ms"])

    def test_id_and_timestamp_helpers_return_uuid_and_utc_iso_timestamp(self):
        generated_id = new_usage_event_id()
        self.assertEqual(generated_id, str(UUID(generated_id)))

        parsed_now = datetime.fromisoformat(utc_now())
        self.assertIsNotNone(parsed_now.tzinfo)


if __name__ == "__main__":
    unittest.main()
