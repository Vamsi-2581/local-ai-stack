from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from core.retention import apply_usage_retention, build_usage_retention_policy


class _FakeUsageStore:
    def __init__(self, result=None):
        self.calls = []
        self._result = result or {
            "removed_events": 4,
            "removed_partitions": 1,
        }

    def prune_before(self, cutoff_iso, config):
        self.calls.append((cutoff_iso, config))
        return dict(self._result)


class UsageRetentionPolicyTests(unittest.TestCase):
    def test_build_usage_retention_policy_defaults_and_positive_days_enable_policy(self):
        self.assertEqual(
            {
                "enabled": True,
                "retention_days": 7,
            },
            build_usage_retention_policy({"analytics": {}}),
        )
        self.assertEqual(
            {
                "enabled": True,
                "retention_days": 14,
            },
            build_usage_retention_policy(
                {"analytics": {"local_retention_days": 14}}
            ),
        )

    def test_build_usage_retention_policy_disables_for_zero_or_none(self):
        for value in (0, None):
            with self.subTest(local_retention_days=value):
                self.assertEqual(
                    {
                        "enabled": False,
                        "retention_days": value,
                    },
                    build_usage_retention_policy(
                        {"analytics": {"local_retention_days": value}}
                    ),
                )

    def test_apply_usage_retention_skips_prune_when_disabled(self):
        store = _FakeUsageStore()
        config = {"analytics": {"local_retention_days": 0}}

        result = apply_usage_retention(
            config=config,
            usage_store=store,
        )

        self.assertEqual([], store.calls)
        self.assertEqual(
            {
                "policy": {
                    "enabled": False,
                    "retention_days": 0,
                },
                "removed_events": 0,
                "removed_partitions": 0,
            },
            result,
        )

    def test_apply_usage_retention_calls_prune_before_with_utc_cutoff_and_merges_counts(self):
        store = _FakeUsageStore(
            result={
                "removed_events": 12,
                "removed_partitions": 2,
            }
        )
        config = {"analytics": {"local_retention_days": 5}}
        fixed_now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

        with patch("core.retention.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            result = apply_usage_retention(
                config=config,
                usage_store=store,
            )

        mock_datetime.now.assert_called_once_with(timezone.utc)
        self.assertEqual(
            [("2026-10-29T12:00:00+00:00", config)],
            store.calls,
        )
        self.assertEqual(
            {
                "policy": {
                    "enabled": True,
                    "retention_days": 5,
                },
                "removed_events": 12,
                "removed_partitions": 2,
            },
            result,
        )


if __name__ == "__main__":
    unittest.main()
