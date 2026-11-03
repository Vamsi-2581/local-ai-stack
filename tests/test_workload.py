import unittest

from core.workload import get_workload_profile, resolve_workload


EXPECTED_PROFILES = {
    "interactive": {
        "description": "Balanced interactive chat for multiple users.",
        "priority": "balanced",
        "latency_priority": "high",
        "throughput_priority": "medium",
        "concurrency_priority": "medium",
    },
    "personal": {
        "description": "Optimized primarily for one interactive user.",
        "priority": "latency",
        "latency_priority": "high",
        "throughput_priority": "low",
        "concurrency_priority": "low",
    },
    "throughput": {
        "description": "Optimized for serving many requests efficiently.",
        "priority": "throughput",
        "latency_priority": "medium",
        "throughput_priority": "high",
        "concurrency_priority": "high",
    },
}


class WorkloadProfileTests(unittest.TestCase):
    def test_get_workload_profile_returns_each_known_profile(self):
        for name, expected in EXPECTED_PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(expected, get_workload_profile(name))

    def test_get_workload_profile_returns_none_for_unknown_profile(self):
        self.assertIsNone(get_workload_profile("unknown"))

    def test_resolve_workload_returns_profile_for_valid_configuration(self):
        resolved = resolve_workload(
            {"performance": {"profile": "interactive"}}
        )

        self.assertEqual(
            {
                "name": "interactive",
                "valid": True,
                **EXPECTED_PROFILES["interactive"],
            },
            resolved,
        )

    def test_resolve_workload_explains_unknown_profile(self):
        resolved = resolve_workload(
            {"performance": {"profile": "experimental"}}
        )

        self.assertEqual(
            {
                "name": "experimental",
                "valid": False,
                "reason": "Unknown workload profile: experimental",
            },
            resolved,
        )


if __name__ == "__main__":
    unittest.main()
