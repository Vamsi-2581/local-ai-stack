import unittest

from core.settings import build_performance_settings, normalize_setting


class PerformanceSettingsTests(unittest.TestCase):
    def test_normalize_setting_manual_mode_uses_explicit_defaults(self):
        normalized = normalize_setting(
            name="context_length",
            requested=8192,
            effective=8192,
        )

        self.assertEqual(
            {
                "name": "context_length",
                "mode": "manual",
                "requested": 8192,
                "recommended": None,
                "effective": 8192,
                "reason": "Explicit value supplied by configuration.",
                "confidence": "explicit",
            },
            normalized,
        )

    def test_normalize_setting_auto_unresolved_sets_requested_none_and_unresolved_status(self):
        normalized = normalize_setting(
            name="concurrency",
            requested="auto",
            effective="auto",
        )

        self.assertEqual(
            {
                "name": "concurrency",
                "mode": "auto",
                "requested": None,
                "recommended": None,
                "effective": "auto",
                "reason": "Automatic resolution is not implemented for this setting yet.",
                "confidence": "unresolved",
            },
            normalized,
        )

    def test_normalize_setting_auto_resolved_uses_recommendation_and_provisional_fallback(self):
        with_recommendation = normalize_setting(
            name="context_length",
            requested="auto",
            effective=16384,
            recommendation={
                "value": 16384,
                "reason": "Selected from available memory.",
                "confidence": "high",
            },
        )
        self.assertEqual(
            {
                "name": "context_length",
                "mode": "auto",
                "requested": None,
                "recommended": 16384,
                "effective": 16384,
                "reason": "Selected from available memory.",
                "confidence": "high",
            },
            with_recommendation,
        )

        fallback_recommendation = normalize_setting(
            name="gpu_memory_utilization",
            requested="auto",
            effective=0.9,
        )
        self.assertEqual(
            {
                "name": "gpu_memory_utilization",
                "mode": "auto",
                "requested": None,
                "recommended": 0.9,
                "effective": 0.9,
                "reason": "Automatically resolved by the planner.",
                "confidence": "provisional",
            },
            fallback_recommendation,
        )

    def test_build_performance_settings_returns_all_three_knobs(self):
        settings = build_performance_settings(
            requested={
                "context_length": "auto",
                "gpu_memory_utilization": 0.85,
                "concurrency": "auto",
            },
            effective={
                "context_length": 4096,
                "gpu_memory_utilization": 0.85,
                "concurrency": "auto",
            },
            recommendations={
                "context_length": {
                    "value": 4096,
                    "reason": "Fitted to model limits.",
                    "confidence": "provisional",
                },
            },
        )

        self.assertEqual(
            {
                "context_length",
                "gpu_memory_utilization",
                "concurrency",
            },
            set(settings.keys()),
        )
        self.assertEqual("auto", settings["context_length"]["mode"])
        self.assertEqual(4096, settings["context_length"]["recommended"])
        self.assertEqual("manual", settings["gpu_memory_utilization"]["mode"])
        self.assertEqual(0.85, settings["gpu_memory_utilization"]["requested"])
        self.assertEqual("unresolved", settings["concurrency"]["confidence"])


if __name__ == "__main__":
    unittest.main()
