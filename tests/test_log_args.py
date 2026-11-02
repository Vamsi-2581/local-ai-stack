from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core.log_args import LogArgumentsError, parse_log_arguments


class LogArgumentParsingTests(unittest.TestCase):
    def test_default_line_count_and_optional_service(self):
        self.assertEqual(
            (None, 40),
            parse_log_arguments([]),
        )
        self.assertEqual(
            ("gateway", 40),
            parse_log_arguments(["gateway"]),
        )

    def test_short_and_long_options_before_or_after_service(self):
        self.assertEqual(
            (None, 5),
            parse_log_arguments(["-n", "5"]),
        )
        self.assertEqual(
            ("gateway", 5),
            parse_log_arguments(["--lines", "5", "gateway"]),
        )
        self.assertEqual(
            ("gateway", 5),
            parse_log_arguments(["gateway", "-n", "5"]),
        )
        self.assertEqual(
            (None, 7),
            parse_log_arguments(["--lines=7"]),
        )

    def test_rejects_non_positive_or_non_integer_counts(self):
        for arguments in (
            ["-n", "0"],
            ["--lines", "-2"],
            ["--lines", "1.5"],
            ["--lines", "five"],
            ["--lines="],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(
                    LogArgumentsError,
                    "positive integer",
                ):
                    parse_log_arguments(arguments)

    def test_rejects_missing_duplicate_and_extra_arguments(self):
        for arguments in (
            ["--lines"],
            ["-n", "5", "--lines", "6"],
            ["gateway", "frontend"],
            ["--unknown"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(LogArgumentsError):
                    parse_log_arguments(arguments)


class LogsCommandTests(unittest.TestCase):
    def run_cli(self, arguments):
        calls = []
        services = [
            {"enabled": True, "id": "gateway", "provider": "builtin"},
            {"enabled": True, "id": "frontend", "provider": "web-ui"},
            {"enabled": False, "id": "disabled", "provider": "unused"},
        ]

        def stub(*args, **kwargs):
            return None

        def tail_log(service_id, lines=30):
            calls.append((service_id, lines))
            return f"{service_id} log output"

        def build_manifest(plan, config):
            return {"services": services}

        def make_module(name, **exports):
            module = types.ModuleType(name)
            for key, value in exports.items():
                setattr(module, key, value)
            return module

        modules = {
            "core.hardware": make_module("core.hardware", inspect_hardware=stub),
            "core.models": make_module("core.models", list_local_models=stub),
            "core.config": make_module("core.config", load_config=lambda: {}),
            "core.planner": make_module("core.planner", build_initial_plan=lambda: {}),
            "core.status": make_module("core.status", get_runtime_status=stub),
            "core.preflight": make_module("core.preflight", run_preflight=stub),
            "core.runtime_manager": make_module(
                "core.runtime_manager",
                ensure_runtime=stub,
                required_runtimes=stub,
            ),
            "core.deployment": make_module(
                "core.deployment",
                build_deployment_manifest=build_manifest,
            ),
            "adapters.inference": make_module(
                "adapters.inference",
                get_inference_adapter=stub,
            ),
            "adapters.executor": make_module(
                "adapters.executor",
                get_executor=stub,
            ),
            "adapters.gateway": make_module(
                "adapters.gateway",
                resolve_gateway_endpoint=stub,
            ),
            "adapters.usage_store": make_module(
                "adapters.usage_store",
                get_usage_store_adapter=stub,
            ),
            "core.lifecycle": make_module(
                "core.lifecycle",
                start_service=stub,
                stop_service=stub,
                tail_log=tail_log,
            ),
            "core.health": make_module(
                "core.health",
                check_deployment_health=stub,
            ),
            "core.acceptance": make_module(
                "core.acceptance",
                run_acceptance_test=stub,
            ),
            "core.platform_state": make_module(
                "core.platform_state",
                build_platform_state=stub,
            ),
        }

        stdout = StringIO()
        stderr = StringIO()
        script = Path(__file__).resolve().parents[1] / "ai"
        with (
            patch.dict(sys.modules, modules),
            patch.object(sys, "argv", ["ai", "logs", *arguments]),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as error:
                exit_code = error.code
            else:
                exit_code = 0

        return exit_code, stdout.getvalue(), stderr.getvalue(), calls

    def test_requested_lines_are_passed_to_each_enabled_service(self):
        exit_code, output, error, calls = self.run_cli(["-n", "5"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", error)
        self.assertEqual(
            [("gateway", 5), ("frontend", 5)],
            calls,
        )
        self.assertIn("gateway log output", output)
        self.assertIn("frontend log output", output)

    def test_service_selection_and_default_line_count_are_preserved(self):
        exit_code, _, error, calls = self.run_cli(["gateway"])

        self.assertEqual(0, exit_code)
        self.assertEqual("", error)
        self.assertEqual([("gateway", 40)], calls)

    def test_flag_after_service_is_accepted(self):
        exit_code, _, error, calls = self.run_cli(
            ["frontend", "--lines", "7"]
        )

        self.assertEqual(0, exit_code)
        self.assertEqual("", error)
        self.assertEqual([("frontend", 7)], calls)

    def test_invalid_count_exits_nonzero_with_short_error(self):
        for arguments in (["-n", "0"], ["--lines", "nope"]):
            with self.subTest(arguments=arguments):
                exit_code, _, error, calls = self.run_cli(arguments)

                self.assertEqual(2, exit_code)
                self.assertIn("positive integer", error)
                self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
