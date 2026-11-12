import unittest
from unittest.mock import patch

from core.health import check_deployment_health, check_service_health

LOCALHOST = "http" + "://127.0.0.1"


class HealthProbeTests(unittest.TestCase):
    def test_vllm_healthy_when_models_returned(self):
        service = {
            "id": "inference",
            "provider": "vllm",
            "url": f"{LOCALHOST}:8000",
        }
        response = {
            "ok": True,
            "status_code": 200,
            "body": b'{"data": [{"id": "local-llm"}]}',
            "error": None,
        }

        with patch("core.health.http_get", return_value=response) as mock_get:
            result = check_service_health(service)

        mock_get.assert_called_once_with(f"{LOCALHOST}:8000/v1/models")
        self.assertTrue(result["healthy"])
        self.assertEqual("inference", result["service"])
        self.assertEqual("vllm", result["provider"])
        self.assertIn("local-llm", result["detail"])

    def test_vllm_unhealthy_on_http_failure_invalid_json_or_empty_models(self):
        service = {
            "id": "inference",
            "provider": "vllm",
            "url": f"{LOCALHOST}:8000",
        }

        failure_cases = [
            (
                {
                    "ok": False,
                    "status_code": 503,
                    "body": b"",
                    "error": "Service Unavailable",
                },
                "HTTP 503",
            ),
            (
                {
                    "ok": True,
                    "status_code": 200,
                    "body": b"not-json",
                    "error": None,
                },
                "Invalid JSON response",
            ),
            (
                {
                    "ok": True,
                    "status_code": 200,
                    "body": b'{"data": []}',
                    "error": None,
                },
                "returned no models",
            ),
        ]

        for response, expected_detail in failure_cases:
            with self.subTest(expected_detail=expected_detail):
                with patch("core.health.http_get", return_value=response):
                    result = check_service_health(service)
                self.assertFalse(result["healthy"])
                self.assertIn(expected_detail, result["detail"])

    def test_builtin_proxy_and_builtin_http_probe_live_endpoint(self):
        for provider, expected_fallback in (
            ("builtin_proxy", "Gateway liveness failed"),
            ("builtin_http", "Platform API liveness failed"),
        ):
            service = {
                "id": provider,
                "provider": provider,
                "url": f"{LOCALHOST}:4000",
            }
            with self.subTest(provider=provider, healthy=True):
                ok_resp = {"ok": True, "status_code": 200, "body": b"ok", "error": None}
                with patch("core.health.http_get", return_value=ok_resp) as mock_get:
                    result = check_service_health(service)
                mock_get.assert_called_once_with(f"{LOCALHOST}:4000/live")
                self.assertTrue(result["healthy"])
                self.assertEqual("HTTP 200", result["detail"])

            with self.subTest(provider=provider, healthy=False):
                err_resp = {"ok": False, "status_code": 500, "body": b"", "error": None}
                with patch("core.health.http_get", return_value=err_resp):
                    result = check_service_health(service)
                self.assertFalse(result["healthy"])
                self.assertEqual(expected_fallback, result["detail"])

    def test_openwebui_probes_base_url_for_healthy_and_unhealthy(self):
        service = {
            "id": "frontend",
            "provider": "openwebui",
            "url": f"{LOCALHOST}:3000",
        }

        ok_resp = {"ok": True, "status_code": 200, "body": b"<html></html>", "error": None}
        with patch("core.health.http_get", return_value=ok_resp) as mock_get:
            result = check_service_health(service)
        mock_get.assert_called_once_with(f"{LOCALHOST}:3000")
        self.assertTrue(result["healthy"])
        self.assertEqual("HTTP 200", result["detail"])

        err_resp = {"ok": False, "status_code": None, "body": b"", "error": "connection refused"}
        with patch("core.health.http_get", return_value=err_resp):
            result = check_service_health(service)
        self.assertFalse(result["healthy"])
        self.assertEqual("connection refused", result["detail"])

    def test_unknown_provider_returns_unhealthy_with_expected_detail(self):
        service = {
            "id": "custom",
            "provider": "custom_engine",
            "url": f"{LOCALHOST}:9000",
        }

        with patch("core.health.http_get") as mock_get:
            result = check_service_health(service)

        mock_get.assert_not_called()
        self.assertFalse(result["healthy"])
        self.assertIn("No health probe is defined", result["detail"])

    def test_check_deployment_health_skips_disabled_services(self):
        manifest = {
            "services": [
                {
                    "id": "gateway",
                    "provider": "builtin_proxy",
                    "url": f"{LOCALHOST}:4000",
                    "enabled": True,
                },
                {
                    "id": "frontend",
                    "provider": "openwebui",
                    "url": f"{LOCALHOST}:3000",
                    "enabled": False,
                },
            ]
        }

        ok_resp = {"ok": True, "status_code": 200, "body": b"ok", "error": None}
        with patch("core.health.http_get", return_value=ok_resp) as mock_get:
            results = check_deployment_health(manifest)

        mock_get.assert_called_once_with(f"{LOCALHOST}:4000/live")
        self.assertEqual(1, len(results))
        self.assertEqual("gateway", results[0]["service"])


if __name__ == "__main__":
    unittest.main()
