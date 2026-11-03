import unittest

from core.identity import build_model_identity, build_service_identity


class IdentityBuilderTests(unittest.TestCase):
    def test_build_model_identity_returns_none_when_selected_model_missing_or_none(self):
        config = {
            "inference": {
                "served_model_name": "local-default",
                "provider": "vllm",
            }
        }

        self.assertIsNone(build_model_identity({}, config))
        self.assertIsNone(build_model_identity({"selected_model": None}, config))

    def test_build_model_identity_returns_normalized_identity_with_metadata(self):
        plan = {
            "selected_model": {
                "name": "Qwen2.5-7B-Instruct-AWQ",
                "path": "/models/Qwen2.5-7B-Instruct-AWQ",
                "architecture": "Qwen2ForCausalLM",
                "model_type": "qwen2",
                "quantization_method": "awq",
            }
        }
        config = {
            "inference": {
                "served_model_name": "assistant-main",
                "provider": "vllm",
            }
        }

        self.assertEqual(
            {
                "logical_name": "assistant-main",
                "model_id": "Qwen2.5-7B-Instruct-AWQ",
                "source": {
                    "type": "local",
                    "path": "/models/Qwen2.5-7B-Instruct-AWQ",
                },
                "provider": "vllm",
                "metadata": {
                    "architecture": "Qwen2ForCausalLM",
                    "model_type": "qwen2",
                    "quantization": "awq",
                },
            },
            build_model_identity(plan, config),
        )

    def test_build_model_identity_defaults_optional_metadata_to_none(self):
        plan = {
            "selected_model": {
                "name": "minimal-model",
                "path": "/models/minimal-model",
            }
        }
        config = {
            "inference": {
                "served_model_name": "minimal",
                "provider": "vllm",
            }
        }

        self.assertEqual(
            {
                "logical_name": "minimal",
                "model_id": "minimal-model",
                "source": {
                    "type": "local",
                    "path": "/models/minimal-model",
                },
                "provider": "vllm",
                "metadata": {
                    "architecture": None,
                    "model_type": None,
                    "quantization": None,
                },
            },
            build_model_identity(plan, config),
        )

    def test_build_service_identity_returns_expected_keys(self):
        service = {
            "id": "inference",
            "role": "llm_engine",
            "provider": "vllm",
            "enabled": True,
            "port": 8000,
        }

        self.assertEqual(
            {
                "service_id": "inference",
                "role": "llm_engine",
                "provider": "vllm",
                "enabled": True,
            },
            build_service_identity(service),
        )


if __name__ == "__main__":
    unittest.main()
