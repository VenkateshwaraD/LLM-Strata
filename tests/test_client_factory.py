"""Tests for core/client_factory.py: provider dispatch (SDK is mocked at the boundary)."""

import os
import sys
import types
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from core import client_factory


def _fake_openai_module() -> types.ModuleType:
    """Build a stub `openai` module recording constructor kwargs."""
    module = types.ModuleType("openai")
    calls: list[tuple] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    module.OpenAI = _FakeClient
    module.AzureOpenAI = _FakeClient
    module.recorded = calls
    return module


class GetClientTests(unittest.TestCase):
    def setUp(self):
        self.fake_openai = _fake_openai_module()
        patcher = patch.dict(sys.modules, {"openai": self.fake_openai})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _profile(self, provider: str, **extra) -> dict:
        return {"provider": provider, **extra}

    def test_openai_provider(self):
        with patch.object(client_factory, "load_profile",
                          return_value=self._profile("openai")):
            client = client_factory.get_client()
        self.assertIsInstance(client, self.fake_openai.OpenAI)

    def test_azure_requires_env(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("azure")):
                with self.assertRaises(KeyError):
                    client_factory.get_client()

    def test_azure_with_env(self):
        env = {"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("azure")):
                client = client_factory.get_client()
        self.assertEqual(client.kwargs["azure_endpoint"], "https://e")

    def test_ollama_default_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("ollama")):
                client = client_factory.get_client()
        self.assertEqual(client.kwargs["base_url"], "http://localhost:11434/v1")

    def test_ollama_env_base_url(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://h:1/v1"}):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("ollama")):
                client = client_factory.get_client()
        self.assertEqual(client.kwargs["base_url"], "http://h:1/v1")

    def test_gemini_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("gemini")):
                with self.assertRaises(KeyError):
                    client_factory.get_client()

    def test_huggingface_requires_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("huggingface")):
                with self.assertRaises(KeyError):
                    client_factory.get_client()

    def test_openai_compatible_without_base_url_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile("openai_compatible")):
                with self.assertRaises(ValueError):
                    client_factory.get_client()

    def test_openai_compatible_profile_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile(
                                  "openai_compatible", base_url="http://x/v1")):
                client = client_factory.get_client()
        self.assertEqual(client.kwargs["base_url"], "http://x/v1")

    def test_openai_compatible_custom_key_env(self):
        env = {"MY_PRIVATE_KEY": "secret"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(client_factory, "load_profile",
                              return_value=self._profile(
                                  "openai_compatible", base_url="http://x/v1",
                                  api_key_env="MY_PRIVATE_KEY")):
                client = client_factory.get_client()
        self.assertEqual(client.kwargs["api_key"], "secret")

    def test_unknown_provider_raises(self):
        with patch.object(client_factory, "load_profile",
                          return_value=self._profile("nope")):
            with self.assertRaises(ValueError) as ctx:
                client_factory.get_client()
        self.assertIn("nope", str(ctx.exception))

    def test_provider_argument_overrides_profile(self):
        with patch.object(client_factory, "load_profile",
                          return_value=self._profile("azure")):
            client = client_factory.get_client("openai")
        self.assertIsInstance(client, self.fake_openai.OpenAI)

    def test_anthropic_uses_litellm_shim(self):
        with patch.object(client_factory, "load_profile",
                          return_value=self._profile("anthropic")):
            client = client_factory.get_client()
        self.assertIsInstance(client, client_factory._LiteLLMClient)
        self.assertEqual(client.chat.completions._prefix, "anthropic/")

    def test_bedrock_uses_litellm_shim(self):
        with patch.object(client_factory, "load_profile",
                          return_value=self._profile("bedrock")):
            client = client_factory.get_client()
        self.assertEqual(client.chat.completions._prefix, "bedrock/")

    def test_litellm_shim_chains_import_error(self):
        completions = client_factory._LiteLLMCompletions("anthropic/")
        if "litellm" in sys.modules:
            self.skipTest("litellm installed; import error path unreachable")
        with self.assertRaises(ImportError) as ctx:
            completions.create(model="m", messages=[])
        self.assertIsNotNone(ctx.exception.__cause__)


class GetProviderTests(unittest.TestCase):
    def test_reads_profile_provider(self):
        with patch.object(client_factory, "load_profile",
                          return_value={"provider": "gemini"}):
            self.assertEqual(client_factory.get_provider(), "gemini")

    def test_defaults_to_openai(self):
        with patch.object(client_factory, "load_profile", return_value={}):
            self.assertEqual(client_factory.get_provider(), "openai")


if __name__ == "__main__":
    unittest.main()
