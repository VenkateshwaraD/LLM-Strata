"""Tests for core/env_loader.py: .env loading, env validation, provider keys."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from core import env_loader


class LoadDotenvTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def _write_env(self, content: str) -> str:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False)
        handle.write(content)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_missing_file_is_noop(self):
        env_loader.load_dotenv("/nonexistent/path/never_exists.env")

    def test_loads_key_values(self):
        env_path = self._write_env("TEST_KEY_A=alpha\nTEST_KEY_B = beta\n")
        env_loader.load_dotenv(env_path)
        self.assertEqual(os.environ["TEST_KEY_A"], "alpha")
        self.assertEqual(os.environ["TEST_KEY_B"], "beta")

    def test_skips_comments_blanks_and_malformed(self):
        env_path = self._write_env(
            "# comment\n\n   \nno_equals_sign\n=empty_key\nTEST_KEY_C=ok\n")
        env_loader.load_dotenv(env_path)
        self.assertEqual(os.environ["TEST_KEY_C"], "ok")
        self.assertNotIn("no_equals_sign", os.environ)

    def test_never_overrides_existing_env(self):
        os.environ["TEST_KEY_D"] = "system"
        env_path = self._write_env("TEST_KEY_D=dotenv\n")
        env_loader.load_dotenv(env_path)
        self.assertEqual(os.environ["TEST_KEY_D"], "system")

    def test_value_with_equals_sign_is_preserved(self):
        env_path = self._write_env("TEST_KEY_E=a=b=c\n")
        env_loader.load_dotenv(env_path)
        self.assertEqual(os.environ["TEST_KEY_E"], "a=b=c")

    def test_relative_path_missing_is_noop(self):
        env_loader.load_dotenv("definitely_not_a_real_file_xyz.env")


class RequireEnvTests(unittest.TestCase):
    def test_all_set_returns(self):
        with patch.dict(os.environ, {"REQ_A": "1", "REQ_B": "2"}):
            env_loader.require_env("REQ_A", "REQ_B")

    def test_missing_exits_one(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                env_loader.require_env("REQ_MISSING")
        self.assertEqual(ctx.exception.code, 1)

    def test_empty_string_counts_as_missing(self):
        with patch.dict(os.environ, {"REQ_EMPTY": ""}):
            with self.assertRaises(SystemExit):
                env_loader.require_env("REQ_EMPTY")

    def test_no_keys_returns(self):
        env_loader.require_env()


class WarnEnvTests(unittest.TestCase):
    def test_warns_on_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertLogs(env_loader.logger, level="WARNING") as logs:
                env_loader.warn_env("WARN_MISSING_X", "WARN_MISSING_Y")
        self.assertIn("WARN_MISSING_X", logs.output[0])
        self.assertIn("WARN_MISSING_Y", logs.output[0])

    def test_silent_when_set(self):
        with patch.dict(os.environ, {"WARN_SET": "1"}):
            with self.assertNoLogs(env_loader.logger, level="WARNING"):
                env_loader.warn_env("WARN_SET")


class ProviderEnvKeysTests(unittest.TestCase):
    def test_known_providers(self):
        self.assertEqual(env_loader._provider_env_keys("openai", {}),
                         ("OPENAI_API_KEY",))
        self.assertEqual(env_loader._provider_env_keys("azure", {}),
                         ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"))
        self.assertEqual(env_loader._provider_env_keys("ollama", {}), ())
        self.assertEqual(env_loader._provider_env_keys("bedrock", {}),
                         ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                          "AWS_REGION_NAME"))

    def test_unknown_provider_raises_valueerror(self):
        with self.assertRaises(ValueError):
            env_loader._provider_env_keys("no_such_provider", {})

    def test_openai_compatible_without_base_url_needs_both(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = env_loader._provider_env_keys("openai_compatible", {})
        self.assertIn("OPENAI_COMPATIBLE_BASE_URL", keys)
        self.assertIn("OPENAI_COMPATIBLE_API_KEY", keys)

    def test_openai_compatible_with_profile_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = env_loader._provider_env_keys(
                "openai_compatible", {"base_url": "http://x/v1"})
        self.assertEqual(keys, ("OPENAI_COMPATIBLE_API_KEY",))

    def test_openai_compatible_no_key_required(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = env_loader._provider_env_keys(
                "openai_compatible",
                {"base_url": "http://x/v1", "api_key_required": False})
        self.assertEqual(keys, ())

    def test_openai_compatible_custom_key_name(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = env_loader._provider_env_keys(
                "openai_compatible",
                {"base_url": "http://x", "api_key_env": "MY_KEY"})
        self.assertIn("MY_KEY", keys)


class RequireProviderEnvTests(unittest.TestCase):
    def test_missing_provider_credentials_exits(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "azure"}):
                with self.assertRaises(SystemExit):
                    env_loader.require_provider_env()

    def test_set_provider_credentials_passes(self):
        env = {"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "e"}
        with patch.dict(os.environ, env, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "azure"}):
                env_loader.require_provider_env()

    def test_explicit_provider_overrides_profile(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "azure"}):
                env_loader.require_provider_env("openai")

    def test_unknown_provider_raises_valueerror(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "bogus"}):
                with self.assertRaises(ValueError):
                    env_loader.require_provider_env()


class WarnProviderEnvTests(unittest.TestCase):
    def test_warns_missing_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "gemini"}):
                with self.assertLogs(env_loader.logger, level="WARNING"):
                    env_loader.warn_provider_env()

    def test_unknown_provider_silent(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "bogus"}):
                with self.assertNoLogs(env_loader.logger, level="WARNING"):
                    env_loader.warn_provider_env()

    def test_openai_compatible_silent(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "openai_compatible"}):
                with self.assertNoLogs(env_loader.logger, level="WARNING"):
                    env_loader.warn_provider_env()


if __name__ == "__main__":
    unittest.main()
