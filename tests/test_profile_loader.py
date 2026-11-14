"""Tests for core/profile_loader.py: profiles.yaml resolution and caching."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from core import profile_loader


class ProfileLoaderTests(unittest.TestCase):
    def setUp(self):
        self._cache_backup = profile_loader._profile_cache
        self._name_backup = profile_loader._active_name_cache
        self._path_backup = profile_loader._PROFILES_PATH
        profile_loader._profile_cache = None
        profile_loader._active_name_cache = None

    def tearDown(self):
        profile_loader._profile_cache = self._cache_backup
        profile_loader._active_name_cache = self._name_backup
        profile_loader._PROFILES_PATH = self._path_backup

    def _write_profiles(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False)
        handle.write(content)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        path = Path(handle.name)
        profile_loader._PROFILES_PATH = path
        return path

    def test_missing_file_returns_empty(self):
        profile_loader._PROFILES_PATH = Path("/nonexistent/profiles.yaml")
        self.assertEqual(profile_loader.load_profile(), {})

    def test_missing_file_active_name_is_none(self):
        profile_loader._PROFILES_PATH = Path("/nonexistent/profiles.yaml")
        self.assertEqual(profile_loader.active_profile_name(), "none")

    def test_missing_file_get_model_uses_fallback(self):
        profile_loader._PROFILES_PATH = Path("/nonexistent/profiles.yaml")
        self.assertEqual(profile_loader.get_model("model", "fallback-x"),
                         "fallback-x")

    def test_active_profile_selected(self):
        self._write_profiles(
            "active_profile: p1\n"
            "profiles:\n"
            "  p1:\n    provider: openai\n    model: m1\n"
            "  p2:\n    provider: azure\n    model: m2\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(profile_loader.load_profile()["model"], "m1")
            self.assertEqual(profile_loader.active_profile_name(), "p1")

    def test_env_overrides_active_profile(self):
        self._write_profiles(
            "active_profile: p1\n"
            "profiles:\n"
            "  p1:\n    provider: openai\n    model: m1\n"
            "  p2:\n    provider: azure\n    model: m2\n")
        with patch.dict(os.environ, {"ACTIVE_PROFILE": "p2"}):
            self.assertEqual(profile_loader.load_profile()["model"], "m2")

    def test_unknown_profile_raises_valueerror(self):
        self._write_profiles(
            "active_profile: p1\n"
            "profiles:\n"
            "  p1:\n    provider: openai\n")
        with patch.dict(os.environ, {"ACTIVE_PROFILE": "ghost"}):
            with self.assertRaises(ValueError) as ctx:
                profile_loader.load_profile()
        self.assertIn("ghost", str(ctx.exception))

    def test_get_model_prefers_profile_over_fallback(self):
        self._write_profiles(
            "active_profile: p1\n"
            "profiles:\n"
            "  p1:\n    model: m1\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(profile_loader.get_model("model", "x"), "m1")
            self.assertEqual(profile_loader.get_model("judge_model", "j"), "j")

    def test_profile_is_cached(self):
        path = self._write_profiles(
            "active_profile: p1\nprofiles:\n  p1:\n    model: m1\n")
        with patch.dict(os.environ, {}, clear=True):
            profile_loader.load_profile()
            path.write_text(
                "active_profile: p1\nprofiles:\n  p1:\n    model: CHANGED\n")
            self.assertEqual(profile_loader.load_profile()["model"], "m1")

    def test_empty_yaml_returns_empty_profile(self):
        self._write_profiles("")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                profile_loader.load_profile()


if __name__ == "__main__":
    unittest.main()
