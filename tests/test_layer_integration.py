"""Tests for langfuse boundary behavior, agent_security runner sections,
garak subprocess handling, and human_redteam non-interactive helpers."""

import importlib.util
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401

REPO_ROOT = _bootstrap.REPO_ROOT


def load_runner(relative_path: str):
    """Import a run_*.py file by repo-relative path."""
    spec = importlib.util.spec_from_file_location(
        f"runner_{Path(relative_path).stem}", REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


langfuse_runner = load_runner("layers/monitor/langfuse/run_langfuse.py")
agent_runner = load_runner("layers/attack/agent_security/run_agent_security.py")
garak_runner = load_runner("layers/scan/garak/run_garak.py")


def _fake_langfuse_module(fail_flush: bool = False):
    """Build a stub `langfuse` package at the external boundary."""
    module = types.ModuleType("langfuse")

    class _Client:
        def __init__(self):
            self.flushed = False

        def flush(self):
            if fail_flush:
                raise ConnectionError("unreachable")
            self.flushed = True

    module.get_client = lambda: _Client()
    return module


class LangfuseBoundaryTests(unittest.TestCase):
    def test_check_langfuse_missing_exits(self):
        real = sys.modules.pop("langfuse", None)
        self.addCleanup(lambda: sys.modules.__setitem__("langfuse", real)
                        if real else None)
        with patch.dict(sys.modules, {"langfuse": None}):
            with self.assertRaises(SystemExit):
                langfuse_runner.check_langfuse()

    def test_get_langfuse_client_sets_default_host(self):
        import os
        with patch.dict(sys.modules, {"langfuse": _fake_langfuse_module()}):
            with patch.dict("os.environ", {}, clear=True):
                client = langfuse_runner.get_langfuse_client({})
                self.assertEqual(os.environ["LANGFUSE_BASE_URL"],
                                 langfuse_runner.DEFAULT_HOST)
        self.assertIsNotNone(client)

    def test_get_langfuse_client_respects_env_host(self):
        import os
        with patch.dict(sys.modules, {"langfuse": _fake_langfuse_module()}):
            with patch.dict("os.environ",
                            {"LANGFUSE_BASE_URL": "https://self.host"}):
                langfuse_runner.get_langfuse_client({})
                self.assertEqual(os.environ["LANGFUSE_BASE_URL"],
                                 "https://self.host")

    def test_verify_connection_ok(self):
        with patch.dict(sys.modules, {"langfuse": _fake_langfuse_module()}):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertTrue(langfuse_runner.verify_connection({}))
        self.assertIn("OK", buffer.getvalue())

    def test_verify_connection_failure_returns_false(self):
        with patch.dict(sys.modules,
                        {"langfuse": _fake_langfuse_module(fail_flush=True)}):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertFalse(langfuse_runner.verify_connection({}))
        self.assertIn("FAILED", buffer.getvalue())


class AgentRunnerSectionTests(unittest.TestCase):
    """run_sections uses keyword-only args namespace + pure memory/tool tests."""

    def _args(self, **overrides):
        values = {"multi_turn_only": False, "tool_auth_only": False,
                  "memory_only": False}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_tool_auth_only_runs_without_client(self):
        config = {
            "tool_policies": {"analyst": {"allowed_tools": ["search"]}},
            "tool_authorization_tests": [
                {"id": "t", "role": "analyst", "tool": "search",
                 "expected": "allow"}],
        }
        sections = agent_runner.run_sections(
            self._args(tool_auth_only=True), config, "m")
        self.assertEqual(list(sections), ["tool_authorization"])
        self.assertTrue(sections["tool_authorization"][0]["passed"])

    def test_memory_only_runs_without_client(self):
        config = {
            "memory_policy": {"allow_untrusted_writes": False},
            "memory_tests": [{"id": "m", "operations": [
                {"action": "read", "scope": "s", "key": "k",
                 "expected_absent": True}]}],
        }
        sections = agent_runner.run_sections(
            self._args(memory_only=True), config, "m")
        self.assertEqual(list(sections), ["memory"])
        self.assertTrue(sections["memory"][0]["passed"])

    def test_multi_turn_requires_provider_env(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.profile_loader.load_profile",
                       return_value={"provider": "openai"}):
                with self.assertRaises(SystemExit):
                    agent_runner.run_sections(self._args(), {}, "m")

    def test_save_report_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(agent_runner, "OUTPUT_DIR", Path(tmp)):
                report_path = agent_runner.save_report({"summary": {"total": 1}})
            data = __import__("json").loads(report_path.read_text())
        self.assertEqual(data["summary"]["total"], 1)


class GarakSubprocessTests(unittest.TestCase):
    def test_check_garak_exits_when_version_fails(self):
        with patch.object(garak_runner, "get_garak_python",
                          return_value=sys.executable):
            with self.assertRaises(SystemExit):
                garak_runner.check_garak()

    def test_check_garak_ok_when_version_succeeds(self):
        fake = SimpleNamespace(returncode=0, stdout="garak 0.9.0\n")
        with patch.object(garak_runner, "get_garak_python", return_value="gp"):
            with patch.object(garak_runner.subprocess, "run",
                              return_value=fake) as mock_run:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    garak_runner.check_garak()
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0],
                         ["gp", "-m", "garak", "--version"])
        self.assertIn("garak 0.9.0", buffer.getvalue())

    def test_run_scan_delegates_exit_code(self):
        fake = SimpleNamespace(returncode=5)
        with patch.object(garak_runner.subprocess, "run",
                          return_value=fake):
            self.assertEqual(
                garak_runner.run_scan(["cmd"], Path("/tmp")), 5)


if __name__ == "__main__":
    unittest.main()
