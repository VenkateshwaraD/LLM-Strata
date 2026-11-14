"""Tests for each layer's config loading, output dir resolution, report
writing, and summary printing — including the empty-results edge case."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
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


RUNNERS = {
    name: load_runner(path) for name, path in {
        "encoded": "layers/attack/encoded_attacks/run_encoded_attacks.py",
        "many_shot": "layers/attack/in_context_attacks/run_many_shot.py",
        "multilingual": "layers/attack/multilingual/run_multilingual.py",
        "tool_inject": "layers/attack/tool_inject/run_tool_inject.py",
        "backdoor": "layers/attack/backdoor/run_backdoor.py",
        "rag_security": "layers/attack/rag_security/run_rag_security.py",
        "agent_security": "layers/attack/agent_security/run_agent_security.py",
        "human_redteam": "layers/attack/human_redteam/run_human_redteam.py",
        "deepeval": "layers/scan/deepeval/run_deepeval.py",
        "ragas": "layers/scan/ragas/run_ragas.py",
        "pyrit": "layers/scan/pyrit/run_pyrit.py",
        "garak": "layers/scan/garak/run_garak.py",
        "llm_guard": "layers/scan/llm_guard/run_guard.py",
        "langfuse": "layers/monitor/langfuse/run_langfuse.py",
    }.items()
}


class LoadConfigTests(unittest.TestCase):
    """Every layer's YAML must parse to a dict — catches config file breakage."""

    def test_all_configs_load(self):
        for name, module in RUNNERS.items():
            with self.subTest(layer=name):
                config = module.load_config()
                self.assertIsInstance(config, dict)
                self.assertTrue(config, f"{name} config is empty")


class ResolveOutputDirTests(unittest.TestCase):
    """Layers resolve `output_dir` configs against the repo root; garak and
    llm_guard use nested keys (reporting.output_dir / settings.log_file)."""

    GENERIC_LAYERS = [name for name in RUNNERS
                      if name not in ("garak", "llm_guard", "agent_security",
                                      "langfuse")]

    def test_relative_anchors_at_repo_root(self):
        for name in self.GENERIC_LAYERS:
            module = RUNNERS[name]
            with self.subTest(layer=name):
                out = module.resolve_output_dir({"output_dir": "results/x"})
                self.assertEqual(out, module.PROJECT_ROOT / "results/x")

    def test_absolute_passthrough(self):
        for name in self.GENERIC_LAYERS:
            module = RUNNERS[name]
            with self.subTest(layer=name):
                self.assertEqual(module.resolve_output_dir(
                    {"output_dir": "/tmp/abs"}), Path("/tmp/abs"))

    def test_garak_nested_config_key(self):
        module = RUNNERS["garak"]
        out = module.resolve_output_dir({"reporting": {"output_dir": "res/x"}})
        self.assertEqual(out, module.PROJECT_ROOT / "res/x")
        out_abs = module.resolve_output_dir(
            {"reporting": {"output_dir": "/tmp/g"}})
        self.assertEqual(out_abs, Path("/tmp/g"))

    def test_llm_guard_log_file_parent(self):
        module = RUNNERS["llm_guard"]
        out = module.resolve_output_dir({"settings": {"log_file": "res/g.log"}})
        self.assertEqual(out, module.PROJECT_ROOT / "res")
        out_abs = module.resolve_output_dir(
            {"settings": {"log_file": "/tmp/g/g.log"}})
        self.assertEqual(out_abs, Path("/tmp/g"))


class SaveResultsTests(unittest.TestCase):
    """Each layer writes a timestamped JSON report that parses and keeps totals."""

    SAMPLE_RESULT = {"verdict": "SAFE", "safe": True, "armed": False,
                     "amplification": False, "injection_verdict": "SAFE",
                     "pii_leaked": False, "score": 0.9, "passed": True,
                     "breached": False, "bypassed_scanner": False}

    def _save(self, name: str, module, output_dir: Path) -> Path:
        if name == "pyrit":
            return module.save_results([self.SAMPLE_RESULT], output_dir, {})
        return module.save_results([self.SAMPLE_RESULT], output_dir)

    def test_report_roundtrip(self):
        for name, module in RUNNERS.items():
            if not hasattr(module, "save_results"):
                continue
            with self.subTest(layer=name):
                with tempfile.TemporaryDirectory() as tmp:
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        report = self._save(name, module, Path(tmp))
                    self.assertTrue(report.exists())
                    data = json.loads(report.read_text())
                self.assertIn("timestamp", data)


class PrintSummaryEmptyTests(unittest.TestCase):
    """print_summary on [] must not crash — guards max(total,1) regressions."""

    def test_empty_results_no_crash(self):
        for name, module in RUNNERS.items():
            if not hasattr(module, "print_summary"):
                continue
            with self.subTest(layer=name):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    module.print_summary([])
                self.assertTrue(buffer.getvalue())


class RagasHelpersTests(unittest.TestCase):
    def test_failure_result_shape(self):
        result = RUNNERS["ragas"]._failure_result(
            {"id": "t", "question": "q"}, ValueError("boom"))
        self.assertFalse(result["passed"])
        self.assertIn("boom", result["error"])

    def test_run_cases_failure_recorded(self):
        runner = RUNNERS["ragas"]
        with patch.object(
                runner, "evaluate_test_case", side_effect=RuntimeError("x")):
            results = runner.run_cases(
                [{"id": "a", "question": "q"}], model="m",
                judge_model="j", threshold=0.5, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["passed"])


class GarakHelpersTests(unittest.TestCase):
    def test_garak_python_env_override(self):
        with patch.dict("os.environ", {"GARAK_PYTHON": sys.executable}):
            self.assertEqual(RUNNERS["garak"].get_garak_python(), sys.executable)

    def test_garak_python_missing_env_exits(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                RUNNERS["garak"].get_garak_python()


if __name__ == "__main__":
    unittest.main()
