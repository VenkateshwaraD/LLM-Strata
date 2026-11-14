"""Tests for pure logic in the remaining layer runners.

Heavy third-party SDKs (deepeval, llm_guard scanner objects) are stubbed at the
module boundary; provider clients use the shared FakeClient stub.
"""

import importlib.util
import io
import json
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


class FakeClient:
    """Chat-completions-compatible stub replaying canned outputs."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        content = next(self.outputs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FailingClient(FakeClient):
    def create(self, **kwargs):
        raise RuntimeError("provider down")


multilingual = load_runner("layers/attack/multilingual/run_multilingual.py")
many_shot = load_runner("layers/attack/in_context_attacks/run_many_shot.py")
deepeval_runner = load_runner("layers/scan/deepeval/run_deepeval.py")
pyrit_runner = load_runner("layers/scan/pyrit/run_pyrit.py")
guard_runner = load_runner("layers/scan/llm_guard/run_guard.py")
redteam_runner = load_runner("layers/attack/human_redteam/run_human_redteam.py")


class MultilingualTests(unittest.TestCase):
    def test_direct_returns_text(self):
        self.assertEqual(
            multilingual.build_prompt("direct", "Spanish", "hola"), "hola")

    def test_mixed_framing_wraps(self):
        out = multilingual.build_prompt("mixed_framing", "Spanish", "hola")
        self.assertIn("Spanish", out)
        self.assertIn("hola", out)

    def test_translate_bridge_wraps(self):
        out = multilingual.build_prompt("translate_bridge", "Arabic", "text")
        self.assertIn("Arabic", out)
        self.assertIn("translate", out.lower())

    def test_unknown_technique_raises(self):
        with self.assertRaises(ValueError):
            multilingual.build_prompt("bogus", "x", "t")

    def _config(self) -> dict:
        return {
            "attacks": [{"category": "c1", "intent": "i",
                         "translations": {"es": "ataque", "ar": "هجوم"}}],
            "languages": [{"code": "es", "name": "Spanish"},
                          {"code": "ar", "name": "Arabic"},
                          {"code": "fr", "name": "French"}],
            "techniques": ["direct", "mixed_framing"],
        }

    def test_skips_languages_without_translation(self):
        # 'fr' has no translation for the attack -> only es+ar x 2 techniques
        with patch.object(multilingual, "get_client",
                          return_value=FakeClient(["resp", "SAFE"])):
            results = multilingual.run_multilingual_tests(
                self._config(), filter_lang=None, filter_technique=None,
                filter_category=None)
        self.assertEqual(len(results), 4)

    def test_filter_lang(self):
        with patch.object(multilingual, "get_client",
                          return_value=FakeClient(["resp", "SAFE"])):
            results = multilingual.run_multilingual_tests(
                self._config(), filter_lang="ar", filter_technique=None,
                filter_category=None)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["lang_code"] == "ar" for result in results))

    def test_empty_config(self):
        with patch.object(multilingual, "get_client",
                          return_value=FakeClient([])):
            self.assertEqual(multilingual.run_multilingual_tests(
                {}, filter_lang=None, filter_technique=None,
                filter_category=None), [])


class ManyShotRunnerTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "model": "m", "evaluator_model": "e",
            "benign_shots": [{"question": "b", "answer": "x"}],
            "harmful_shots": [{"question": "h", "answer": "y"}],
            "target_attacks": [{"category": "c1", "intent": "i",
                                "question": "q1"},
                               {"category": "c2", "intent": "i2",
                                "question": "q2"}],
            "shot_counts": [5],
            "strategies": ["pure_harmful", "camouflaged"],
        }

    def test_matrix_size(self):
        with patch.object(many_shot, "get_client",
                          return_value=FakeClient(["resp", "SAFE"])):
            results = many_shot.run_many_shot_tests(self._config())
        self.assertEqual(len(results), 4)  # 2 attacks x 2 strategies x 1 count

    def test_filter_strategy(self):
        with patch.object(many_shot, "get_client",
                          return_value=FakeClient(["resp", "SAFE"])):
            results = many_shot.run_many_shot_tests(
                self._config(), filter_strategy="camouflaged")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["strategy"] == "camouflaged" for result in results))

    def test_failing_client_records_errors(self):
        with patch.object(many_shot, "get_client",
                          return_value=FailingClient([])):
            results = many_shot.run_many_shot_tests(self._config())
        self.assertTrue(all(result["verdict"] == "ERROR" for result in results))

    def test_gradual_escalation_ratio(self):
        shots = many_shot._gradual_escalation_shots(
            10, [{"question": "b", "answer": "x"}],
            [{"question": "h", "answer": "y"}])
        benign = sum(1 for shot in shots if shot["question"] == "b")
        self.assertEqual((benign, len(shots) - benign), (6, 4))


def _fake_deepeval():
    """Build a stub `deepeval` package tree at the external boundary."""
    package = types.ModuleType("deepeval")
    test_case_mod = types.ModuleType("deepeval.test_case")
    metrics_mod = types.ModuleType("deepeval.metrics")
    models_mod = types.ModuleType("deepeval.models")
    base_mod = types.ModuleType("deepeval.models.base_model")

    class LLMTestCase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DeepEvalBaseLLM:
        pass

    class _Metric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.score = 0.9
            self.success = True
            self.reason = "ok"

        def measure(self, case):
            self.case = case

    test_case_mod.LLMTestCase = LLMTestCase
    for name in ("BiasMetric", "HallucinationMetric", "MisuseMetric",
                 "PIILeakageMetric", "ToxicityMetric"):
        setattr(metrics_mod, name, _Metric)
    base_mod.DeepEvalBaseLLM = DeepEvalBaseLLM
    package.test_case = test_case_mod
    package.metrics = metrics_mod
    package.models = models_mod
    models_mod.base_model = base_mod
    return {"deepeval": package, "deepeval.test_case": test_case_mod,
            "deepeval.metrics": metrics_mod, "deepeval.models": models_mod,
            "deepeval.models.base_model": base_mod}


class DeepEvalParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(
            deepeval_runner._parse_structured_output('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        out = deepeval_runner._parse_structured_output(
            "```json\n{\"a\": 2}\n```")
        self.assertEqual(out, {"a": 2})

    def test_prose_wrapped_json(self):
        out = deepeval_runner._parse_structured_output(
            'Sure! Here is the result: {"a": 3} done.')
        self.assertEqual(out, {"a": 3})

    def test_invalid_raises_valueerror(self):
        with self.assertRaises(ValueError):
            deepeval_runner._parse_structured_output("no json at all")

    def test_broken_json_raises_valueerror(self):
        with self.assertRaises(ValueError):
            deepeval_runner._parse_structured_output("{unclosed: ")


class DeepEvalRunTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "model": "m", "evaluator_model": "e",
            "metrics": [{"name": "bias", "threshold": 0.4}],
            "test_cases": [
                {"category": "bias", "input": "in1"},
                {"category": "toxicity", "input": "in2"},
                {"category": "unknown_cat", "input": "in3"},
            ],
        }

    def test_run_all_tests_filters_and_skips_unknown(self):
        with patch.dict(sys.modules, _fake_deepeval()):
            with patch.object(deepeval_runner, "get_client",
                              return_value=FakeClient(["resp", "SAFE"])):
                results = deepeval_runner.run_all_tests(
                    self._config(), filter_metric=None, verbose=False)
        # 2 valid categories; 'unknown_cat' is skipped
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["passed"] for result in results))

    def test_filter_metric(self):
        with patch.dict(sys.modules, _fake_deepeval()):
            with patch.object(deepeval_runner, "get_client",
                              return_value=FakeClient(["resp", "SAFE"])):
                results = deepeval_runner.run_all_tests(
                    self._config(), filter_metric="toxicity", verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metric"], "toxicity")

    def test_failing_client_records_failure(self):
        with patch.dict(sys.modules, _fake_deepeval()):
            with patch.object(deepeval_runner, "get_client",
                              return_value=FailingClient([])):
                results = deepeval_runner.run_all_tests(
                    self._config(), filter_metric="bias", verbose=False)
        self.assertFalse(results[0]["passed"])
        self.assertIsNone(results[0]["score"])

    def test_failure_result_shape(self):
        result = deepeval_runner._failure_result("bias", "in", ValueError("x"))
        self.assertEqual(result["metric"], "bias")
        self.assertFalse(result["passed"])


class PyritHelperTests(unittest.TestCase):
    def test_openai_endpoint_from_profile(self):
        endpoint, key = pyrit_runner._openai_endpoint(
            {"endpoint": "http://p", "base_url": "http://b"})
        self.assertEqual(endpoint, "http://p")
        self.assertIsNone(key)

    def test_azure_endpoint_suffix(self):
        with patch.dict("os.environ", {}, clear=True):
            endpoint, _ = pyrit_runner._azure_endpoint(
                {"endpoint": "https://e.openai.azure.com/"})
        self.assertEqual(endpoint, "https://e.openai.azure.com/openai/v1")

    def test_azure_endpoint_keeps_v1(self):
        endpoint, _ = pyrit_runner._azure_endpoint(
            {"endpoint": "https://e/openai/v1"})
        self.assertEqual(endpoint, "https://e/openai/v1")

    def test_azure_endpoint_none(self):
        with patch.dict("os.environ", {}, clear=True):
            endpoint, _ = pyrit_runner._azure_endpoint({})
        self.assertIsNone(endpoint)

    def test_compatible_endpoint_env_fallback(self):
        with patch.dict("os.environ",
                        {"OPENAI_COMPATIBLE_BASE_URL": "http://x"}):
            endpoint, key = pyrit_runner._compatible_endpoint({})
        self.assertEqual(endpoint, "http://x")
        self.assertEqual(key, "not-needed")

    def test_result_value_first_match(self):
        obj = SimpleNamespace(outcome=None, achieved_objective=True)
        self.assertTrue(
            pyrit_runner._result_value(obj, "outcome", "achieved_objective"))

    def test_result_value_default(self):
        self.assertEqual(pyrit_runner._result_value(
            SimpleNamespace(), "missing", default="d"), "d")

    def test_json_value_passthrough_primitives(self):
        self.assertEqual(pyrit_runner._json_value(None), None)
        self.assertEqual(pyrit_runner._json_value(3), 3)
        self.assertEqual(pyrit_runner._json_value("s"), "s")

    def test_json_value_coerces_objects(self):
        self.assertIsInstance(pyrit_runner._json_value(Path("/x")), str)

    def test_is_successful_outcome(self):
        result = SimpleNamespace(outcome=SimpleNamespace(value="success"))
        self.assertTrue(pyrit_runner._is_successful_result(result))

    def test_is_successful_enum_string(self):
        result = SimpleNamespace(outcome="SUCCESS")
        self.assertTrue(pyrit_runner._is_successful_result(result))

    def test_is_successful_falls_back_to_objective(self):
        result = SimpleNamespace(outcome=None, achieved_objective=True)
        self.assertTrue(pyrit_runner._is_successful_result(result))

    def test_is_successful_false(self):
        result = SimpleNamespace(outcome="failure")
        self.assertFalse(pyrit_runner._is_successful_result(result))

    def test_select_objectives_filter(self):
        config = {"objectives": [{"id": "a"}, {"id": "b"}]}
        selected = pyrit_runner.select_objectives(config, "b")
        self.assertEqual([obj["id"] for obj in selected], ["b"])

    def test_select_objectives_all(self):
        config = {"objectives": [{"id": "a"}, {"id": "b"}]}
        self.assertEqual(len(pyrit_runner.select_objectives(config, None)), 2)

    def test_select_objectives_unknown_id_exits(self):
        config = {"objectives": [{"id": "a"}]}
        with self.assertRaises(SystemExit):
            pyrit_runner.select_objectives(config, "ghost")


class GuardScanTests(unittest.TestCase):
    class _Scanner:
        def __init__(self, name, valid=True, score=0.0):
            self.name = name
            self._valid = valid
            self._score = score

        def scan(self, text, **kwargs):
            return f"[{self.name}]{text}", self._valid, self._score

    def test_scan_prompt_chains_scanners(self):
        scanners = [self._Scanner("s1"), self._Scanner("s2", valid=False)]
        sanitized, valids, scores = guard_runner.scan_prompt(scanners, "p")
        self.assertEqual(sanitized, "[s2][s1]p")
        self.assertEqual(valids, {"s1": True, "s2": False})
        self.assertIn("s1", scores)

    def test_scan_output(self):
        scanners = [self._Scanner("o1")]
        sanitized, valids, _ = guard_runner.scan_output(scanners, "p", "out")
        self.assertEqual(sanitized, "[o1]out")
        self.assertEqual(valids, {"o1": True})

    def test_scan_empty_scanners_passthrough(self):
        sanitized, valids, scores = guard_runner.scan_prompt([], "p")
        self.assertEqual((sanitized, valids, scores), ("p", {}, {}))

    def test_save_report_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(guard_runner, "OUTPUT_DIR", Path(tmp)):
                report_path = guard_runner.save_report([{"a": 1}], 1, 2, 3)
            data = json.loads(report_path.read_text())
        self.assertEqual(data["passed"], 1)
        self.assertEqual(data["blocked"], 2)


class HumanRedteamTests(unittest.TestCase):
    def test_check_category_valid(self):
        config = {"attack_categories": [{"id": "a"}, {"id": "b"}]}
        redteam_runner.check_category(config, "b")

    def test_check_category_invalid_exits(self):
        config = {"attack_categories": [{"id": "a"}]}
        with self.assertRaises(SystemExit):
            redteam_runner.check_category(config, "zz")

    def test_resolve_output_dir_relative(self):
        out = redteam_runner.resolve_output_dir(
            {"output_dir": "results/human_redteam"})
        self.assertEqual(out, redteam_runner.PROJECT_ROOT / "results/human_redteam")

    def test_resolve_output_dir_absolute(self):
        out = redteam_runner.resolve_output_dir({"output_dir": "/tmp/rt"})
        self.assertEqual(out, Path("/tmp/rt"))

    def test_show_reports_handles_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "human_redteam_bad.json").write_text("{corrupt")
            good = {"timestamp": "t", "breached": 1, "borderline": 0,
                    "held": 2}
            (tmp_path / "human_redteam_ok.json").write_text(json.dumps(good))
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                redteam_runner.show_reports(tmp_path)
        self.assertIn("breached=1", buffer.getvalue())

    def test_show_reports_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                redteam_runner.show_reports(Path(tmp))
        self.assertIn("No reports", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
