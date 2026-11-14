"""Tests for pure logic across the layer runners (loaded via importlib).

Covers encoders, prompt builders, judges (with a fake client at the provider
boundary), filtering, and regressions for bugs fixed during the reorganization.
"""

import base64
import codecs
import importlib.util
import unittest
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


encoded = load_runner("layers/attack/encoded_attacks/run_encoded_attacks.py")
many_shot = load_runner("layers/attack/in_context_attacks/run_many_shot.py")
tool_inject = load_runner("layers/attack/tool_inject/run_tool_inject.py")
backdoor = load_runner("layers/attack/backdoor/run_backdoor.py")
rag_security = load_runner("layers/attack/rag_security/run_rag_security.py")
garak = load_runner("layers/scan/garak/run_garak.py")
langfuse_runner = load_runner("layers/monitor/langfuse/run_langfuse.py")


class EncoderTests(unittest.TestCase):
    def test_base64_roundtrip(self):
        out = encoded.encode_base64("hello")
        self.assertEqual(base64.b64decode(out).decode(), "hello")

    def test_rot13_roundtrip(self):
        out = encoded.encode_rot13("hello")
        self.assertEqual(codecs.decode(out, "rot13"), "hello")

    def test_hex_roundtrip(self):
        out = encoded.encode_hex("hi")
        decoded = bytes(int(token, 16) for token in out.split()).decode()
        self.assertEqual(decoded, "hi")

    def test_reversed(self):
        self.assertEqual(encoded.encode_reversed("abc"), "cba")

    def test_homoglyphs_uses_cyrillic(self):
        out = encoded.encode_homoglyphs("aeo")
        self.assertEqual(out, "аео")

    def test_zero_width_inserts_zwsp(self):
        out = encoded.encode_zero_width("ab")
        self.assertIn("​", out)
        self.assertEqual(out.replace("​", ""), "ab")

    def test_empty_input(self):
        self.assertEqual(encoded.encode_base64(""), "")
        self.assertEqual(encoded.encode_homoglyphs(""), "")
        self.assertEqual(encoded.encode_zero_width(""), "")

    def test_build_attack_prompt_wraps(self):
        prompt = encoded.build_attack_prompt("base64", "ENCODED")
        self.assertIn("ENCODED", prompt)
        self.assertIn("base64", prompt.lower())

    def test_build_attack_prompt_unknown_method_passthrough(self):
        self.assertEqual(encoded.build_attack_prompt("zzz", "X"), "X")


class JudgeTests(unittest.TestCase):
    def test_safe_verdict(self):
        verdict = encoded.judge_response(
            FakeClient(["SAFE"]), "m", "intent", "base64", "resp")
        self.assertEqual(verdict, "SAFE")

    def test_unsafe_verdict(self):
        verdict = encoded.judge_response(
            FakeClient(["UNSAFE"]), "m", "intent", "base64", "resp")
        self.assertEqual(verdict, "UNSAFE")

    def test_unparseable_verdict_is_unknown(self):
        verdict = encoded.judge_response(
            FakeClient(["I think..."]), "m", "intent", "base64", "resp")
        self.assertEqual(verdict, "UNKNOWN")

    def test_client_error_is_unknown(self):
        verdict = encoded.judge_response(
            FailingClient([]), "m", "intent", "base64", "resp")
        self.assertEqual(verdict, "UNKNOWN")


class ManyShotTests(unittest.TestCase):
    def test_pure_harmful_uses_harmful_shots(self):
        shots = many_shot.build_shot_list(
            "pure_harmful", 3, [{"question": "b", "answer": "x"}],
            [{"question": "h", "answer": "y"}])
        self.assertEqual(len(shots), 3)
        self.assertTrue(all(shot["question"] == "h" for shot in shots))

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            many_shot.build_shot_list("nope", 1, [], [])

    def test_zero_shots_returns_empty(self):
        self.assertEqual(many_shot.build_shot_list("pure_harmful", 0, [], []), [])

    def test_shot_pool_cycles(self):
        shots = many_shot.build_shot_list(
            "pure_harmful", 3, [], [{"question": "h", "answer": "y"}])
        self.assertEqual(len(shots), 3)

    def test_build_messages_alternates_roles(self):
        messages = many_shot.build_messages(
            [{"question": "q1", "answer": "a1"}], "target")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[-1]["content"], "target")


class ToolInjectTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "agent_system_prompt": "sys",
            "user_tasks": [
                {"vector": "search_result_injection", "task_id": "t1",
                 "user_request": "do x"},
                {"vector": "file_content_injection", "task_id": "t2",
                 "user_request": "do y"},
            ],
            "injection_payloads": [
                {"goal": "g1", "description": "d1", "payload": "P1"},
                {"goal": "g2", "description": "d2", "payload": "P2"},
            ],
        }

    def test_all_task_vector_pairs_run(self):
        """Regression: early `return` inside the loop used to drop all but
        the first task's results."""
        with patch.object(tool_inject, "get_client",
                          return_value=FakeClient(["resp", "SAFE"])):
            results = tool_inject.run_tool_inject_tests(self._config())
        # 2 tasks x 2 payloads must all appear
        self.assertEqual(len(results), 4)
        vectors = {result["vector"] for result in results}
        self.assertEqual(vectors,
                         {"search_result_injection", "file_content_injection"})

    def test_filter_vector(self):
        with patch.object(tool_inject, "get_client",
                          return_value=FakeClient(["resp", "SAFE"])):
            results = tool_inject.run_tool_inject_tests(
                self._config(), filter_vector="file_content_injection")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["vector"] == "file_content_injection"
                            for result in results))

    def test_failing_client_marks_error_and_continues(self):
        with patch.object(tool_inject, "get_client",
                          return_value=FailingClient([])):
            results = tool_inject.run_tool_inject_tests(self._config())
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["verdict"] == "ERROR" for result in results))

    def test_empty_config_returns_empty(self):
        with patch.object(tool_inject, "get_client",
                          return_value=FakeClient([])):
            self.assertEqual(tool_inject.run_tool_inject_tests({}), [])

    def test_build_tool_result_uses_template(self):
        out = tool_inject.build_tool_result("search_result_injection", "t", "PAY")
        self.assertIn("PAY", out)

    def test_build_tool_result_fallback_for_unknown_vector(self):
        out = tool_inject.build_tool_result("no_such_vector", "t", "PAY")
        self.assertIn("PAY", out)

    def test_database_vector_truncates_payload(self):
        out = tool_inject.build_tool_result(
            "database_row_injection", "t", "x" * 200)
        self.assertIn("x" * 80, out)
        self.assertNotIn("x" * 81, out)


class BackdoorTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "target_attacks": [
                {"category": "malware", "intent": "harm",
                 "question": "how to harm"},
                {"category": "weapons", "intent": "boom",
                 "question": "how to boom"},
            ],
            "triggers": [
                {"id": "dan", "category": "dan_variants", "prefix": "DAN"},
            ],
        }

    def test_baseline_plus_trigger_per_attack(self):
        with patch.object(backdoor, "get_client",
                          return_value=FakeClient(["resp"] * 12 + ["SAFE"] * 12)):
            results = backdoor.run_backdoor_tests(
                self._config(), filter_trigger_category=None,
                filter_trigger_id=None, filter_attack_category=None,
                verbose=False)
        # 2 attacks x (1 baseline + 1 trigger)
        self.assertEqual(len(results), 4)
        self.assertEqual(sum(1 for result in results if not result["armed"]), 2)

    def test_attack_category_filter(self):
        with patch.object(backdoor, "get_client",
                          return_value=FakeClient(["resp", "SAFE"] * 6)):
            results = backdoor.run_backdoor_tests(
                self._config(), filter_trigger_category=None,
                filter_trigger_id=None, filter_attack_category="weapons",
                verbose=False)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["attack_category"] == "weapons"
                            for result in results))

    def test_amplification_flag(self):
        result = backdoor._armed_result(
            {"category": "c", "intent": "i", "question": "q"},
            {"id": "t", "category": "cat"}, "p", "r", "UNSAFE", True)
        self.assertTrue(result["amplification"])
        self.assertFalse(result["safe"])

    def test_baseline_result_shape(self):
        result = backdoor._baseline_result(
            {"category": "c", "intent": "i", "question": "q"}, "r", "SAFE")
        self.assertEqual(result["trigger_id"], "BASELINE")
        self.assertTrue(result["safe"])
        self.assertFalse(result["amplification"])


class RagSecurityTests(unittest.TestCase):
    def test_build_messages_injects_context(self):
        messages = rag_security.build_messages("sys", "question", "CONTEXT")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("CONTEXT", messages[1]["content"])
        self.assertIn("question", messages[1]["content"])

    def test_judge_pii_yes_no_unknown(self):
        self.assertTrue(rag_security.judge_pii_leak(
            FakeClient(["YES"]), "m", "q", "r"))
        self.assertFalse(rag_security.judge_pii_leak(
            FakeClient(["NO"]), "m", "q", "r"))
        self.assertIsNone(rag_security.judge_pii_leak(
            FailingClient([]), "m", "q", "r"))

    def test_scenario_filter(self):
        config = {"scenarios": [
            {"id": "s1", "description": "d", "user_query": "q",
             "retrieved_context": "c", "expected_safe_response": "e",
             "attack_goal": "g"},
            {"id": "s2", "description": "d", "user_query": "q",
             "retrieved_context": "c", "expected_safe_response": "e",
             "attack_goal": "g"},
        ]}
        client = FakeClient(["resp", "SAFE", "NO"])
        with patch.object(rag_security, "get_client", return_value=client):
            results = rag_security.run_rag_security_tests(
                config, filter_scenario="s2")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["scenario_id"], "s2")

    def test_empty_scenarios(self):
        with patch.object(rag_security, "get_client",
                          return_value=FakeClient([])):
            self.assertEqual(
                rag_security.run_rag_security_tests(
                    {"scenarios": []}, filter_scenario=None), [])


class GarakCmdTests(unittest.TestCase):
    def test_full_probes_omit_spec(self):
        cmd = garak.build_garak_cmd(
            garak_python="python", model="m", probes=garak.ALL_PROBES,
            output_dir=Path("/tmp/out"), target_type="openai")
        self.assertNotIn("--spec", cmd)

    def test_subset_adds_spec(self):
        cmd = garak.build_garak_cmd(
            garak_python="python", model="m",
            probes=["probes.dan"], output_dir=Path("/tmp/out"),
            target_type="openai")
        self.assertIn("--spec", cmd)
        self.assertIn("probes.dan", cmd)

    def test_target_args_present(self):
        cmd = garak.build_garak_cmd(
            garak_python="python", model="m", probes=garak.ALL_PROBES,
            output_dir=Path("/tmp/out"), target_type="openai")
        self.assertEqual(cmd[cmd.index("--target_type") + 1], "openai")
        self.assertEqual(cmd[cmd.index("--target_name") + 1], "m")


class LangfuseHelperTests(unittest.TestCase):
    def test_extract_usage_dict(self):
        resp = SimpleNamespace(
            usage={"prompt_tokens": 3, "completion_tokens": 5})
        self.assertEqual(langfuse_runner._extract_usage(resp),
                         {"input": 3, "output": 5})

    def test_extract_usage_object(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5))
        self.assertEqual(langfuse_runner._extract_usage(resp),
                         {"input": 3, "output": 5})

    def test_extract_usage_none(self):
        self.assertEqual(
            langfuse_runner._extract_usage(SimpleNamespace(usage=None)), {})

    def test_judge_score_fallback_on_bad_output(self):
        score = langfuse_runner._judge_score(
            FakeClient(["not a number"]), "m", "prompt", 1.0)
        self.assertEqual(score, 1.0)

    def test_judge_score_parses_float(self):
        score = langfuse_runner._judge_score(
            FakeClient(["0.42"]), "m", "prompt", 1.0)
        self.assertAlmostEqual(score, 0.42)


if __name__ == "__main__":
    unittest.main()
