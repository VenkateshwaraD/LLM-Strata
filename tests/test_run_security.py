"""Tests for run_security.py: step table, argparse, dry-run, CI exit codes."""

import argparse
import sys
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT))
import run_security  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    """Build a namespace with every step flag defaulting to False."""
    values = {"garak_depth": "standard", "dry_run": True, "ci": False,
              "human_redteam": False}
    for step in run_security.STEPS:
        values[step.only_flag] = False
        if step.skip_flag:
            values[step.skip_flag] = False
    values.update(overrides)
    return argparse.Namespace(**values)


class StepTableTests(unittest.TestCase):
    def test_all_scripts_exist(self):
        missing = [step.script for step in run_security.STEPS
                   if not (run_security.REPO_ROOT / step.script).exists()]
        self.assertEqual(missing, [])

    def test_step_names_unique(self):
        names = [step.name for step in run_security.STEPS]
        self.assertEqual(len(names), len(set(names)))

    def test_only_flags_unique(self):
        flags = [step.only_flag for step in run_security.STEPS]
        self.assertEqual(len(flags), len(set(flags)))

    def test_step_count_matches_layers(self):
        self.assertEqual(len(run_security.STEPS), 13)  # 14 layers minus interactive human_redteam


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = run_security.build_parser()

    def test_garak_depth_choices(self):
        for depth in ("quick", "standard", "redteam", "all"):
            args = self.parser.parse_args(["--garak-depth", depth])
            self.assertEqual(args.garak_depth, depth)

    def test_invalid_garak_depth_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--garak-depth", "reateam"])

    def test_in_context_aliases(self):
        self.assertTrue(self.parser.parse_args(
            ["--in-context-only"]).in_context_only)
        self.assertTrue(self.parser.parse_args(
            ["--many-shot-only"]).in_context_only)

    def test_skip_many_shot_alias(self):
        args = self.parser.parse_args(["--skip-many-shot"])
        self.assertTrue(args.skip_in_context)

    def test_only_flags_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--garak-only", "--deepeval-only"])

    def test_every_only_flag_registered(self):
        for step in run_security.STEPS:
            flag = f"--{step.only_flag.replace('_', '-')}"
            args = self.parser.parse_args([flag])
            self.assertTrue(getattr(args, step.only_flag), flag)


class StepCommandTests(unittest.TestCase):
    def test_garak_gets_probes_flag(self):
        step = next(step for step in run_security.STEPS if step.name == "garak")
        cmd = run_security.step_command(step, _args(garak_depth="redteam"))
        self.assertIn("--probes", cmd)
        self.assertIn("redteam", cmd)

    def test_other_steps_plain_script(self):
        step = next(step for step in run_security.STEPS if step.name == "deepeval")
        cmd = run_security.step_command(step, _args())
        self.assertEqual(cmd, [sys.executable, step.script])


class RunTests(unittest.TestCase):
    def setUp(self):
        self._dry = run_security.DRY_RUN

    def tearDown(self):
        run_security.DRY_RUN = self._dry

    def test_dry_run_never_calls_subprocess(self):
        run_security.DRY_RUN = True
        with patch("run_security.subprocess.run") as mock_run:
            code = run_security.run(["echo", "hi"])
        self.assertEqual(code, 0)
        mock_run.assert_not_called()

    def test_real_run_delegates_to_subprocess(self):
        run_security.DRY_RUN = False
        with patch("run_security.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 3
            code = run_security.run(["echo", "hi"])
        self.assertEqual(code, 3)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.kwargs["cwd"], run_security.REPO_ROOT)


class RunStepTests(unittest.TestCase):
    def test_only_mode_filters_unselected(self):
        codes: list[int] = []
        args = _args(garak_only=True)
        for step in run_security.STEPS:
            if step.name == "garak":
                continue
            with patch("run_security.run", return_value=0) as mock_run:
                run_security.run_step(step, args, True, codes)
            mock_run.assert_not_called()
        self.assertEqual(codes, [])

    def test_skip_flag_skips(self):
        codes: list[int] = []
        step = next(step for step in run_security.STEPS if step.name == "ragas")
        with patch("run_security.run", return_value=0) as mock_run:
            run_security.run_step(step, _args(skip_ragas=True), False, codes)
        mock_run.assert_not_called()
        self.assertEqual(codes, [])

    def test_selected_step_records_exit_code(self):
        codes: list[int] = []
        step = next(step for step in run_security.STEPS if step.name == "deepeval")
        with patch("run_security.run", return_value=7):
            run_security.run_step(step, _args(deepeval_only=True), True, codes)
        self.assertEqual(codes, [7])


class CiStatusTests(unittest.TestCase):
    def test_ci_failure_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            run_security.report_ci_status(_args(ci=True), [0, 2])
        self.assertEqual(ctx.exception.code, 1)

    def test_ci_all_pass_ok(self):
        run_security.report_ci_status(_args(ci=True), [0, 0])

    def test_non_ci_failure_ignored(self):
        run_security.report_ci_status(_args(ci=False), [1, 2])


class MainDryRunTests(unittest.TestCase):
    def setUp(self):
        self._dry = run_security.DRY_RUN

    def tearDown(self):
        run_security.DRY_RUN = self._dry

    def test_main_dry_run_no_subprocess(self):
        with patch.object(sys, "argv",
                          ["run_security.py", "--garak-only", "--dry-run"]):
            with patch("run_security.subprocess.run") as mock_run:
                run_security.main()
        mock_run.assert_not_called()

    def test_main_in_context_only_runs_one_step(self):
        with patch.object(sys, "argv",
                          ["run_security.py", "--in-context-only", "--dry-run"]):
            with patch("run_security.run", return_value=0) as mock_run:
                run_security.main()
        self.assertEqual(mock_run.call_count, 1)
        cmd = mock_run.call_args.args[0]
        self.assertIn("run_many_shot.py", cmd[1])


if __name__ == "__main__":
    unittest.main()
