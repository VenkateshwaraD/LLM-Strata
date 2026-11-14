"""run_deepeval.py - Run DeepEval LLM safety testing.

100% Python, no Node.js required.

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

DeepEval tests whether your LLM produces UNSAFE or HARMFUL content.
It complements Garak (which tests attack surface) by testing output safety.

Metrics:
    bias          - gender, racial, political bias
    toxicity      - hate speech, abusive language
    hallucination - fabricated / unsupported facts
    misuse        - jailbreaks, policy violations
    pii           - PII / privacy leakage (SSN, card numbers, passwords)

Requirements:
    pip install deepeval
    Credentials for the active provider must be set

Usage:
    python layers/scan/deepeval/run_deepeval.py                 # full safety scan
    python layers/scan/deepeval/run_deepeval.py --metric bias   # single metric only
    python layers/scan/deepeval/run_deepeval.py --metric pii    # PII leakage only
    python layers/scan/deepeval/run_deepeval.py --verbose       # show per-test details
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml
from core.env_loader import load_dotenv, require_provider_env
from core.profile_loader import get_model, active_profile_name
from core.client_factory import get_client

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "deepeval_config.yaml"
DEFAULT_OUTPUT_DIR = "results/deepeval"
DEFAULT_MODEL = "gpt-4o-mini"
METRIC_NAMES = ("bias", "toxicity", "hallucination", "misuse", "pii")


def load_config() -> dict:
    """Load deepeval_config.yaml."""
    with open(CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file) or {}


def check_deepeval() -> None:
    """Fail fast if the deepeval package is not installed."""
    try:
        import deepeval  # noqa: F401
    except ImportError:
        logger.error("deepeval not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


def create_deepeval_model(model_name: str):
    """Adapt the active provider client to DeepEval's custom model interface."""
    from deepeval.models.base_model import DeepEvalBaseLLM

    class ProviderModel(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = model_name
            self.name = model_name   # required by DeepEvalBaseLLM; super().__init__() is skipped
            self.client = get_client()
            self.model = self.load_model()

        def load_model(self):
            return self.client

        def _generate_text(self, prompt: str) -> str:
            response = self.load_model().chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        def generate(self, prompt: str, schema=None):
            text = self._generate_text(prompt)
            if schema is None:
                return text
            data = _parse_structured_output(text)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            return schema.parse_obj(data)

        async def a_generate(self, prompt: str, schema=None):
            return await asyncio.to_thread(self.generate, prompt, schema)

        def get_model_name(self):
            return self.model_name

    return ProviderModel()


def _parse_structured_output(text: str):
    """Parse JSON from evaluator output, tolerating code fences and prose."""
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start, end = payload.find("{"), payload.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Evaluator returned invalid structured output.") from None
        try:
            return json.loads(payload[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Evaluator returned invalid structured output.") from exc


def _resolve_evaluator(evaluator_model, config: dict | None = None):
    """Return evaluator_model unchanged if set; otherwise build one from the active profile.

    Ensures that runner functions called directly (not via run_all_tests) always
    use the profile-resolved evaluator model rather than falling through to
    DeepEval's internal default.
    """
    if evaluator_model is not None:
        return evaluator_model
    if config is None:
        config = load_config()
    return create_deepeval_model(
        get_model("evaluator_model", config.get("evaluator_model", DEFAULT_MODEL))
    )


def _make_metric(metric_name: str, evaluator_model, threshold: float):
    """Instantiate the DeepEval metric for a category name."""
    from deepeval.metrics import (
        BiasMetric, HallucinationMetric, MisuseMetric, PIILeakageMetric, ToxicityMetric,
    )
    builders = {
        "bias": lambda: BiasMetric(threshold=threshold, model=evaluator_model),
        "toxicity": lambda: ToxicityMetric(threshold=threshold, model=evaluator_model),
        "hallucination": lambda: HallucinationMetric(threshold=threshold, model=evaluator_model),
        "misuse": lambda: MisuseMetric(domain="general AI assistant", threshold=threshold,
                                       model=evaluator_model),
        "pii": lambda: PIILeakageMetric(threshold=threshold, model=evaluator_model),
    }
    if metric_name not in builders:
        raise ValueError(f"Unknown metric '{metric_name}'.")
    return builders[metric_name]()


def run_metric_test(
    metric_name: str,
    *,
    model_name: str,
    test_input: str,
    verbose: bool,
    evaluator_model=None,
    threshold: float = 0.5,
) -> dict:
    """Run one DeepEval metric against the model's response to test_input."""
    from deepeval.test_case import LLMTestCase

    evaluator_model = _resolve_evaluator(evaluator_model)
    client = get_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": test_input}],
    )
    actual_output = response.choices[0].message.content

    context = {"context": [test_input]} if metric_name == "hallucination" else {}
    test_case = LLMTestCase(input=test_input, actual_output=actual_output, **context)
    metric = _make_metric(metric_name, evaluator_model, threshold)
    metric.measure(test_case)

    result = {
        "metric": metric_name,
        "input": test_input,
        "score": metric.score,
        "passed": metric.success,  # set by metric.is_successful() - authoritative
        "reason": metric.reason,
    }
    if verbose:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"    [{status}] {metric_name} score={metric.score:.2f} | {test_input[:60]}...")
    return result


def _failure_result(category: str, test_input: str, exc: Exception) -> dict:
    return {
        "metric": category,
        "input": test_input,
        "score": None,
        "passed": False,
        "reason": str(exc),
    }


def run_all_tests(config: dict, filter_metric: str | None, verbose: bool) -> list[dict]:
    """Run every configured test case (optionally one metric) and collect results."""
    model_name = get_model("model", config.get("model", DEFAULT_MODEL))
    evaluator = _resolve_evaluator(None, config)
    thresholds = {
        metric["name"]: metric.get("threshold", 0.5)
        for metric in config.get("metrics", [])
        if isinstance(metric, dict) and metric.get("name")
    }
    results = []
    for test_case in config.get("test_cases", []):
        category, test_input = test_case.get("category"), test_case.get("input")
        if filter_metric and category != filter_metric:
            continue
        if category not in METRIC_NAMES:
            print(f"    SKIP: unknown category '{category}'")
            continue
        try:
            results.append(run_metric_test(
                category, model_name=model_name, test_input=test_input,
                verbose=verbose, evaluator_model=evaluator,
                threshold=thresholds.get(category, 0.5),
            ))
        except Exception as exc:  # per-case boundary: record failure, continue the scan
            logger.error("metric %s failed on input %.50s: %s", category, test_input, exc)
            results.append(_failure_result(category, test_input, exc))
    return results


def print_summary(results: list[dict]) -> None:
    """Print pass/fail totals and details for failed tests."""
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    print()
    print("  Safety Test Summary:")
    print(f"    Total  : {total}")
    print(f"    Passed : {passed}  (model refused / responded safely)")
    print(f"    Failed : {total - passed}  (model produced unsafe content)")

    for result in results:
        if result["passed"]:
            continue
        score_str = f"{result['score']:.2f}" if result["score"] is not None else "N/A"
        print(f"    [{result['metric'].upper()}] score={score_str}")
        print(f"      Input  : {result['input'][:80]}")
        print(f"      Reason : {result['reason']}")


def save_results(results: list[dict], output_dir: Path) -> Path:
    """Write a timestamped JSON report and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"safety_report_{timestamp}.json"
    report = {
        "timestamp": timestamp,
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "results": results,
    }
    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"\n  Report saved: {report_path}")
    return report_path


def resolve_output_dir(config: dict) -> Path:
    """Resolve the results directory (relative paths anchor at repo root)."""
    output_dir = Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def print_banner(config: dict, args: argparse.Namespace) -> None:
    """Print the run banner."""
    model_name = get_model("model", config.get("model", DEFAULT_MODEL))
    evaluator_model = get_model("evaluator_model", config.get("evaluator_model", DEFAULT_MODEL))
    print()
    print("=" * 55)
    print("  DeepEval - LLM Safety Testing")
    print("=" * 55)
    print(f"  Profile         : {active_profile_name()}")
    print(f"  Model under test: {model_name}")
    print(f"  Evaluator model : {evaluator_model}")
    print(f"  Config          : {CONFIG_PATH}")
    print(f"  Metrics         : {args.metric or 'all'}")
    print()
    print("  Running safety tests...")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="DeepEval LLM Safety Tester")
    parser.add_argument("--metric", default=None, choices=list(METRIC_NAMES),
                        help="Run a single metric only")
    parser.add_argument("--verbose", action="store_true", help="Show per-test details")
    return parser


def main() -> None:
    """Run the DeepEval safety scan."""
    args = build_parser().parse_args()
    config = load_config()
    print_banner(config, args)
    require_provider_env()
    check_deepeval()

    results = run_all_tests(config, filter_metric=args.metric, verbose=args.verbose)
    print_summary(results)
    output_dir = resolve_output_dir(config)
    save_results(results, output_dir)

    print()
    print("=" * 55)
    print(f"  Safety scan complete! Results in: {output_dir}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
