"""run_agent_security.py - Agentic Security Testing.

Tests multi-turn attack resistance, tool authorization, and agent memory isolation
without executing real production tools.

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

WHAT IS AGENTIC SECURITY TESTING?
  Agentic systems can preserve state across turns, call tools on a user's behalf,
  and write information to memory. These capabilities create security risks that
  are not covered by single-turn prompt or model safety tests.

TEST AREAS:
  multi_turn_attacks       - Detect unsafe behavior across scripted conversations
  tool_authorization_tests - Verify deny-by-default role and tool policies
  memory_tests             - Verify namespace isolation and trusted-write rules

Usage:
    python layers/attack/agent_security/run_agent_security.py
    python layers/attack/agent_security/run_agent_security.py --multi-turn-only
    python layers/attack/agent_security/run_agent_security.py --tool-auth-only
    python layers/attack/agent_security/run_agent_security.py --memory-only
    python layers/attack/agent_security/run_agent_security.py --dry-run
    python layers/attack/agent_security/run_agent_security.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: core.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # attack dir: agent_security.*

import yaml
from core.client_factory import get_client
from core.env_loader import load_dotenv, require_provider_env
from core.profile_loader import active_profile_name, get_model

from agent_security.core import (
    run_memory_tests,
    run_multiturn_attacks,
    run_tool_authorization_tests,
    summarize_sections,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "agent_security_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results/agent_security"


# -- Configuration and report output -------------------------------------------------

def load_config() -> dict:
    """Load the agent security test configuration from YAML."""
    with open(CONFIG_PATH, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def save_report(report: dict) -> Path:
    """Write a timestamped JSON report and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"agent_security_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
    return report_path


# -- CLI ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Agentic LLM Security Tester")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--multi-turn-only", action="store_true",
                       help="Run multi-turn attack tests only")
    group.add_argument("--tool-auth-only", action="store_true",
                       help="Run tool authorization tests only")
    group.add_argument("--memory-only", action="store_true",
                       help="Run memory isolation tests only")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configured tests without calling an LLM")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the result of every test")
    return parser


def print_dry_run(config: dict) -> None:
    """List the configured tests without calling an LLM."""
    print(f"Profile: {active_profile_name()}")
    print(f"Multi-turn attacks: {len(config.get('multi_turn_attacks', []))}")
    print(f"Tool authorization tests: {len(config.get('tool_authorization_tests', []))}")
    print(f"Memory tests: {len(config.get('memory_tests', []))}")


def run_sections(args: argparse.Namespace, config: dict, model: str) -> dict[str, list[dict]]:
    """Run the selected test sections and return their results keyed by section name."""
    run_multi = not (args.tool_auth_only or args.memory_only)
    run_auth = not (args.multi_turn_only or args.memory_only)
    run_memory = not (args.multi_turn_only or args.tool_auth_only)

    sections: dict[str, list[dict]] = {}
    if run_multi:
        require_provider_env()
        sections["multi_turn"] = run_multiturn_attacks(
            client=get_client(),
            model=model,
            cases=config.get("multi_turn_attacks", []),
            max_tokens=int(config.get("max_tokens", 400)),
        )
    if run_auth:
        sections["tool_authorization"] = run_tool_authorization_tests(config)
    if run_memory:
        sections["memory"] = run_memory_tests(config)
    return sections


def print_verbose(sections: dict[str, list[dict]]) -> None:
    """Print every test result with its pass/fail status."""
    for section, results in sections.items():
        for result in results:
            status = "PASS" if result.get("passed") else "FAIL"
            print(f"  [{status}] {section}/{result['id']}")
            if result.get("error"):
                print(f"    {result['error']}")


def main() -> int:
    """Run the selected agent security tests and save a JSON report."""
    args = build_parser().parse_args()
    config = load_config()

    if args.dry_run:
        print_dry_run(config)
        return 0

    model = get_model("model", config.get("model", "gpt-4o-mini"))
    sections = run_sections(args, config, model)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": active_profile_name(),
        "model": model,
        "summary": summarize_sections(sections),
        "results": sections,
    }
    report_path = save_report(report)

    print(f"Profile: {report['profile']}")
    print(f"Agentic security tests: {report['summary']['passed']}/{report['summary']['total']} passed")
    print(f"Report: {report_path}")
    if args.verbose:
        print_verbose(sections)

    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
