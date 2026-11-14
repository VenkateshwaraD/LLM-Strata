"""run_human_redteam.py - Structured human red team session.

Interactive CLI for human testers to probe the LLM with novel attacks.

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

This tool:
  - Guides the human tester through attack categories
  - Sends prompts to the real model and shows the response
  - Lets the tester score results (0-safe, 1-borderline, 2-breached)
  - Logs everything to results/human_redteam/ in the same format as other tools

Why human red teaming:
  Automated tools (Garak, PyRIT) test known patterns.
  Humans find novel, creative, context-aware attacks no tool has seen.

Requirements:
  The profile's provider credentials must be set (e.g. OPENAI_API_KEY)

Usage:
    python layers/attack/human_redteam/run_human_redteam.py               # start a session
    python layers/attack/human_redteam/run_human_redteam.py --report       # view past reports
    python layers/attack/human_redteam/run_human_redteam.py --category novel_jailbreak
"""

import argparse
import json
import logging
import os
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
CONFIG_PATH = Path(__file__).resolve().parent / "human_redteam_config.yaml"

_SCORE_STATUS = {0: "HELD", 1: "BORDERLINE", 2: "BREACHED"}


def load_config() -> dict:
    """Load human_redteam_config.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def send_to_model(model: str, prompt: str) -> str:
    """Send a prompt to the target model and return the response."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def print_categories(categories: list[dict]) -> None:
    """List the available attack categories."""
    print()
    print(" Attack categories:")
    for category in categories:
        print(f"   [{category['id']}] {category['name']}")
        print(f"       {category['description']}")
    print()


def print_instructions(categories: list[dict], model: str) -> None:
    """Print the session banner and tester instructions."""
    print(f"\n Profile           : {active_profile_name()}")
    print(f" Target model      : {model}")
    print(f" Categories        : {len(categories)}")
    print()
    print(" Instructions:")
    print("   - For each category, enter your attack prompt")
    print("   - The model response will be shown")
    print("   - Score the result: 0-safe, 1-borderline, 2-breached")
    print("   - Type 'skip' to skip a category, 'quit' to end session")
    print()


def print_category_header(category: dict) -> None:
    """Print one category's goal and example approaches."""
    print("-" * 55)
    print(f" Category : {category['name']}")
    print(f" Goal     : {category['description']}")
    print(" Example approaches:")
    for example in category.get("examples", []):
        print(f"   {example}")
    print()


def prompt_for_score(scoring: dict) -> int:
    """Ask the tester to score the response until a valid score is given."""
    print()
    print(" Score this response:")
    for score, label in scoring.items():
        print(f"   {score} = {label}")
    while True:
        score_input = input(" Your score (0/1/2): ").strip()
        if score_input in ("0", "1", "2"):
            return int(score_input)
        print(" Invalid. Enter 0, 1, or 2.")


def record_attempt(category: dict, prompt: str, model: str,
                   scoring: dict, results: list[dict]) -> bool:
    """Send one prompt, let the tester score it, and append the result.

    Returns:
        False if the model call failed (skip remaining attempts in category).
    """
    print()
    print(" Sending to model...")
    try:
        response = send_to_model(model, prompt)
    except Exception as exc:  # interactive boundary: report and let the tester move on
        print(f"  ERROR: {exc}")
        logger.error("model call failed in category '%s': %s", category["id"], exc)
        return False

    print()
    print("  --- Model Response ---------------------------------")
    print(f" {response}")
    print("  -----------------------------------------------")

    score = prompt_for_score(scoring)
    notes = input(" Notes (optional, press Enter to skip): ").strip()
    results.append({
        "category": category["id"],
        "category_name": category["name"],
        "prompt": prompt,
        "response": response,
        "score": score,
        "score_label": scoring.get(score, ""),
        "notes": notes,
        "breached": score == 2,
    })
    print(f"\n Logged [{_SCORE_STATUS[score]}]\n")
    another = input(" Try another prompt for this category? (y/n): ").strip().lower()
    return another == "y"


def run_category(category: dict, model: str, scoring: dict,
                 results: list[dict]) -> str:
    """Run the prompt loop for one category.

    Returns:
        'quit' | 'done' — whether the tester ended the whole session.
    """
    print_category_header(category)
    while True:
        prompt = input(" Your attack prompt (or 'skip'/'quit'): ").strip()
        if prompt.lower() == "quit":
            return "quit"
        if prompt.lower() == "skip":
            print("  [SKIPPED]\n")
            return "done"
        if not prompt:
            continue
        if not record_attempt(category, prompt, model, scoring, results):
            return "done"


def run_session(config: dict, filter_category: str | None) -> list[dict]:
    """Run an interactive human red team session."""
    model = get_model("model", config.get("model", "gpt-4o-mini"))
    scoring = config.get("scoring", {})
    categories = [category for category in config.get("attack_categories", [])
                  if not filter_category or category["id"] == filter_category]

    print_instructions(categories, model)
    results: list[dict] = []
    for category in categories:
        if run_category(category, model, scoring, results) == "quit":
            print("\n Session ended by user.")
            break
    return results


def print_summary(results: list[dict]) -> None:
    """Print held/borderline/breached totals and confirmed vulnerabilities."""
    breached = sum(1 for result in results if result["breached"])
    border = sum(1 for result in results if result["score"] == 1)
    held = sum(1 for result in results if result["score"] == 0)

    print()
    print("-" * 55)
    print(" Human Red Team Session Summary")
    print("-" * 55)
    print(f" Total attempts : {len(results)}")
    print(f" Model held     : {held}    (score=0, safe)")
    print(f" Borderline     : {border}    (score=1, review needed)")
    print(f" Breached       : {breached}    (score=2, vulnerability confirmed)")

    if breached > 0:
        print()
        print(" Confirmed vulnerabilities:")
        for result in results:
            if result["breached"]:
                print(f"   [{result['category_name']}]")
                print(f"       Prompt   : {result['prompt'][:80]}")
                print(f"       Response : {result['response'][:80]}")
                if result["notes"]:
                    print(f"       Notes    : {result['notes']}")


def save_results(results: list[dict], output_dir: Path) -> Path:
    """Write a timestamped JSON session report and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"human_redteam_{timestamp}.json"
    report = {
        "timestamp": timestamp,
        "tester": os.environ.get("USERNAME", "unknown"),
        "total": len(results),
        "breached": sum(1 for result in results if result["breached"]),
        "borderline": sum(1 for result in results if result["score"] == 1),
        "held": sum(1 for result in results if result["score"] == 0),
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"\n Report saved: {report_path}")
    return report_path


def show_reports(output_dir: Path) -> None:
    """Show a summary of all past human red team sessions."""
    reports = sorted(output_dir.glob("human_redteam_*.json"), reverse=True)
    if not reports:
        print(f"  No reports found in {output_dir}/")
        print("  Run a session first: python layers/attack/human_redteam/run_human_redteam.py")
        return

    print(f"\n Past sessions ({len(reports)} found):\n")
    for report_path in reports:
        try:
            with open(report_path, encoding="utf-8") as report_file:
                data = json.load(report_file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable report %s: %s", report_path.name, exc)
            continue
        print(f"  {data['timestamp']}  |  breached={data['breached']}"
              f"  borderline={data['borderline']}  held={data['held']}  |  {report_path.name}")


def resolve_output_dir(config: dict) -> Path:
    """Resolve the results directory (relative paths anchor at repo root)."""
    output_dir = Path(config.get("output_dir", "results/human_redteam"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Human Red Team Session Runner")
    parser.add_argument("--report", action="store_true",
                        help="Show past session reports")
    parser.add_argument("--category", default=None,
                        help="Focus on a single attack category ID")
    return parser


def check_category(config: dict, category_id: str) -> None:
    """Validate that --category names a configured attack category."""
    valid_ids = [category["id"] for category in config.get("attack_categories", [])]
    if category_id not in valid_ids:
        print(f"  ERROR: Unknown category '{category_id}'")
        print(f"  Valid: {', '.join(valid_ids)}")
        raise SystemExit(1)


def print_footer() -> None:
    """Print the session-complete footer."""
    print()
    print("-" * 55)
    print(" Session complete.")
    print("  View reports: python layers/attack/human_redteam/run_human_redteam.py --report")
    print("-" * 55)


def main() -> None:
    """Run an interactive session or list past reports."""
    args = build_parser().parse_args()

    print()
    print("-" * 55)
    print(" Human Red Team - Manual Attack Session")
    print("-" * 55)

    config = load_config()
    output_dir = resolve_output_dir(config)
    if args.report:
        show_reports(output_dir)
        return

    require_provider_env()
    if args.category:
        check_category(config, args.category)

    print_categories(config.get("attack_categories", []))
    results = run_session(config, filter_category=args.category)
    if results:
        print_summary(results)
        save_results(results, output_dir)
    else:
        print("\n No attempts recorded.")
    print_footer()


if __name__ == "__main__":
    main()
