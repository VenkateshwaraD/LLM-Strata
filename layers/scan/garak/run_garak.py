"""run_garak.py - Run the Garak LLM vulnerability scanner.

Cross-platform: Windows and Linux.

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

Garak produces 4 outputs per scan (source: docs.garak.ai):
  1. Screen output  - live scan progress (what's running, what's scheduled)
  2. Report log     - every prompt, every response, every evaluation result
  3. Hit log        - only the cases where garak found a vulnerability
  4. Debug log      - internal operations log for troubleshooting

All file logs are saved to results/garak/ using the --report_prefix flag.
Note: Garak v0.17+ uses --spec instead of the deprecated --probes flag (handled automatically).

Usage:
    python layers/scan/garak/run_garak.py                                    # standard probes (default)
    python layers/scan/garak/run_garak.py --model gpt-4o                     # override OpenAI model name
    python layers/scan/garak/run_garak.py --target_type huggingface --model gpt2
    python layers/scan/garak/run_garak.py --target_type ollama --model llama2
    python layers/scan/garak/run_garak.py --probes quick                     # 3 probes (fast)
    python layers/scan/garak/run_garak.py --probes standard                  # 7 probes (default)
    python layers/scan/garak/run_garak.py --probes redteam                   # 6 aggressive probes
    python layers/scan/garak/run_garak.py --probes all                       # every probe (slow)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml
from core.env_loader import load_dotenv, warn_provider_env
from core.profile_loader import get_model, active_profile_name

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "garak_config.yaml"


# -- Probe presets -------------------------------------------------------------
# Curated subsets for different scan depths
QUICK_PROBES = [
    "probes.dan",               # DAN jailbreaks (fast, high signal)
    "probes.promptinject",      # prompt injection
    "probes.knownbadsignatures",  # known attack signatures
]

STANDARD_PROBES = [
    "probes.dan",
    "probes.promptinject",
    "probes.grandma",
    "probes.leakreplay",
    "probes.knownbadsignatures",
    "probes.continuation",
    "probes.misleading",
]

REDTEAM_PROBES = [
    "probes.atkgen",            # AI-generated adaptive attack prompts
    "probes.encoding",          # encoding evasion (base64, rot13, hex)
    "probes.gcg",               # Greedy Coordinate Gradient adversarial suffix
    "probes.dan",               # DAN jailbreak family
    "probes.grandma",           # social engineering
    "probes.promptinject",      # prompt injection
]

ALL_PROBES = "all"  # runs every garak probe (slow, comprehensive)

PROBE_SETS = {
    "quick": QUICK_PROBES,
    "standard": STANDARD_PROBES,
    "redteam": REDTEAM_PROBES,
    "all": ALL_PROBES,
}


def load_config() -> dict:
    """Load garak_config.yaml, returning an empty dict if it is missing."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file) or {}


def get_garak_python() -> str:
    """Return the interpreter for Garak's isolated environment."""
    configured = os.environ.get("GARAK_PYTHON")
    if configured:
        interpreter = Path(configured)
    elif sys.platform == "win32":
        interpreter = PROJECT_ROOT / ".venv-garak" / "Scripts" / "python.exe"
    else:
        interpreter = PROJECT_ROOT / ".venv-garak" / "bin" / "python"

    if not interpreter.exists():
        logger.error("Garak environment not found.")
        logger.error("Run: python setup.py - or set GARAK_PYTHON to an installed Garak interpreter.")
        sys.exit(1)
    return str(interpreter)


def check_garak() -> None:
    """Verify garak is installed in its isolated environment."""
    garak_python = get_garak_python()
    result = subprocess.run(
        [garak_python, "-m", "garak", "--version"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("garak is not installed in the Garak environment. Run: python setup.py")
        sys.exit(1)
    print(f"  garak: {result.stdout.strip()}")


def build_garak_cmd(
    *,
    garak_python: str,
    model: str,
    probes: list[str] | str,
    output_dir: Path,
    target_type: str,
) -> list[str]:
    """Assemble the garak CLI invocation."""
    cmd = [
        garak_python, "-m", "garak",
        "--target_type", target_type,
        "--target_name", model,
        "--report_prefix", str(output_dir / "scan"),
    ]
    if probes != ALL_PROBES:
        # --spec replaces deprecated --probes (official CLI ref v0.17+)
        cmd += ["--spec", ",".join(probes)]
    return cmd


def run_scan(cmd: list[str], output_dir: Path) -> int:
    """Execute the garak command and return its exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def build_parser(config: dict) -> argparse.ArgumentParser:
    """Build the CLI argument parser with config-derived defaults."""
    parser = argparse.ArgumentParser(description="Garak LLM Security Scanner")
    parser.add_argument("--model", default=get_model("model", config.get("model_name", "gpt-4o-mini")),
                        help="Model name (e.g. gpt-4o-mini, gpt2, llama2)")
    parser.add_argument("--target_type", default=config.get("model_type", "openai"),
                        help="Model provider: openai | huggingface | ollama | rest | bedrock | litellm")
    parser.add_argument("--probes", default="standard", choices=list(PROBE_SETS),
                        help="Probe depth: quick | standard | redteam | all")
    return parser


def resolve_output_dir(config: dict) -> Path:
    """Resolve the results directory (relative paths anchor at repo root)."""
    output_dir = Path(config.get("reporting", {}).get("output_dir", "results/garak"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def print_banner(args: argparse.Namespace) -> None:
    """Print the run banner."""
    print()
    print("=" * 55)
    print("  Garak - LLM Vulnerability Scanner")
    print(f"  Profile : {active_profile_name()}")
    print(f"  Target  : {args.target_type} / {args.model}")
    print(f"  Probes  : {args.probes}")
    print("=" * 55)


def main() -> int:
    """Load config, verify the environment, and run the scan."""
    config = load_config()
    args = build_parser(config).parse_args()

    print_banner(args)
    check_garak()
    warn_provider_env()

    probes = PROBE_SETS[args.probes]
    if probes == ALL_PROBES:
        print("  Running ALL probes (this will take a long time)...")

    output_dir = resolve_output_dir(config)
    cmd = build_garak_cmd(
        garak_python=get_garak_python(),
        model=args.model,
        probes=probes,
        output_dir=output_dir,
        target_type=args.target_type,
    )
    exit_code = run_scan(cmd, output_dir)
    if exit_code != 0:
        print(f"\n  Garak exited with status {exit_code}.")
        return exit_code

    print()
    print("=" * 55)
    print(f"  Scan complete! Results in: {output_dir}/")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
