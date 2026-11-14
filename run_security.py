"""run_security.py - Run the full LLM security + safety pipeline.

Layers: Garak + DeepEval + RAGAS + PyRIT + LLM Guard (scan)
        Encoded + In-Context + Multilingual + Tool-Inject + Backdoor
        + RAG Security + Agent Security + Human Red Team (attack)
        LangFuse (monitor)
Cross-platform: Windows and Linux.

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

Usage:
    python run_security.py                       # run all automated tools
    python run_security.py --garak-only          # Garak vulnerability scan only
    python run_security.py --deepeval-only       # DeepEval safety scan only
    python run_security.py --ragas-only          # RAGAS RAG safety scan only
    python run_security.py --pyrit-only          # PyRIT red team only
    python run_security.py --guard-only          # LLM Guard runtime demo only
    python run_security.py --langfuse-only       # LangFuse monitoring demo only
    python run_security.py --human-redteam       # launch human red team session
    python run_security.py --encoded-only        # encoded/obfuscated attack scan only
    python run_security.py --in-context-only     # in-context attack jailbreak scan only
    python run_security.py --multilingual-only   # multilingual attack scan only
    python run_security.py --tool-inject-only    # agentic tool-call injection only
    python run_security.py --backdoor-only       # backdoor trigger scan only
    python run_security.py --rag-security-only   # RAG/vector embedding security only
    python run_security.py --agent-security-only # agentic security tests only
    python run_security.py --garak-depth redteam  # red team Garak scan
    python run_security.py --garak-depth all      # deep Garak scan (slow)
    python run_security.py --skip-pyrit          # skip PyRIT (slow, agentic)
    python run_security.py --skip-ragas          # skip RAGAS (only if no RAG app)
    python run_security.py --skip-langfuse       # skip LangFuse (if keys not set)
    python run_security.py --skip-encoded        # skip encoded attack scan
    python run_security.py --skip-in-context     # skip in-context attack jailbreak scan
    python run_security.py --skip-multilingual   # skip multilingual attack scan
    python run_security.py --skip-tool-inject    # skip tool-call injection scan
    python run_security.py --skip-backdoor       # skip backdoor trigger scan
    python run_security.py --skip-rag-security   # skip RAG/vector security scan
    python run_security.py --dry-run             # print all steps without executing
    python run_security.py --ci                  # CI mode: skip interactive tools, exit 1 on any failure
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.env_loader import load_dotenv, warn_provider_env

load_dotenv()

DRY_RUN = False
REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    """One pipeline step: an --*-only flag, an optional --skip-* flag, and a runner."""

    name: str
    only_flag: str | None
    skip_flag: str | None
    title: str
    subtitle: str
    script: str
    results_dir: str


STEPS: tuple[Step, ...] = (
    Step("garak", "garak_only", None,
         "Garak: Vulnerability Scan + Red Teaming (pre-deployment)",
         "Probing your LLM for vulnerabilities with adversarial inputs...",
         "layers/scan/garak/run_garak.py", "results/garak/"),
    Step("deepeval", "deepeval_only", None,
         "DeepEval: Safety Testing (bias + toxicity + hallucination + PII)",
         "Testing for harmful output, bias, hallucination, PII leakage, prompt injection...",
         "layers/scan/deepeval/run_deepeval.py", "results/deepeval/"),
    Step("ragas", "ragas_only", "skip_ragas",
         "RAGAS: RAG Pipeline Safety (context poisoning + faithfulness)",
         "Testing for poisoned context, hallucination, retrieval quality...",
         "layers/scan/ragas/run_ragas.py", "results/ragas/"),
    Step("pyrit", "pyrit_only", "skip_pyrit",
         "PyRIT: Multi-turn Agentic Red Teaming",
         "Running goal-directed adaptive attacks across multiple turns...",
         "layers/scan/pyrit/run_pyrit.py", "results/pyrit/"),
    Step("guard", "guard_only", None,
         "LLM Guard: Runtime Protection Demo",
         "Testing input/output scanners against sample attack prompts...",
         "layers/scan/llm_guard/run_guard.py", "results/llm_guard/"),
    Step("langfuse", "langfuse_only", "skip_langfuse",
         "LangFuse: Production Monitoring Demo",
         "Sending traced requests + scoring for safety in dashboard...",
         "layers/monitor/langfuse/run_langfuse.py", "results/langfuse/"),
    Step("encoded", "encoded_only", "skip_encoded",
         "Encoded Attacks: Bypass & Obfuscation Testing",
         "Testing base64, rot13, hex, morse, leetspeak, homoglyphs and more...",
         "layers/attack/encoded_attacks/run_encoded_attacks.py", "results/encoded_attacks/"),
    Step("in-context", "in_context_only", "skip_in_context",
         "Many-Shot Jailbreaks: In-Context Conditioning Attacks",
         "Testing pure-harmful, gradual-escalation, and camouflaged strategies...",
         "layers/attack/in_context_attacks/run_many_shot.py", "results/in_context_attacks/"),
    Step("multilingual", "multilingual_only", "skip_multilingual",
         "Multilingual Attacks: Cross-Language Safety Bypass",
         "Testing Arabic, Chinese, Russian, Hindi, Spanish, French, German...",
         "layers/attack/multilingual/run_multilingual.py", "results/multilingual/"),
    Step("tool-inject", "tool_inject_only", "skip_tool_inject",
         "Tool-Call Injection: Agentic Indirect Prompt Injection",
         "Injecting malicious instructions via search, file, email, database results...",
         "layers/attack/tool_inject/run_tool_inject.py", "results/tool_inject/"),
    Step("backdoor", "backdoor_only", "skip_backdoor",
         "Backdoor Triggers: Trojan Phrase Amplification Testing",
         "Measuring compliance rate with / without DAN, authority, roleplay triggers...",
         "layers/attack/backdoor/run_backdoor.py", "results/backdoor/"),
    Step("rag-security", "rag_security_only", "skip_rag_security",
         "RAG Security: Vector / Embedding Weakness Testing (LLM08)",
         "Testing 9 attack vectors: injection, PII leak, embedding authority override, flooding...",
         "layers/attack/rag_security/run_rag_security.py", "results/rag_security/"),
    Step("agent-security", "agent_security_only", "skip_agent_security",
         "Agentic Security: Multi-turn, Tool Authorization, Memory",
         "Testing multi-turn attacks, tool permissions, and agent memory isolation...",
         "layers/attack/agent_security/run_agent_security.py", "results/agent_security/"),
)


def run(cmd: list[str]) -> int:
    """Run a command from the repository root, honoring --dry-run."""
    print(f"\n  > {' '.join(cmd)}\n")
    if DRY_RUN:
        print("  [DRY RUN] - command not executed")
        return 0
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="LLM Strata - Security & Safety Pipeline Runner")
    group = parser.add_mutually_exclusive_group()
    for step in STEPS:
        flag = step.only_flag.replace("_", "-")
        if step.name == "in-context":
            group.add_argument(f"--{flag}", "--many-shot-only", dest=step.only_flag,
                               action="store_true", help=f"Run {step.name} scan only")
        else:
            group.add_argument(f"--{flag}", dest=step.only_flag, action="store_true",
                               help=f"Run {step.name} only")
    group.add_argument("--human-redteam", action="store_true", help="Launch human red team session")
    parser.add_argument("--garak-depth", default="standard",
                        choices=["quick", "standard", "redteam", "all"],
                        help="Garak probe depth (default: standard)")
    for step in STEPS:
        if step.skip_flag:
            flag = step.skip_flag.replace("_", "-")
            if step.name == "in-context":
                parser.add_argument(f"--{flag}", "--skip-many-shot", dest=step.skip_flag,
                                    action="store_true", help=f"Skip {step.name} scan")
            else:
                parser.add_argument(f"--{flag}", dest=step.skip_flag, action="store_true",
                                    help=f"Skip {step.name} scan")
    parser.add_argument("--dry-run", action="store_true", help="Print all steps without executing")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: skip interactive tools, exit 1 on any failure")
    return parser


def step_command(step: Step, args: argparse.Namespace) -> list[str]:
    """Build the subprocess command for a step."""
    cmd = [sys.executable, step.script]
    if step.name == "garak":
        cmd += ["--probes", args.garak_depth]
    return cmd


def run_step(step: Step, args: argparse.Namespace, only_one: bool, exit_codes: list[int]):
    """Run or skip a single step and record its exit code."""
    only_selected = getattr(args, step.only_flag)
    skipped = step.skip_flag and getattr(args, step.skip_flag)
    if only_one and not only_selected:
        return
    if not only_selected and skipped:
        print(f"\n  [SKIPPED] {step.name} - use --{step.only_flag.replace('_', '-')} to run separately")
        return
    print_header(f"STEP - {step.title}")
    print(f"  {step.subtitle}")
    exit_codes.append(run(step_command(step, args)))
    print(f"\n  Results saved to: {step.results_dir}")


def run_human_redteam(args: argparse.Namespace, only_one: bool):
    """Run the interactive human red team session (never automatic, never in CI)."""
    if args.human_redteam and not args.ci:
        print_header("STEP - Human Red Team: Manual Attack Session")
        print("  Interactive session - a human tester probes the model...")
        run([sys.executable, "layers/attack/human_redteam/run_human_redteam.py"])
        print("\n  Human red team results saved to: results/human_redteam/")
    elif not only_one:
        print("\n  [INFO] Human red team not run automatically (interactive).")
        print("     Run manually: python run_security.py --human-redteam")


def print_summary(exit_codes: list[int]):
    """Print the results directory map and next steps."""
    print_header("Security + Safety Run Complete")
    print("  Results:")
    for step in STEPS:
        print(f"  {step.results_dir:<26} - {step.title.split(':')[0]}")
    print("  results/human_redteam/    - human red team session logs")
    print("  cloud.langfuse.com        - live production trace dashboard")
    print()
    print("  Next steps:")
    print("  1. Review reports in results/ for security vulnerabilities and safety gaps")
    print("  2. Integrate LLM Guard into your app for continuous monitoring")
    print("  3. Instrument your app with LangFuse SDK for continuous monitoring")
    print("  4. Run human red team: python run_security.py --human-redteam")
    print("  5. Re-run after fixes to verify they're resolved")
    print("=" * 55)


def report_ci_status(args: argparse.Namespace, exit_codes: list[int]):
    """Exit non-zero in CI mode when any step failed."""
    if args.ci and any(code != 0 for code in exit_codes):
        failed = sum(1 for code in exit_codes if code != 0)
        print(f"\n  CI: {failed} tool(s) exited with errors - pipeline marked as failed.")
        sys.exit(1)


def main():
    """Parse arguments and run the pipeline."""
    args = build_parser().parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run

    print_header("LLM Strata - Security & Safety Pipeline")
    print(f"  Platform : {sys.platform}")
    if DRY_RUN:
        print("  Mode     : DRY RUN - no commands will be executed")
    warn_provider_env()

    only_one = any(getattr(args, step.only_flag) for step in STEPS) or args.human_redteam

    exit_codes: list[int] = []
    for step in STEPS:
        run_step(step, args, only_one, exit_codes)
    run_human_redteam(args, only_one)
    print_summary(exit_codes)
    report_ci_status(args, exit_codes)


if __name__ == "__main__":
    main()
