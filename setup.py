"""setup.py - One-time setup for LLM Security Framework.

Cross-platform: Windows and Linux.

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

Usage: python setup.py
"""

import subprocess
import sys
from pathlib import Path

RESULT_DIRS = (
    "garak", "deepeval", "ragas", "pyrit", "llm_guard", "human_redteam",
    "encoded_attacks", "in_context_attacks", "multilingual", "tool_inject",
    "backdoor", "rag_security", "agent_security",
)


def run(cmd: list[str], check: bool = True):
    """Run a command, echoing it first."""
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)


def check_python() -> None:
    """Fail fast if the interpreter is too old."""
    print("\n[1/4] Checking Python...")
    version = sys.version_info
    if version < (3, 10):
        print(f"  ERROR: Python 3.10+ required (found {version.major}.{version.minor})")
        sys.exit(1)
    print(f"  Python {version.major}.{version.minor}.{version.micro} OK")
    if version >= (3, 13):
        print(f"  WARNING: Python {version.major}.{version.minor} detected.")
        print("    Some dependencies (garak, ragas, pyrit) may not yet support Python 3.13+.")
        print("    Use Python 3.10-3.12 for best compatibility.")


def create_venv(name: str) -> Path:
    """Create a virtual environment if it does not already exist."""
    venv = Path(name)
    if venv.exists():
        print(f"  {name} already exists - skipping.")
    else:
        run([sys.executable, "-m", "venv", name])
        print(f"  Virtual environment created at {name}/")
    return venv


def venv_pip(venv: Path) -> str:
    """Return the pip executable path inside a virtual environment."""
    if sys.platform == "win32":
        return str(venv / "Scripts" / "pip")
    return str(venv / "bin" / "pip")


def create_envs() -> None:
    """Create the main and isolated Garak virtual environments."""
    print("\n[2/4] Creating virtual environments...")
    create_venv(".venv")
    garak_venv = create_venv(".venv-garak")
    run([venv_pip(garak_venv), "install", "--upgrade", "pip"])
    run([venv_pip(garak_venv), "install", "-r", "requirements-garak.txt"])
    print("  Garak packages installed.")


def install_dependencies() -> None:
    """Install main requirements into .venv."""
    print("\n[3/4] Installing dependencies...")
    pip = venv_pip(Path(".venv"))
    run([pip, "install", "--upgrade", "pip"])
    run([pip, "install", "-r", "requirements.txt"])
    print("  Main packages installed.")


def create_result_dirs() -> None:
    """Create the results directory for each layer."""
    print("\n[4/4] Creating results directories...")
    for layer in RESULT_DIRS:
        Path(f"results/{layer}").mkdir(parents=True, exist_ok=True)
        print(f"  results/{layer}/")
    print("  (LangFuse results live at cloud.langfuse.com)")


def print_instructions() -> None:
    """Print post-setup usage instructions."""
    activate = ".venv\\Scripts\\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    print()
    print_header("Setup complete!")
    print(f"  Activate environment:\n    {activate}")
    print("\n  Set credentials for your active provider in profiles.yaml")
    print("\n  Then run:")
    print("    python run_security.py                 # run all automated tools")
    print("    python run_security.py --garak-only    # Garak only")
    print("    python run_security.py --human-redteam # Human red team session")
    print("=" * 55)


def main() -> None:
    """Run the full setup."""
    print_header("LLM Strata - Setup")
    check_python()
    create_envs()
    install_dependencies()
    create_result_dirs()
    print_instructions()


if __name__ == "__main__":
    main()
