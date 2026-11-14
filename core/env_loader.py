"""core/env_loader.py - Minimal .env file loader (no external dependencies).

Author : Venkateshwara Doijode
Project: LLM Strata - End-to-End LLM Security & Safety Framework

Reads key-value pairs from .env and sets them as environment variables.
Drop-in replacement for python-dotenv's load_dotenv().

Priority order (highest to lowest):
  1. System environment variable  - already set in OS (e.g. via PowerShell,
                                     Windows System Properties, or CI/CD pipeline)
                                     These are NEVER overwritten by .env.
  2. .env file                   - only applied if the variable is NOT already
                                     set in the system environment.

This means:
  - Set the credentials for your active provider
  OPENAI_API_KEY in your .env for local development
  - Set it as a system/CI variable in production - it will take precedence
  - You never need to change code between environments
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_KEYS_BY_PROVIDER = {
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
    "ollama": (),
    "gemini": ("GEMINI_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
}


def load_dotenv(env_path: str = ".env"):
    """Load variables from a .env file into os.environ.

    Variables already present in the system environment are never
    overwritten. Missing files are silently ignored.

    Args:
        env_path: Path to the .env file. Relative paths resolve against
            the repository root (the parent of the ``core`` package).
    """
    path = Path(env_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        return
    with open(path) as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def require_env(*keys: str):
    """Require that every given environment variable is set.

    Use this for tools that CANNOT run without the key. Exits the process
    with a non-zero status if any are missing.

    Args:
        *keys: Environment variable names to check.

    Example:
        require_env("OPENAI_API_KEY")
        require_env("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    """
    missing = [name for name in keys if not os.environ.get(name)]
    if not missing:
        return
    for name in missing:
        logger.error("%s not set.", name)
    logger.error("Add missing variables to your .env file or set them as system environment variables.")
    sys.exit(1)


def _provider_env_keys(provider: str, profile: dict) -> tuple[str, ...]:
    """Return the env var names a provider requires, or raise for unknown providers.

    Args:
        provider: Provider name (already lowercased).
        profile: Active profile dict (used for openai_compatible/custom).

    Raises:
        ValueError: If the provider name is not supported.
    """
    if provider in ("openai_compatible", "custom"):
        required: tuple[str, ...] = ()
        if not profile.get("base_url") and not os.environ.get("OPENAI_COMPATIBLE_BASE_URL"):
            required = ("OPENAI_COMPATIBLE_BASE_URL",)
        if profile.get("api_key_required", True):
            required += (profile.get("api_key_env", "OPENAI_COMPATIBLE_API_KEY"),)
        return required
    if provider not in _ENV_KEYS_BY_PROVIDER:
        raise ValueError(f"Unknown provider '{provider}'.")
    return _ENV_KEYS_BY_PROVIDER[provider]


def _active_provider(provider: str | None) -> tuple[str, dict]:
    """Return the effective provider name and the active profile."""
    from core.profile_loader import load_profile

    profile = load_profile()
    selected = (provider or profile.get("provider", "openai")).lower()
    return selected, profile


def require_provider_env(provider: str | None = None):
    """Require credentials for the active provider or a named provider.

    Args:
        provider: Provider to validate. Defaults to the active profile's
            provider from profiles.yaml.
    """
    selected, profile = _active_provider(provider)
    require_env(*_provider_env_keys(selected, profile))


def warn_env(*keys: str):
    """Log a warning for each missing environment variable, then continue.

    Use this for tools that have an offline/mock fallback.

    Args:
        *keys: Environment variable names to check.

    Example:
        warn_env("GEMINI_API_KEY")
    """
    missing = [name for name in keys if not os.environ.get(name)]
    if missing:
        logger.warning("%s not set - tool may run in offline/mock mode.", ", ".join(missing))


def warn_provider_env(provider: str | None = None):
    """Warn if credentials for the active provider are missing but continue.

    Provider-aware counterpart to require_provider_env(). Use this for tools
    that have an offline/mock fallback (e.g. Garak, LLM Guard).

    Args:
        provider: Provider to check. Defaults to the active profile's provider.

    Example:
        warn_provider_env()   # warns about whichever provider is active in profiles.yaml
    """
    selected, profile = _active_provider(provider)
    if selected in ("openai_compatible", "custom"):
        required: tuple[str, ...] = ()
    elif selected not in _ENV_KEYS_BY_PROVIDER:
        return  # unknown provider - nothing to warn about
    else:
        required = _ENV_KEYS_BY_PROVIDER[selected]
    warn_env(*required)
