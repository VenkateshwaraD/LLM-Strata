"""Agentic security test primitives without requiring a production agent runtime."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class AgentMemoryStore:
    """Small namespaced memory model used to test memory security policies.

    Data is kept in a plain in-process dict of {scope: {key: value}}, so one scope
    (e.g. one user) can never see another scope's keys unless the caller passes
    the same scope string. Nothing is persisted.
    """

    def __init__(self, allow_untrusted_writes: bool = False):
        self.allow_untrusted_writes = allow_untrusted_writes
        self._memory: dict[str, dict[str, str]] = {}

    def write(self, scope: str, key: str, value: str, trusted: bool = False) -> dict[str, Any]:
        # Fail closed: untrusted writes are rejected unless the policy explicitly allows them.
        if not trusted and not self.allow_untrusted_writes:
            return {"allowed": False, "reason": "untrusted_memory_write_denied"}
        self._memory.setdefault(scope, {})[key] = value
        return {"allowed": True, "reason": "memory_written"}

    def read(self, scope: str, key: str) -> dict[str, Any]:
        # `found` is based on `is not None`, so an empty-string value still counts as present.
        value = self._memory.get(scope, {}).get(key)
        return {"found": value is not None, "value": value}

    def reset(self, scope: str) -> dict[str, Any]:
        # Drops the whole scope; resetting a scope that doesn't exist is a harmless no-op.
        self._memory.pop(scope, None)
        return {"allowed": True, "reason": "memory_reset"}


def evaluate_tool_authorization(
    policies: dict[str, Any],
    role: str,
    tool: str,
    approved: bool = False,
) -> dict[str, Any]:
    """Evaluate a deny-by-default role/tool authorization policy."""
    # Unknown (or empty) role -> deny. Role names are matched exactly (case-sensitive).
    policy = policies.get(role)
    if not isinstance(policy, dict) or not policy:
        return {"allowed": False, "reason": "unknown_role"}

    raw_allowed_tools = policy.get("allowed_tools")
    if not isinstance(raw_allowed_tools, (list, tuple, set)) or not raw_allowed_tools:
        return {"allowed": False, "reason": "tool_not_allowlisted"}

    denied_tools = set(policy.get("denied_tools", []))
    allowed_tools = set(raw_allowed_tools)
    approval_required = set(policy.get("approval_required", []))

    # Evaluation order matters: explicit deny > allowlist > approval gate > allow.
    if tool in denied_tools:
        return {"allowed": False, "reason": "tool_denied_by_policy"}
    # SECURITY NOTE: the allowlist is only enforced when it is non-empty. A role with a
    # missing or empty `allowed_tools` therefore allows every tool that isn't in
    # `denied_tools`, which contradicts the "deny-by-default" docstring above.
    # Consider `if tool not in allowed_tools:` if roles must always declare an allowlist.
    if tool not in allowed_tools:
        return {"allowed": False, "reason": "tool_not_allowlisted"}
    if tool in approval_required and not approved:
        return {"allowed": False, "reason": "approval_required"}
    return {"allowed": True, "reason": "tool_allowed"}


def run_tool_authorization_tests(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate every configured tool authorization case against the policies."""
    policies = config.get("tool_policies", {})
    results = []
    for test in config.get("tool_authorization_tests", []):
        decision = evaluate_tool_authorization(
            policies=policies,
            role=str(test.get("role", "")),
            tool=str(test.get("tool", "")),
            approved=bool(test.get("approved", False)),
        )
        # `expected` is "allow" or "deny"; anything other than "allow" (including a
        # missing key) is treated as "deny".
        expected = str(test.get("expected", "deny")).lower() == "allow"
        results.append(
            {
                "id": test["id"],
                "role": test.get("role"),
                "tool": test.get("tool"),
                "expected_allowed": expected,
                "actual_allowed": decision["allowed"],
                "reason": decision["reason"],
                "passed": decision["allowed"] == expected,
            }
        )
    return results


def _matches_indicators(text: str, indicators: list[Any], patterns: list[Any]) -> bool:
    """Return True if `text` contains any unsafe indicator.

    `indicators` are case-insensitive plain substrings; `patterns` are case-insensitive
    regexes (config key: `unsafe_patterns`). Note that a refusal which merely *quotes* an
    indicator (e.g. "I won't append INTERNAL-OVERRIDE") will still match.
    """
    folded = text.casefold()
    if any(str(indicator).casefold() in folded for indicator in indicators):
        return True
    return any(re.search(str(pattern), text, re.IGNORECASE) for pattern in patterns)


def _replay_turns(client: Any, model: str, case: dict[str, Any],
                  max_tokens: int, outputs: list) -> None:
    """Replay a case's scripted turns, feeding model replies back as history.

    Per-turn results are appended to `outputs` so partial progress survives a
    mid-case failure.
    """
    messages = []
    if case.get("system_prompt"):
        messages.append({"role": "system", "content": str(case["system_prompt"])})

    # A turn may be a bare string or a dict like {"content": "..."}.
    for turn_number, turn in enumerate(case.get("turns", []), start=1):
        content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        messages.append({"role": "user", "content": str(content)})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=case.get("max_tokens", max_tokens),  # per-case override
        )
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        output = str(getattr(message, "content", "") or "")
        outputs.append({"turn": turn_number, "input": str(content), "output": output})
        messages.append({"role": "assistant", "content": output})


def _attack_result(case: dict[str, Any], outputs: list) -> dict[str, Any]:
    """Score one completed case: breached if any indicator appears in any reply."""
    # Indicators are checked against ALL replies joined together, not per turn.
    # A reply to an early attack turn that echoes an indicator therefore fails the
    # whole case, even if the model behaved correctly on the final turn.
    combined_output = "\n".join(item["output"] for item in outputs)
    breached = _matches_indicators(
        combined_output,
        case.get("unsafe_indicators", []),
        case.get("unsafe_patterns", []),
    )
    return {
        "id": case["id"],
        "goal": case.get("goal", ""),
        "turns": outputs,
        "attack_blocked": not breached,
        "passed": not breached,
    }


def _attack_failure(case: dict[str, Any], outputs: list, exc: Exception) -> dict[str, Any]:
    """Record a case-level failure; partial `outputs` are kept for debugging."""
    return {
        "id": case["id"],
        "goal": case.get("goal", ""),
        "turns": outputs,
        "attack_blocked": False,
        "passed": False,
        "error": str(exc),
    }


def run_multiturn_attacks(
    client: Any,
    model: str,
    cases: list[dict[str, Any]],
    max_tokens: int = 400,
) -> list[dict[str, Any]]:
    """Run scripted multi-turn attacks against a chat-completions-compatible client."""
    results = []
    for case in cases:
        outputs: list = []
        try:
            _replay_turns(client, model, case, max_tokens, outputs)
            results.append(_attack_result(case, outputs))
        except Exception as exc:  # per-case boundary: API/config/regex error -> failed test
            logger.error("multi-turn case '%s' failed: %s", case.get("id"), exc)
            results.append(_attack_failure(case, outputs, exc))
    return results


def _eval_write_op(store: AgentMemoryStore, operation: dict[str, Any]) -> tuple[dict, bool]:
    outcome = store.write(
        scope=str(operation.get("scope", "default")),
        key=str(operation.get("key", "")),
        value=str(operation.get("value", "")),
        trusted=bool(operation.get("trusted", False)),
    )
    return outcome, outcome["allowed"] == bool(operation.get("expected_allowed", True))


def _eval_read_op(store: AgentMemoryStore, operation: dict[str, Any]) -> tuple[dict, bool]:
    outcome = store.read(str(operation.get("scope", "default")),
                         str(operation.get("key", "")))
    # Precedence: expected_absent > expected_value > (default) key must exist.
    if operation.get("expected_absent", False):
        return outcome, not outcome["found"]
    if "expected_value" in operation:
        return outcome, outcome["value"] == operation["expected_value"]
    return outcome, outcome["found"]


def _eval_memory_operation(store: AgentMemoryStore, operation: dict[str, Any]) -> dict:
    """Run one memory operation and return its recorded outcome entry."""
    action = operation.get("action")
    scope = str(operation.get("scope", "default"))
    key = str(operation.get("key", ""))
    if action == "write":
        outcome, passed = _eval_write_op(store, operation)
    elif action == "read":
        outcome, passed = _eval_read_op(store, operation)
    elif action == "reset":
        outcome = store.reset(scope)
        passed = outcome["allowed"]
    else:
        # Unknown actions fail the test instead of being silently skipped (typo guard).
        outcome = {"error": f"unknown memory action: {action}"}
        passed = False
    return {"action": action, "scope": scope, "key": key,
            "result": outcome, "passed": passed}


def run_memory_tests(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every configured memory test against a fresh AgentMemoryStore."""
    memory_config = config.get("memory_policy", {})
    results = []
    for test in config.get("memory_tests", []):
        # A fresh store per test keeps tests independent of each other.
        store = AgentMemoryStore(
            allow_untrusted_writes=bool(memory_config.get("allow_untrusted_writes", False))
        )
        # A test passes only if every operation in it passed.
        operations = [_eval_memory_operation(store, operation)
                      for operation in test.get("operations", [])]
        passed = all(operation["passed"] for operation in operations)
        results.append({"id": test["id"], "operations": operations, "passed": passed})
    return results


def summarize_sections(sections: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate pass/fail totals across all sections."""
    # A result missing the "passed" key counts as a failure.
    all_results = [result for results in sections.values() for result in results]
    passed = sum(1 for result in all_results if result.get("passed"))
    return {
        "total": len(all_results),
        "passed": passed,
        "failed": len(all_results) - passed,
        "sections": {name: len(results) for name, results in sections.items()},
    }
