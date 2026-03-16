from __future__ import annotations

import argparse
import json
import sys

from agentic_ai.runtime import build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular ReAct agent from the command line.")
    parser.add_argument("goal", nargs="?", help="High-level goal for the agent.")
    parser.add_argument("--goal", dest="goal_flag", help="High-level goal for the agent.")
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit a machine-readable JSON payload.",
    )
    return parser.parse_args()


def resolve_goal(args: argparse.Namespace) -> str:
    goal = args.goal_flag or args.goal
    if goal:
        return goal.strip()

    return sys.stdin.read().strip()


def main() -> int:
    args = parse_args()
    goal = resolve_goal(args)

    if not goal:
        message = "A goal is required. Pass it as an argument or via stdin."
        if args.emit_json:
            print(json.dumps({"ok": False, "error": message}))
        else:
            print(message, file=sys.stderr)
        return 1

    try:
        provider_name, agent = build_agent()
        answer = agent.run(goal)
        payload = {
            "ok": True,
            "provider": provider_name,
            "goal": goal,
            "answer": answer,
        }
        if args.emit_json:
            print(json.dumps(payload))
        else:
            print(f"LLM provider: {provider_name}")
            print(f"Goal: {goal}")
            print(f"Answer: {answer}")
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "error": str(exc),
            "type": type(exc).__name__,
        }
        if args.emit_json:
            print(json.dumps(payload))
        else:
            print(f"{payload['type']}: {payload['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
