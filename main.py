from __future__ import annotations

from agentic_ai.runtime import build_agent


def main() -> None:
    provider_name, agent = build_agent()
    goal = "What is 123 * 456?"
    answer = agent.run(goal)

    print(f"LLM provider: {provider_name}")
    print(f"Goal: {goal}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
