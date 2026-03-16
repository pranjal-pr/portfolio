from __future__ import annotations

import json
import re
from typing import Protocol


class LLMClient(Protocol):
    """Minimal interface the agent needs from an LLM backend."""

    def generate(self, prompt: str) -> str:
        ...


class OpenAIResponsesClient:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for OpenAIResponsesClient."
            ) from exc

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(model=self.model, input=prompt)
        output_text = getattr(response, "output_text", "").strip()
        if not output_text:
            raise RuntimeError("The OpenAI response did not include any text output.")
        return output_text


class DemoReActLLM:
    """
    Deterministic fallback used when no API key is configured.

    It only knows how to route arithmetic questions to the calculator tool, which
    keeps the ReAct loop runnable for local testing.
    """

    _goal_pattern = re.compile(r"Current goal:\s*(.*?)\n\nScratchpad:", re.DOTALL)
    _observation_pattern = re.compile(r"Observation:\s*(.+)")
    _expression_pattern = re.compile(r"([0-9\.\s\+\-\*\/%\(\)]+)")

    def generate(self, prompt: str) -> str:
        goal = self._extract_goal(prompt)
        observation = self._extract_last_observation(prompt)
        expression = self._extract_expression(goal)

        if observation is not None and expression is not None:
            return (
                "Thought: I have the calculator result and can answer the user directly.\n"
                f"Final Answer: {expression} = {observation}."
            )

        if expression is not None:
            return (
                "Thought: I should use the calculator to compute the arithmetic exactly.\n"
                "Action: calculator\n"
                f"Action Input: {json.dumps({'expression': expression})}"
            )

        return (
            "Thought: The local demo backend cannot reason about this task beyond arithmetic.\n"
            "Final Answer: Configure OPENAI_API_KEY to use a real model for general goals."
        )

    def _extract_goal(self, prompt: str) -> str:
        match = self._goal_pattern.search(prompt)
        return match.group(1).strip() if match else ""

    def _extract_last_observation(self, prompt: str) -> str | None:
        matches = self._observation_pattern.findall(prompt)
        return matches[-1].strip() if matches else None

    def _extract_expression(self, goal: str) -> str | None:
        candidates = [candidate.strip() for candidate in self._expression_pattern.findall(goal)]
        candidates = [candidate for candidate in candidates if any(char.isdigit() for char in candidate)]
        candidates = [
            candidate.rstrip("?.!,")
            for candidate in candidates
            if any(operator in candidate for operator in ("+", "-", "*", "/", "%"))
        ]
        return max(candidates, key=len, default=None)
