from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from .llm import LLMClient
from .memory import ConversationMemory, LongTermMemoryStore
from .tools import Tool


@dataclass(slots=True)
class ParsedReActResponse:
    thought: str | None
    action: str | None
    action_input: str | None
    final_answer: str | None


class Agent:
    """Core ReAct agent loop with pluggable LLM and tools."""

    def __init__(
        self,
        llm: LLMClient,
        tools: Iterable[Tool],
        memory: ConversationMemory | None = None,
        long_term_memory: LongTermMemoryStore | None = None,
        max_steps: int = 8,
        history_window: int = 6,
    ) -> None:
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory or ConversationMemory()
        self.long_term_memory = long_term_memory or LongTermMemoryStore()
        self.max_steps = max_steps
        self.history_window = history_window

        if not self.tools:
            raise ValueError("Agent requires at least one registered tool.")

    def run(self, goal: str) -> str:
        self.memory.add("user", goal)
        scratchpad: list[str] = []
        recalled_memories = self.long_term_memory.search(goal, top_k=3)

        for step_index in range(1, self.max_steps + 1):
            prompt = self._build_prompt(goal=goal, scratchpad=scratchpad, recalled_memories=recalled_memories)
            raw_response = self.llm.generate(prompt)

            # The loop parses the model's plain-text ReAct response.
            # If the model emits an Action, we execute the tool and append the
            # resulting Observation back into the scratchpad for the next turn.
            parsed = self._parse_react_response(raw_response)

            if parsed.final_answer:
                final_answer = parsed.final_answer.strip()
                self.memory.add("assistant", final_answer)
                return final_answer

            if not parsed.action or not parsed.action_input:
                scratchpad.append(
                    "\n".join(
                        [
                            raw_response.strip(),
                            "Observation: Response format invalid. Use either "
                            "'Thought/Action/Action Input' or 'Thought/Final Answer'.",
                        ]
                    )
                )
                continue

            observation = self._execute_tool(action_name=parsed.action, raw_action_input=parsed.action_input)
            scratchpad.append(
                "\n".join(
                    [
                        f"Thought: {parsed.thought or 'Tool use required.'}",
                        f"Action: {parsed.action}",
                        f"Action Input: {parsed.action_input}",
                        f"Observation: {observation}",
                    ]
                )
            )

        raise RuntimeError(f"Agent did not finish within {self.max_steps} steps.")

    def _build_prompt(self, goal: str, scratchpad: list[str], recalled_memories: list[str]) -> str:
        tool_descriptions = "\n".join(tool.format_for_prompt() for tool in self.tools.values())
        conversation_history = self.memory.as_prompt(limit=self.history_window)
        long_term_context = (
            "\n".join(f"- {memory}" for memory in recalled_memories)
            if recalled_memories
            else "No relevant long-term memories."
        )
        scratchpad_text = "\n\n".join(scratchpad) if scratchpad else "No tool calls yet."

        return f"""You are an autonomous AI agent that follows the ReAct pattern.

Available tools:
{tool_descriptions}

Rules:
- Think briefly and explicitly.
- If you need a tool, respond using exactly:
  Thought: <reasoning>
  Action: <tool name>
  Action Input: <valid JSON object>
- If you can answer the user, respond using exactly:
  Thought: <reasoning>
  Final Answer: <answer>
- Never invent an Observation. Observations come only from executed tools.
- Only use tool names from the available tools list.

Conversation history:
{conversation_history}

Long-term memory:
{long_term_context}

Current goal:
{goal}

Scratchpad:
{scratchpad_text}
"""

    def _parse_react_response(self, response_text: str) -> ParsedReActResponse:
        thought = self._extract_block(response_text, "Thought", stop_labels=("Action", "Final Answer"))
        action = self._extract_line(response_text, "Action")
        action_input = self._extract_block(
            response_text,
            "Action Input",
            stop_labels=("Observation", "Thought", "Final Answer"),
        )
        final_answer = self._extract_block(response_text, "Final Answer")

        return ParsedReActResponse(
            thought=thought,
            action=action,
            action_input=action_input,
            final_answer=final_answer,
        )

    def _extract_line(self, text: str, label: str) -> str | None:
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_block(
        self,
        text: str,
        label: str,
        stop_labels: tuple[str, ...] = (),
    ) -> str | None:
        if stop_labels:
            stop_pattern = "|".join(re.escape(item) for item in stop_labels)
            pattern = rf"^{re.escape(label)}:\s*(.*?)(?=^\s*(?:{stop_pattern}):|\Z)"
        else:
            pattern = rf"^{re.escape(label)}:\s*(.*)$"

        match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _execute_tool(self, action_name: str, raw_action_input: str) -> str:
        tool = self.tools.get(action_name)
        if tool is None:
            available_tools = ", ".join(sorted(self.tools))
            return f"Unknown tool '{action_name}'. Available tools: {available_tools}."

        try:
            action_input = json.loads(raw_action_input)
        except json.JSONDecodeError as exc:
            return f"Action Input must be valid JSON. Parse error: {exc.msg}."

        if not isinstance(action_input, dict):
            return "Action Input must decode to a JSON object."

        try:
            return tool.run(action_input)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
