from __future__ import annotations

import os

from .agent import Agent
from .llm import DemoReActLLM, LLMClient, OpenAIResponsesClient
from .memory import ConversationMemory, LongTermMemoryStore
from .tools import CalculatorTool


def build_llm() -> tuple[str, LLMClient]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if api_key:
        return ("openai", OpenAIResponsesClient(model=model, api_key=api_key))

    return ("demo", DemoReActLLM())


def build_agent() -> tuple[str, Agent]:
    provider_name, llm = build_llm()
    agent = Agent(
        llm=llm,
        tools=[CalculatorTool()],
        memory=ConversationMemory(),
        long_term_memory=LongTermMemoryStore(),
    )
    return provider_name, agent
