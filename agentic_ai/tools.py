from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from operator import add, floordiv, mod, mul, pow, sub, truediv
from typing import Any, Callable, ClassVar, Mapping


class Tool(ABC):
    """Base contract every external action must implement."""

    name: str
    description: str
    input_schema: Mapping[str, str]

    def format_for_prompt(self) -> str:
        arguments = ", ".join(f"{key}: {value}" for key, value in self.input_schema.items())
        return f"- {self.name}: {self.description} Input -> {{{arguments}}}"

    @abstractmethod
    def run(self, tool_input: Mapping[str, Any]) -> str:
        """Execute the tool and return a plain-text observation."""


class CalculatorTool(Tool):
    """Safely evaluates basic arithmetic expressions."""

    name = "calculator"
    description = (
        "Evaluates arithmetic expressions with +, -, *, /, //, %, **, and parentheses."
    )
    input_schema = {"expression": "string"}

    _binary_operators: ClassVar[dict[type[ast.operator], Callable[[float, float], float]]] = {
        ast.Add: add,
        ast.Sub: sub,
        ast.Mult: mul,
        ast.Div: truediv,
        ast.FloorDiv: floordiv,
        ast.Mod: mod,
        ast.Pow: pow,
    }
    _unary_operators: ClassVar[dict[type[ast.unaryop], Callable[[float], float]]] = {
        ast.UAdd: lambda value: value,
        ast.USub: lambda value: -value,
    }

    def run(self, tool_input: Mapping[str, Any]) -> str:
        expression = tool_input.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Calculator expects {'expression': '<arithmetic expression>'}.")

        parsed = ast.parse(expression, mode="eval")
        result = self._evaluate_node(parsed.body)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return str(result)

    def _evaluate_node(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.BinOp):
            operator = self._binary_operators.get(type(node.op))
            if operator is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return operator(self._evaluate_node(node.left), self._evaluate_node(node.right))

        if isinstance(node, ast.UnaryOp):
            operator = self._unary_operators.get(type(node.op))
            if operator is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return operator(self._evaluate_node(node.operand))

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")
