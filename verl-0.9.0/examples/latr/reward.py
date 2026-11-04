# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Countdown reward for the response-only interface used by verl 0.9.0."""

import ast
import math
import operator
import re


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return (-1 if isinstance(node.op, ast.USub) else 1) * _evaluate(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
        operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        return operation[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    raise ValueError("Unsupported Countdown expression")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if "countdown" not in data_source.lower():
        raise ValueError("This reward supports Countdown only")
    # Keep the old last-line extraction rule, but do not require an assistant prefix
    # because the new reward manager decodes only response tokens.
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str.split("\n")[-1])
    if not matches:
        return {"score": 0.0, "acc": 0.0}
    equation = matches[-1].strip()
    try:
        numbers = [int(item) for item in re.findall(r"\d+", equation)]
        if sorted(numbers) != sorted(ground_truth["numbers"]):
            return {"score": 0.1, "acc": 0.0}
        if len(equation) > 1024 or not re.fullmatch(r"[\d+\-*/().\s]+", equation):
            raise ValueError("Invalid expression")
        result = _evaluate(ast.parse(equation, mode="eval").body)
        correct = math.isfinite(result) and abs(result - ground_truth["target"]) < 1e-5
    except (ValueError, SyntaxError, ArithmeticError, RecursionError):
        correct = False
    return {"score": 1.0 if correct else 0.1, "acc": float(correct)}
