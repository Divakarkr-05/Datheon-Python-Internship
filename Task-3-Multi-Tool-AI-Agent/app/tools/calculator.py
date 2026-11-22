"""
Safe Mathematical Calculator Tool for Task 3.
Evaluates arithmetic and mathematical expressions using AST (Abstract Syntax Tree) parsing.
Strictly disallows unrestricted eval(), exec(), and arbitrary code execution.
"""

import ast
import math
import re
from typing import Any, Union
from langchain_core.tools import tool

# Supported safe functions and constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "min": min,
    "max": max,
    "pow": pow,
}

# Supported AST operators
SAFE_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}

MAX_EXPRESSION_NODES = 100


def _preprocess_expression(expr: str) -> str:
    """Clean and normalize mathematical expression string."""
    cleaned = expr.strip()

    # Normalize unicode symbols
    cleaned = cleaned.replace("×", "*").replace("÷", "/")

    # Replace power caret ^ with **
    cleaned = cleaned.replace("^", "**")

    # Handle percentage pattern: "X% of Y" -> "((X / 100.0) * (Y))"
    cleaned = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\(?\d+(?:\.\d+)?\)?)",
        r"((\1 / 100.0) * \2)",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Handle trailing percentage pattern: "X%" -> "(X / 100.0)"
    # Ensure it's not a modulo operation (e.g. '10 % 3')
    cleaned = re.sub(
        r"(\d+(?:\.\d+)?)\s*%(?!\s*\d)",
        r"(\1 / 100.0)",
        cleaned,
    )

    return cleaned


def _eval_ast_node(node: ast.AST, node_count: list) -> Union[int, float]:
    """Recursively evaluate an AST node safely."""
    node_count[0] += 1
    if node_count[0] > MAX_EXPRESSION_NODES:
        raise ValueError("Expression is too complex (exceeded maximum AST node limit).")

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, node_count)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # Backward compatibility with older Python AST representations
    elif isinstance(node, getattr(ast, "Num", type(None))):
        return getattr(node, "n")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        left = _eval_ast_node(node.left, node_count)
        right = _eval_ast_node(node.right, node_count)

        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division or modulo by zero is undefined.")

        if op_type == ast.Pow:
            # Prevent excessive exponentiation attacks
            if isinstance(right, (int, float)) and right > 1000:
                raise OverflowError("Exponent too large (maximum allowed power is 1000).")

        return SAFE_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand, node_count)
        return SAFE_OPERATORS[op_type](operand)

    elif isinstance(node, ast.Name):
        name = node.id.lower()
        if name in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[name]
        raise ValueError(f"Disallowed identifier or variable name: '{node.id}'")

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Dynamic or nested function calls are not allowed.")

        func_name = node.func.id.lower()
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Disallowed function call: '{node.func.id}'. Safe functions: {list(SAFE_FUNCTIONS.keys())}")

        args = [_eval_ast_node(arg, node_count) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)

    raise ValueError(f"Disallowed expression element: {node.__class__.__name__}")


def evaluate_expression(expression: str) -> str:
    """
    Safely evaluate a mathematical expression string.
    Returns a human-readable result or structured error message.
    """
    if not expression or not expression.strip():
        return "Error: Empty expression provided."

    try:
        preprocessed = _preprocess_expression(expression)
        parsed = ast.parse(preprocessed, mode="eval")
        node_count = [0]
        result = _eval_ast_node(parsed, node_count)

        # Clean integer display if float has no fractional part
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"Result: {result}"

    except ZeroDivisionError as zde:
        return f"Error: {str(zde)}"
    except OverflowError as oe:
        return f"Error: {str(oe)}"
    except SyntaxError:
        return f"Error: Invalid mathematical syntax in expression '{expression}'."
    except ValueError as ve:
        return f"Error: {str(ve)}"
    except Exception as exc:
        return f"Error: Failed to evaluate expression: {str(exc)}"


@tool("calculator")
def calculator(expression: str) -> str:
    """
    Safely evaluate mathematical and arithmetic expressions.
    Use this tool to calculate numbers, percentages, sums, products, division, powers, or basic functions (sqrt, abs, round).
    Examples:
      - '12500000 * 0.10'
      - '15% of 8500'
      - '(450 + 120) / 2'
      - 'sqrt(144) + 2^5'
    Input must be a valid mathematical expression string.
    """
    return evaluate_expression(expression)
