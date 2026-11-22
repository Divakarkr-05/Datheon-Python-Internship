"""
Unit tests for the Safe Mathematical Calculator Tool.
Tests arithmetic, operator precedence, powers, percentages, functions, syntax, and security boundaries.
"""

import pytest
from app.tools.calculator import calculator, evaluate_expression


class TestCalculatorOperations:
    """Test basic and advanced mathematical calculations."""

    def test_basic_arithmetic(self) -> None:
        assert evaluate_expression("2 + 3") == "Result: 5"
        assert evaluate_expression("100 - 45") == "Result: 55"
        assert evaluate_expression("12 * 12") == "Result: 144"
        assert evaluate_expression("100 / 4") == "Result: 25"
        assert evaluate_expression("10 // 3") == "Result: 3"
        assert evaluate_expression("10 % 3") == "Result: 1"

    def test_powers_and_carets(self) -> None:
        assert evaluate_expression("2 ** 8") == "Result: 256"
        assert evaluate_expression("2 ^ 8") == "Result: 256"
        assert evaluate_expression("3 ^ 3") == "Result: 27"

    def test_parentheses_and_precedence(self) -> None:
        assert evaluate_expression("2 + 3 * 4") == "Result: 14"
        assert evaluate_expression("(2 + 3) * 4") == "Result: 20"
        assert evaluate_expression("((10 + 20) * 2) / 6") == "Result: 10"

    def test_unary_operations(self) -> None:
        assert evaluate_expression("-5 + 10") == "Result: 5"
        assert evaluate_expression("+15 - -5") == "Result: 20"
        assert evaluate_expression("-(3 * 4)") == "Result: -12"

    def test_percentage_patterns(self) -> None:
        # 'X% of Y'
        assert evaluate_expression("10% of 500") == "Result: 50"
        assert evaluate_expression("5% of 1000") == "Result: 50"
        assert evaluate_expression("12.5% of 200") == "Result: 25"

        # Trailing percentage
        assert evaluate_expression("50% * 200") == "Result: 100"
        assert evaluate_expression("25%") == "Result: 0.25"

    def test_safe_functions_and_constants(self) -> None:
        assert evaluate_expression("sqrt(144)") == "Result: 12"
        assert evaluate_expression("abs(-42)") == "Result: 42"
        assert evaluate_expression("round(3.14159, 2)") == "Result: 3.14"
        assert evaluate_expression("min(10, 20, 5)") == "Result: 5"
        assert evaluate_expression("max(10, 20, 5)") == "Result: 20"
        assert evaluate_expression("round(pi, 4)") == "Result: 3.1416"

    def test_langchain_tool_wrapper(self) -> None:
        result = calculator.invoke({"expression": "15 * 4"})
        assert result == "Result: 60"
        assert calculator.name == "calculator"


class TestCalculatorSecurityAndErrors:
    """Test security boundaries and error handling."""

    def test_division_by_zero(self) -> None:
        result = evaluate_expression("10 / 0")
        assert "Error:" in result
        assert "zero" in result.lower()

    def test_modulo_by_zero(self) -> None:
        result = evaluate_expression("10 % 0")
        assert "Error:" in result

    def test_empty_expression(self) -> None:
        assert "Error: Empty expression" in evaluate_expression("")
        assert "Error: Empty expression" in evaluate_expression("   ")

    def test_syntax_errors(self) -> None:
        assert "Error: Invalid mathematical syntax" in evaluate_expression("5 + * 2")
        assert "Error: Invalid mathematical syntax" in evaluate_expression("((2 + 3)")

    def test_disallowed_identifier(self) -> None:
        result = evaluate_expression("unknown_var + 5")
        assert "Error: Disallowed identifier" in result

    def test_disallowed_functions(self) -> None:
        result = evaluate_expression("eval('2+2')")
        assert "Error:" in result
        assert "Disallowed function" in result

    def test_code_injection_prevention(self) -> None:
        result = evaluate_expression("__import__('os').system('dir')")
        assert "Error:" in result

    def test_excessive_exponent_overflow_prevention(self) -> None:
        result = evaluate_expression("2 ^ 10000")
        assert "Error:" in result
        assert "too large" in result.lower()
