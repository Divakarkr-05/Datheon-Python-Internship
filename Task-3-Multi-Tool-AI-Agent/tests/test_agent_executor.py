"""
Unit and integration tests for Multi-Tool Autonomous AI Agent Executor.
Tests tool registration, tool selection, sequential multi-step tool calls,
max iterations boundary, error recovery, and observable callbacks with mocked LLM.
"""

from typing import Any, List
from unittest.mock import MagicMock
import pytest

from langchain_core.messages import AIMessage

from app.agent.callbacks import ObservableAgentCallbackHandler, ExecutionStep
from app.agent.executor import MultiToolAgent, AgentRunResult, create_multi_tool_agent
from tests.conftest import MockToolCallingChatModel


class TestAgentToolRegistration:
    """Verify tool registration and tool availability."""

    def test_tools_registered(self) -> None:
        agent = create_multi_tool_agent(llm=MockToolCallingChatModel(responses=[]))
        registered_names = [t.name for t in agent.tools]
        assert "web_search" in registered_names
        assert "calculator" in registered_names
        assert "file_reader" in registered_names
        assert len(registered_names) == 3


class TestAgentExecutionFlow:
    """Test autonomous agent execution loop with mocked LLM responses."""

    def test_empty_query_handling(self) -> None:
        agent = create_multi_tool_agent(llm=MockToolCallingChatModel(responses=[]))
        result = agent.run("", stream_to_console=False)
        assert result.success is False
        assert "non-empty" in result.final_answer

    def test_single_tool_execution(self) -> None:
        """Test agent autonomously calling the calculator tool and returning the final answer."""
        mock_llm = MockToolCallingChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "12500000 * 0.10"},
                            "id": "call_calc_001",
                        }
                    ],
                ),
                AIMessage(content="10% of $12,500,000 is $1,250,000."),
            ]
        )

        agent = create_multi_tool_agent(llm=mock_llm)
        result = agent.run("Calculate 10% of 12,500,000", stream_to_console=False)

        assert result.success is True
        assert "$1,250,000" in result.final_answer
        assert len(result.intermediate_steps) == 1
        assert result.intermediate_steps[0][0].tool == "calculator"
        assert "1250000" in str(result.intermediate_steps[0][1])

    def test_sequential_multi_tool_execution(self, temp_data_dir, mock_config) -> None:
        """
        Test compound query:
        1. file_reader reads company revenue
        2. calculator computes 20% of it
        3. LLM returns final answer
        """
        mock_llm = MockToolCallingChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "file_reader",
                            "args": {"filename": "sample.txt"},
                            "id": "call_file_001",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "12500000 * 0.20"},
                            "id": "call_calc_002",
                        }
                    ],
                ),
                AIMessage(
                    content="Based on sample.txt, Datheon's revenue is $12,500,000, and 20% of that is $2,500,000."
                ),
            ]
        )

        agent = create_multi_tool_agent(llm=mock_llm)
        result = agent.run(
            "Read sample.txt to find revenue and compute 20% of it",
            stream_to_console=False,
        )

        assert result.success is True
        assert "$2,500,000" in result.final_answer
        assert len(result.intermediate_steps) == 2
        tools_called = [step[0].tool for step in result.intermediate_steps]
        assert tools_called == ["file_reader", "calculator"]

    def test_max_iteration_limit(self) -> None:
        """Test agent halts gracefully when hitting max iterations."""
        # Provide repeating tool calls that exceed max_iterations
        mock_llm = MockToolCallingChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "1 + 1"},
                            "id": f"call_loop_{i}",
                        }
                    ],
                )
                for i in range(10)
            ]
        )

        agent = create_multi_tool_agent(llm=mock_llm, max_iterations=2)
        result = agent.run("Loop test", stream_to_console=False)

        assert result is not None
        assert len(result.intermediate_steps) <= 2


class TestObservableCallbacks:
    """Test callback step capturing."""

    def test_callback_captures_steps(self) -> None:
        handler = ObservableAgentCallbackHandler(stream_to_console=False)

        mock_action = MagicMock()
        mock_action.tool = "web_search"
        mock_action.tool_input = {"query": "test query"}

        handler.on_agent_action(mock_action)
        handler.on_tool_end("search result text")

        assert len(handler.steps) == 4
        assert handler.steps[0].step_type == "action"
        assert "[Action] Selecting web_search." in handler.steps[0].message
        assert handler.steps[1].step_type == "tool"
        assert handler.steps[2].step_type == "observation"
        assert handler.steps[3].step_type == "action"
