"""
Observable Agent Callback Handler.
Captures and formats observable execution steps (actions, tools, observations, next steps)
without exposing raw internal prompt chains or private chain-of-thought tokens.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import sys

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish


@dataclass
class ExecutionStep:
    """Represents a discrete observable execution step by the agent."""

    step_type: str  # 'action', 'tool', 'observation', 'error', 'finish'
    tool_name: Optional[str] = None
    tool_input: Optional[Any] = None
    tool_output: Optional[str] = None
    message: str = ""


class ObservableAgentCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that prints clean, user-friendly
    observable execution steps to the terminal in real time and retains
    structured step history for verification and testing.
    """

    def __init__(self, stream_to_console: bool = True) -> None:
        super().__init__()
        self.stream_to_console = stream_to_console
        self.steps: List[ExecutionStep] = []

    def _log(self, text: str) -> None:
        """Stream message to console if enabled."""
        if self.stream_to_console:
            print(text, flush=True)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Invoked when the agent decides to invoke a tool."""
        tool_name = action.tool
        tool_input = action.tool_input

        # Concise action step
        action_msg = f"[Action] Selecting {tool_name}."
        tool_msg = f"[Tool] {tool_name}\n[Input] {tool_input}"

        self.steps.append(
            ExecutionStep(
                step_type="action",
                tool_name=tool_name,
                tool_input=tool_input,
                message=action_msg,
            )
        )
        self.steps.append(
            ExecutionStep(
                step_type="tool",
                tool_name=tool_name,
                tool_input=tool_input,
                message=tool_msg,
            )
        )

        self._log(f"\n{action_msg}")
        self._log(tool_msg)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Invoked when a tool completes execution."""
        # Sanitize and truncate output if very long for clean display
        out_str = str(output).strip()
        obs_display = out_str if len(out_str) <= 600 else f"{out_str[:600]}... [truncated]"

        obs_msg = f"[Observation] {obs_display}"
        next_action_msg = "[Action] Analyzing observation to determine next step."

        self.steps.append(
            ExecutionStep(
                step_type="observation",
                tool_output=out_str,
                message=obs_msg,
            )
        )
        self.steps.append(
            ExecutionStep(
                step_type="action",
                message=next_action_msg,
            )
        )

        self._log(obs_msg)
        self._log(next_action_msg)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Invoked when a tool raises an unhandled exception."""
        err_msg = f"[Observation Error] Tool encountered an error: {str(error)}"
        fallback_action_msg = "[Action] Evaluating recovery from tool failure."

        self.steps.append(
            ExecutionStep(
                step_type="error",
                tool_output=str(error),
                message=err_msg,
            )
        )
        self.steps.append(
            ExecutionStep(
                step_type="action",
                message=fallback_action_msg,
            )
        )

        self._log(err_msg)
        self._log(fallback_action_msg)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Invoked when the agent generates the final answer."""
        final_answer = finish.return_values.get("output", "")
        finish_msg = f"\n[Final Answer]\n{final_answer}\n"

        self.steps.append(
            ExecutionStep(
                step_type="finish",
                message=finish_msg,
            )
        )
        self._log(finish_msg)
