"""
Multi-Tool Autonomous AI Agent Executor.
Coordinates LangChain tool-calling agents, tool dispatch, sequential execution,
and observation passing with observable step reporting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from app.agent.callbacks import ObservableAgentCallbackHandler, ExecutionStep
from app.agent.llm import get_gemini_llm
from app.core.config import config
from app.core.logger import logger
from app.tools.calculator import calculator
from app.tools.file_reader import file_reader
from app.tools.web_search import web_search

SYSTEM_PROMPT = """You are an autonomous AI research and computation assistant.
You are equipped with three specialized tools:
1. web_search: Search the live web in real-time for up-to-date facts, current news, GDP numbers, population statistics, or general web knowledge.
2. calculator: Safely evaluate mathematical expressions, percentages (e.g. '10% of 500'), arithmetic formulas, powers, and basic math functions.
3. file_reader: Read text from local documents (.txt and .pdf) residing inside the project's safe data directory.

CRITICAL EXECUTION GUIDELINES:
- Autonomously choose and invoke tools to solve compound problems.
- Decompose compound user queries into logical, sequential tool calls:
  * When asked to research and compute: first retrieve the factual metric via web_search or file_reader, then use calculator to compute the requested math based strictly on the retrieved metric.
  * When asked to read a local file: invoke file_reader with the filename.
  * When asked a pure calculation: invoke calculator directly.
- Never guess, invent, or hallucinate numbers, URLs, or document contents.
- If a tool reports that text extraction is unavailable (e.g. scanned image PDF) or no results found, clearly explain the limitation without fabricating answers.

SOURCE QUALITY, DATES & METRICS:
- Prioritize High Authority sources (governmental agencies, academic institutions, multilateral organizations like UN/World Bank, official reports).
- Never treat snippets from unverified, low-authority, or user-generated forum sources as verified facts.
- Clearly distinguish between projections/estimates versus historical census figures.
- Explicitly state publication years, reference periods, or source dates when citing statistics.

OUTPUT DISCIPLINE:
- Formulate a clear, direct, and well-reasoned final answer.
- Do NOT output internal monologues, drafting notes, or private chain-of-thought scratchpads.
- Deliver only the polished, final response citing the verified facts, file names, or search results used.
"""


@dataclass
class AgentRunResult:
    """Encapsulates the complete result of an agent run."""

    query: str
    final_answer: str
    steps: List[ExecutionStep] = field(default_factory=list)
    intermediate_steps: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class MultiToolAgent:
    """
    Autonomous AI Agent equipped with Web Search, Calculator, and File Reader tools.
    Wraps LangChain tool-calling agent with strict iteration limits and observable tracing.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm or get_gemini_llm()
        self.tools = tools if tools is not None else [web_search, calculator, file_reader]
        self.max_iterations = max_iterations or config.max_iterations

        # Build prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt or SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create LangChain tool-calling agent and executor
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            verbose=False,
        )

    def run(
        self,
        query: str,
        chat_history: Optional[List[Any]] = None,
        stream_to_console: bool = True,
    ) -> AgentRunResult:
        """
        Execute the agent on a user query with observable step reporting.
        """
        if not query or not query.strip():
            return AgentRunResult(
                query=query,
                final_answer="Please provide a non-empty question or instruction.",
                success=False,
                error="Empty query",
            )

        callback_handler = ObservableAgentCallbackHandler(stream_to_console=stream_to_console)

        try:
            inputs: Dict[str, Any] = {"input": query.strip()}
            if chat_history:
                inputs["chat_history"] = chat_history

            response = self.executor.invoke(
                inputs,
                config={"callbacks": [callback_handler]},
            )

            output_text = response.get("output", "")
            intermediate = response.get("intermediate_steps", [])

            return AgentRunResult(
                query=query,
                final_answer=output_text,
                steps=callback_handler.steps,
                intermediate_steps=intermediate,
                success=True,
            )

        except Exception as exc:
            error_message = f"Agent execution encountered an error: {str(exc)}"
            logger.error("Agent run failed: %s", exc, exc_info=True)

            if stream_to_console:
                print(f"\n[Execution Error] {error_message}\n")

            return AgentRunResult(
                query=query,
                final_answer=error_message,
                steps=callback_handler.steps,
                success=False,
                error=str(exc),
            )


def create_multi_tool_agent(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    max_iterations: Optional[int] = None,
) -> MultiToolAgent:
    """Factory helper to construct a configured MultiToolAgent."""
    return MultiToolAgent(
        llm=llm,
        tools=tools,
        max_iterations=max_iterations,
    )
