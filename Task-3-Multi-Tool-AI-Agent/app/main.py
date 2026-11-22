"""
Interactive Console Interface for Task 3: Multi-Tool Autonomous AI Agent.
Supports interactive multi-turn sessions and single-query command-line runs.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure project root directory is on sys.path for direct script execution
TASK_ROOT = Path(__file__).resolve().parent.parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.agent.executor import MultiToolAgent, create_multi_tool_agent
from app.agent.llm import get_gemini_llm
from app.core.config import config, get_masked_api_key
from app.core.logger import logger
from app.tools.calculator import calculator
from app.tools.file_reader import file_reader
from app.tools.web_search import web_search

console = Console()


def display_welcome_banner() -> None:
    """Print clean startup banner and active configuration table."""
    banner_text = (
        "[bold cyan]Datheon Python Internship — Task 3[/bold cyan]\n"
        "[bold white]Multi-Tool Autonomous AI Agent[/bold white]\n"
        "[dim]Equipped with Real-Time Web Search, Safe Calculator & Local File Reader[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", expand=False))

    config_table = Table(title="Agent Configuration", border_style="dim", show_header=True)
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")

    config_table.add_row("LLM Model", config.gemini_model)
    config_table.add_row(
        "Gemini API Key",
        f"{get_masked_api_key(config.gemini_api_key)} ({'Detected' if config.gemini_api_key else 'Missing'})",
    )
    config_table.add_row("Search Provider", config.search_provider.upper())
    config_table.add_row("Max Search Results", str(config.search_max_results))
    config_table.add_row("Data Directory", str(config.data_dir))
    config_table.add_row("Max Agent Iterations", str(config.max_iterations))

    console.print(config_table)
    console.print(
        "[yellow]Type your question below, or use commands: [bold]tools[/bold], [bold]files[/bold], [bold]clear[/bold], [bold]exit[/bold][/yellow]\n"
    )


def list_registered_tools() -> None:
    """Display registered tools and descriptions in a table."""
    tool_table = Table(title="Registered Agent Tools", border_style="blue")
    tool_table.add_column("Tool Name", style="bold cyan")
    tool_table.add_column("Description", style="white")

    tool_table.add_row(
        "web_search",
        "Search the live web in real time for factual knowledge, current news, GDP numbers, population statistics."
    )
    tool_table.add_row(
        "calculator",
        "Safe AST-based arithmetic evaluator. Supports +, -, *, /, %, **, ^, parentheses, percentages ('10% of 500')."
    )
    tool_table.add_row(
        "file_reader",
        "Secure local document extractor for .txt and .pdf files located within the project data directory."
    )
    console.print(tool_table)


def list_available_files() -> None:
    """List readable documents inside the configured data directory."""
    files = list(config.data_dir.glob("*"))
    valid_files = [f for f in files if f.suffix.lower() in {".txt", ".pdf"}]

    if not valid_files:
        console.print(f"[yellow]No .txt or .pdf files found in {config.data_dir}[/yellow]")
        return

    file_table = Table(title=f"Available Documents in {config.data_dir.name}/", border_style="green")
    file_table.add_column("Filename", style="bold green")
    file_table.add_column("Type", style="cyan")
    file_table.add_column("Size", style="dim")

    for f in valid_files:
        size_kb = f.stat().st_size / 1024
        file_table.add_row(f.name, f.suffix.upper()[1:], f"{size_kb:.1f} KB")

    console.print(file_table)


def run_interactive_session(agent: MultiToolAgent) -> None:
    """Run interactive continuous prompt session."""
    display_welcome_banner()

    session_history: List[str] = []

    while True:
        try:
            user_input = console.input("[bold green]User > [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session interrupted. Goodbye![/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"exit", "quit", ":q"}:
            console.print("[dim]Exiting Multi-Tool Agent. Goodbye![/dim]")
            break
        elif cmd == "tools":
            list_registered_tools()
            continue
        elif cmd == "files":
            list_available_files()
            continue
        elif cmd == "clear":
            console.clear()
            display_welcome_banner()
            continue
        elif cmd == "help":
            console.print(
                "[cyan]Commands:[/cyan]\n"
                "  [bold]tools[/bold]  - List registered agent tools\n"
                "  [bold]files[/bold]  - List documents in local data directory\n"
                "  [bold]clear[/bold]  - Clear console screen\n"
                "  [bold]exit[/bold]   - Quit the application\n"
            )
            continue

        # Execute agent query
        start_time = time.time()
        console.print(f"\n[bold cyan]Agent processing query:[/bold cyan] [italic]{user_input}[/italic]")

        result = agent.run(user_input, stream_to_console=True)
        elapsed = time.time() - start_time

        session_history.append(user_input)
        console.print(f"[dim]Finished in {elapsed:.2f}s | Iterations: {len(result.intermediate_steps)}[/dim]\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Datheon Internship Task 3: Multi-Tool Autonomous AI Agent"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single non-interactive query and exit.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["duckduckgo", "tavily"],
        help="Override web search provider.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override maximum agent iteration steps.",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Display list of registered tools and exit.",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="Display list of readable data files and exit.",
    )

    args = parser.parse_args()

    if args.tools:
        list_registered_tools()
        return

    if args.files:
        list_available_files()
        return

    if args.provider:
        config.search_provider = args.provider

    if args.max_iterations:
        config.max_iterations = args.max_iterations

    # Validate Gemini API Key
    if not config.gemini_api_key and not args.tools and not args.files:
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY or GOOGLE_API_KEY is not set.\n"
            "Please copy [bold].env.example[/bold] to [bold].env[/bold] and configure your Gemini API key."
        )
        sys.exit(1)

    try:
        agent = create_multi_tool_agent(max_iterations=config.max_iterations)
    except Exception as exc:
        console.print(f"[bold red]Initialization Error:[/bold red] {exc}")
        sys.exit(1)

    if args.query:
        console.print(f"[bold cyan]User Request:[/bold cyan] {args.query}")
        result = agent.run(args.query, stream_to_console=True)
        if not result.success:
            sys.exit(1)
        return

    run_interactive_session(agent)


if __name__ == "__main__":
    main()
