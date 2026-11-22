"""
Unit tests for the CLI console application.
Tests argument parsing, listing tools, listing files, and non-interactive queries.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.main import main, list_registered_tools, list_available_files


class TestCLIConsole:
    """Test CLI commands and flags."""

    def test_list_tools_command(self, capsys: pytest.CaptureFixture) -> None:
        with patch("sys.argv", ["main.py", "--tools"]):
            main()
        captured = capsys.readouterr()
        assert "web_search" in captured.out
        assert "calculator" in captured.out
        assert "file_reader" in captured.out

    def test_list_files_command(self, temp_data_dir: Path, mock_config, capsys: pytest.CaptureFixture) -> None:
        with patch("sys.argv", ["main.py", "--files"]):
            main()
        captured = capsys.readouterr()
        assert "sample.txt" in captured.out

    @patch("app.main.create_multi_tool_agent")
    def test_single_query_flag(self, mock_create_agent, capsys: pytest.CaptureFixture) -> None:
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        mock_res = MagicMock()
        mock_res.success = True
        mock_res.final_answer = "Mocked calculation result: 42"
        mock_agent.run.return_value = mock_res

        with patch("sys.argv", ["main.py", "--query", "Calculate 6 * 7"]):
            main()

        mock_agent.run.assert_called_once_with("Calculate 6 * 7", stream_to_console=True)

    @patch("app.main.config")
    def test_missing_api_key_exits(self, mock_cfg) -> None:
        mock_cfg.gemini_api_key = None
        with patch("sys.argv", ["main.py", "--query", "test"]), pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
