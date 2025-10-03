import pytest
from unittest.mock import AsyncMock
from mcp_transcriptions_server.main import mcp, FullRequest
from pathlib import Path


def test_mcp_server_initialization():
    """Test that the MCP server is initialized correctly."""
    assert mcp.name == "Transcriptions"


@pytest.mark.asyncio
async def test_mcp_server_tools():
    """Test that the MCP server has the correct tools."""
    tools = await mcp.list_tools()
    assert any(tool.name == "get_transcription_from_file" for tool in tools)
    tool = next(tool for tool in tools if tool.name == "get_transcription_from_file")
    assert tool.name == "get_transcription_from_file"
    # You might need to adjust this part based on how your MCP framework inspects tool signatures
    # This is a basic check
    assert "inputs" in tool.inputSchema["properties"]


@pytest.mark.asyncio
async def test_get_transcription_from_file_tool_direct_return(mocker):
    """Test the tool's direct return functionality."""
    mock_process = AsyncMock(return_value="direct transcript")
    mocker.patch(
        "mcp_transcriptions_server.main.process_transcript_request", new=mock_process
    )

    request = FullRequest(input_path=Path("dummy.mp3"))
    results = await mcp.call_tool("get_transcription_from_file", {"inputs": [request.model_dump()]})
    assert results[0][0].text == "direct transcript"
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_get_transcription_from_file_tool_save_to_file(mocker, tmp_path):
    """Test the tool's save-to-file functionality."""
    output_path = tmp_path / "transcript.txt"
    mock_process = AsyncMock()
    mocker.patch(
        "mcp_transcriptions_server.main.process_transcript_request", new=mock_process
    )

    request = FullRequest(input_path=Path("dummy.mp3"), output_path=output_path)
    results = await mcp.call_tool("get_transcription_from_file", {"inputs": [request.model_dump()]})
    assert results[0][0].text == f"Transcription saved to {output_path}"
    # Check that process_transcript_request was called with a file-like object
    mock_process.assert_called_once()
    call_args = mock_process.call_args
    assert call_args[0][0].input_path == request.input_path
    assert hasattr(call_args[1]["file"], "write")