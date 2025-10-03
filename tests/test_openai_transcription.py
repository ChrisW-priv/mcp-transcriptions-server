
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from mcp_transcriptions_server.openai_transciption import (
    get_a_transcript_from_file,
    process_transcript_request,
    TranscriptionRequest,
)


@pytest.mark.asyncio
async def test_get_a_transcript_from_file_mocked(mocker):
    """Test that the async client is called correctly."""
    mock_async_openai = MagicMock()
    mock_async_openai.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="mocked transcription")
    )
    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.AsyncOpenAI",
        return_value=mock_async_openai,
    )
    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.get_file_bytes",
        return_value=b"dummy bytes",
    )

    request = TranscriptionRequest(input_path=Path("dummy.mp3"))
    result = await get_a_transcript_from_file(request)

    assert result == "mocked transcription"
    mock_async_openai.audio.transcriptions.create.assert_called_once()


@pytest.mark.asyncio
async def test_process_transcript_request_save_to_file(mocker, tmp_path):
    """Test that the transcript is saved to a file."""
    request = TranscriptionRequest(input_path=Path("dummy.mp3"))
    output_path = tmp_path / "output.txt"

    async def mock_get_transcript(request):
        return "saved transcription"

    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.get_a_transcript_from_file",
        side_effect=mock_get_transcript,
    )


@pytest.mark.asyncio
async def test_process_transcript_request_return_directly(mocker):
    """Test that the transcript is returned directly."""
    request = TranscriptionRequest(input_path=Path("dummy.mp3"))

    async def mock_get_transcript(request):
        return "direct transcription"

    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.get_a_transcript_from_file",
        side_effect=mock_get_transcript,
    )

    result = await process_transcript_request(request)

    assert result == "direct transcription"
