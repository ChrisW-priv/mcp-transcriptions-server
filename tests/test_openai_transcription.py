import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import io

from mcp_transcriptions_server.openai_transciption import (
    get_a_transcript_from_file,
    process_transcript_request,
    TranscriptionRequest,
)


async def mock_get_transcript(request):
    return {"text": "saved transcription"}


@pytest.mark.asyncio
async def test_get_a_transcript_from_file_mocked(mocker):
    """Test that the async client is called correctly."""
    mock_async_openai = MagicMock()
    mock_async_openai.audio.transcriptions.create = AsyncMock(
        return_value="mocked transcription"
    )
    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.AsyncOpenAI",
        return_value=mock_async_openai,
    )
    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.get_file_bytes",
        return_value=b"dummy bytes",
    )

    result = await get_a_transcript_from_file(
        TranscriptionRequest(input_path=Path("dummy.mp3"))
    )

    assert result["text"] == "mocked transcription"
    mock_async_openai.audio.transcriptions.create.assert_called_once()


@pytest.mark.asyncio
async def test_process_transcript_request_save_to_file(mocker):
    """Test that the transcript is saved to a file using StringIO."""

    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.get_a_transcript_from_file",
        side_effect=mock_get_transcript,
    )

    request = TranscriptionRequest(input_path=Path("dummy.mp3"))
    string_io_buffer = io.StringIO()
    result = await process_transcript_request(request, file=string_io_buffer)

    assert result == "saved transcription"
    assert string_io_buffer.getvalue() == "saved transcription"


@pytest.mark.asyncio
async def test_process_transcript_request_return_directly(mocker):
    """Test that the transcript is returned directly."""

    mocker.patch(
        "mcp_transcriptions_server.openai_transciption.get_a_transcript_from_file",
        side_effect=mock_get_transcript,
    )

    request = TranscriptionRequest(input_path=Path("dummy.mp3"))
    result = await process_transcript_request(request)

    assert result == "saved transcription"
