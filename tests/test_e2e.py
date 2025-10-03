from io import StringIO
import pytest
import os
from pathlib import Path
from mcp_transcriptions_server.openai_transciption import (
    process_transcript_request,
    TranscriptionRequest,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_text_transcript_from_file_e2e():
    """
    End-to-end test for the OpenAI transcription service.
    This test requires an OpenAI API key to be set in the environment variable OPENAI_API_KEY.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping e2e test")

    buff = StringIO()
    transcript = await process_transcript_request(
        TranscriptionRequest(input_path=Path(__file__).parent / "dummy_audio.mp3"), buff
    )

    assert "hello world" in transcript.lower()
    assert "hello world" in buff.getvalue().lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_json_transcript_from_file_e2e():
    """
    End-to-end test for the OpenAI transcription service.
    This test requires an OpenAI API key to be set in the environment variable OPENAI_API_KEY.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping e2e test")

    buff = StringIO()
    transcript = await process_transcript_request(
        TranscriptionRequest(
            input_path=Path(__file__).parent / "dummy_audio.mp3",
            response_format="json",
        ),
        buff,
    )

    assert "hello world" in transcript.lower()
    assert "hello world" in buff.getvalue().lower()
