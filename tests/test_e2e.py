import pytest
import os
from pathlib import Path
from mcp_transcriptions_server.openai_transciption import (
    get_a_transcript_from_file,
    TranscriptionRequest,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_a_transcript_from_file_e2e():
    """
    End-to-end test for the OpenAI transcription service.
    This test requires an OpenAI API key to be set in the environment variable OPENAI_API_KEY.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping e2e test")

    audio_path = Path(__file__).parent / "dummy_audio.mp3"
    request = TranscriptionRequest(input_path=audio_path)

    transcript = await get_a_transcript_from_file(request)

    assert "hello world" in transcript.lower()
