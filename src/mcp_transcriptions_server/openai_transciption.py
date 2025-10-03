from typing import Literal, TextIO

from pydantic import BaseModel, Field
from pathlib import Path
from openai import AsyncOpenAI
from openai.types import AudioModel, AudioResponseFormat
from anyio import open_file
from io import BytesIO


prompt_description = """An optional prompt to guide the transcription model's
output. Effective prompts can:

1. Correct specific words/acronyms: Include technical terms or names that might be misrecognized
    Example: "The transcript discusses OpenAI's DALL·E and GPT-4 technology"

2. Maintain context from previous segments: Include the last part of previous transcript
    Note: Model only considers final 224 tokens of the prompt

3. Enforce punctuation: Include properly punctuated example text
    Example: "Hello, welcome to my lecture. Today, we'll discuss..."

4. Preserve filler words: Include example with verbal hesitations
    Example: "Umm, let me think like, hmm... Okay, here's what I'm thinking"

5. Set writing style: Use examples in desired format (simplified/traditional, formal/casual)

The model will try to match the style and formatting of your prompt."""


class InputParams(BaseModel):
    input_path: Path


class TranscriptionRequest(InputParams):
    model: AudioModel = Field(
        "gpt-4o-mini-transcribe", description="The model to use for transcription"
    )
    response_format: AudioResponseFormat | None = Field(
        None, description="The format of the transcription"
    )
    prompt: str | None = Field(None, description=prompt_description)
    timestamp_granularities: list[Literal["word", "segment"]] | None = Field(
        None,
        description="The timestamp granularities to populate. ONLY APPLICABLE IF MODEL IS whisper-1",
    )


async def get_file_bytes(filepath: Path) -> bytes:
    async with await open_file(filepath, "rb") as file:
        return await file.read()


async def get_a_transcript_from_file(transcript_request: TranscriptionRequest) -> str:
    """
    Retrieves the file specified and sends the bytes to the transcription API.
    Returns the transcription text.
    """
    args = transcript_request.model_dump(exclude_none=True)
    filepath = args.pop("input_path")
    audio_file_bytes = await get_file_bytes(filepath)
    bytes = BytesIO(audio_file_bytes)
    bytes.name = filepath.name
    client = AsyncOpenAI()
    transcription = await client.audio.transcriptions.create(file=bytes, **args)
    return transcription.text


async def process_transcript_request(
    request: TranscriptionRequest, file: TextIO | None = None
) -> str:
    transcript = await get_a_transcript_from_file(request)
    if file is not None:
        file.write(transcript)
    return transcript
