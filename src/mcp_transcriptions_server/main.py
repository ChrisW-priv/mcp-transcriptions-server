import asyncio
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mcp_transcriptions_server.openai_transciption import (
    TranscriptionRequest,
    process_transcript_request,
)


mcp = FastMCP("Transcriptions")


def main() -> None:
    """Run main entrypoint."""
    mcp.run()


class FullRequest(TranscriptionRequest):
    output_path: Path | None = Field(
        None,
        description="Path of the file to save the transcript to. "
        "If not provided, the transcript will be returned as a string directly.",
    )


@mcp.tool()
async def get_transcription_from_file(inputs: list[FullRequest]) -> list[str]:
    """
    Transcribes audio files asynchronously.
    """

    async def process_transcription(input: FullRequest) -> str:
        """Transcribes a single file asynchronously.
        If the output path of the request is provided, the transcript will be saved to the file.
        If the output path is not provided, the transcript will be returned as a string.
        """
        if input.output_path is not None:
            with open(input.output_path, "w") as f:
                await process_transcript_request(input, file=f)
            return f"Transcription saved to {input.output_path}"
        transcript = await process_transcript_request(input)
        return transcript

    return await asyncio.gather(*[process_transcription(input) for input in inputs])


if __name__ == "__main__":
    main()
