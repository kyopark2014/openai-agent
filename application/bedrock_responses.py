"""Bedrock OpenAI Responses API helpers (no LangChain)."""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

from aws_bedrock_token_generator import provide_token
from openai import BedrockOpenAI

logger = logging.getLogger("bedrock-responses")

DEFAULT_MODEL_ID = "openai.gpt-5.5"
DEFAULT_REGION = "us-east-2"


def get_responses_region(model_type: str, bedrock_region: str, model_id: str = "") -> str:
    """Resolve Bedrock region for OpenAI-on-Bedrock Responses API."""
    if model_id == DEFAULT_MODEL_ID or model_id.startswith("openai.gpt-5"):
        return DEFAULT_REGION
    if model_type == "openai" and bedrock_region:
        return bedrock_region
    return DEFAULT_REGION


def get_responses_model_id(model_id: str, model_type: str) -> str:
    if model_type == "openai" and model_id:
        return model_id
    return DEFAULT_MODEL_ID


def create_client(region: str) -> BedrockOpenAI:
    return BedrockOpenAI(
        aws_region=region,
        bedrock_token_provider=lambda r=region: provide_token(region=r),
    )


def complete_text(
    *,
    instructions: str,
    user_input: str,
    model_id: str,
    region: str,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Single-turn text completion via Responses API."""
    client = create_client(region)
    kwargs: dict[str, Any] = {
        "model": model_id,
        "instructions": instructions,
        "input": user_input,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)
    return response.output_text or ""


def complete_multimodal(
    *,
    instructions: str,
    text: str,
    image_base64: str,
    model_id: str,
    region: str,
    media_type: str = "image/png",
) -> str:
    """Image + text completion via Responses API."""
    client = create_client(region)
    response = client.responses.create(
        model=model_id,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{image_base64}",
                    },
                    {"type": "input_text", "text": text},
                ],
            }
        ],
    )
    return response.output_text or ""


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """Simple recursive text splitter (replaces langchain RecursiveCharacterTextSplitter)."""
    if not text:
        return []
    separators = separators or ["\n\n", "\n", ". ", " ", ""]
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    rest_seps = separators[1:] if len(separators) > 1 else [""]

    if sep and sep in text:
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for i, part in enumerate(parts):
            piece = part if i == len(parts) - 1 else part + sep
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                if len(piece) > chunk_size and rest_seps:
                    chunks.extend(
                        split_text(piece, chunk_size, chunk_overlap, rest_seps)
                    )
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
    elif rest_seps:
        mid = max(1, len(text) // 2)
        left = split_text(text[:mid], chunk_size, chunk_overlap, rest_seps)
        right = split_text(text[mid:], chunk_size, chunk_overlap, rest_seps)
        return left + right
    else:
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i : i + chunk_size])
        return chunks

    merged: list[str] = []
    for ch in chunks:
        if merged and len(merged[-1]) < chunk_overlap and len(merged[-1]) + len(ch) <= chunk_size:
            merged[-1] += ch
        else:
            merged.append(ch)
    return merged
