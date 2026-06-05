"""Minimal OpenAI Agents SDK + Amazon Bedrock example."""

import asyncio

from aws_bedrock_token_generator import provide_token
from agents import Agent, Runner, function_tool, set_default_openai_client
from openai import AsyncBedrockOpenAI

AWS_REGION = "us-east-2"


@function_tool
def greet(name: str) -> str:
    return f"Hello, {name}!"


async def main():
    client = AsyncBedrockOpenAI(
        aws_region=AWS_REGION,
        bedrock_token_provider=lambda: provide_token(region=AWS_REGION),
    )
    set_default_openai_client(client)

    agent = Agent(
        name="assistant",
        instructions="Answer briefly in Korean.",
        model="openai.gpt-5.5",
        tools=[greet],
    )

    result = await Runner.run(agent, "greet tool로 World에게 인사해줘")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
