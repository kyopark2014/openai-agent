from openai import BedrockOpenAI


Bedrock_API_Key = "YOUR_BEDROCK_API_KEY"

client = BedrockOpenAI(
    aws_region="us-east-2",
    api_key=Bedrock_API_Key,
)

stream = client.responses.create(
    model="openai.gpt-5.5",
    input="AWS에서 OpenAI API를 사용하는 방법을 설명해주세요.",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)

print()