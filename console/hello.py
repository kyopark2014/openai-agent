from openai import BedrockOpenAI

client = BedrockOpenAI(aws_region="us-east-2")

response = client.responses.create(
    model="openai.gpt-5.5",
    input="AWS에서 OpenAI API를 사용하는 방법을 설명해주세요.",
)

print(response.output_text)