# OpenAI Agents SDK로 Agent 개발하기

여기에서는 AWS 환경에서 [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents)를 이용해 agent를 개발하는 방법에 대해 설명합니다.

## OpenAI Agents SDK

[OpenAI Agents SDK - Python](https://github.com/openai/openai-agents-python)을 활용합니다.

아래와 같이 Openai SDK를 설치합니다.

```bash
pip install -U "openai>=2.40.0"
```


## Hello World

아래와 같이 Bedrock Key를 등록합니다.

```baseh
export AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY"
```

이후 아래와 같이 결과를 확인할 수 있습니다.

```python
from openai import BedrockOpenAI

client = BedrockOpenAI(aws_region="us-east-2")

response = client.responses.create(
    model="openai.gpt-5.5",
    input="AWS에서 OpenAI API를 사용하는 방법을 설명해주세요.",
)

print(response.output_text)
```

## Streaming

아래와 같이 stream형태로 결과를 보여줄 수 있습니다.

```python
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
```


```bash
pip install aws-bedrock-token-generator
```

[OpenAI models in Amazon Bedrock](https://developers.openai.com/api/docs/guides/amazon-bedrock)
