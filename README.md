# OpenAI Agents SDK로 Agent 개발하기

여기에서는 AWS 환경에서 [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents)를 이용해 agent를 개발하는 방법에 대해 설명합니다.

## OpenAI Agents SDK

[OpenAI Agents SDK - Python](https://github.com/openai/openai-agents-python)을 활용합니다.


## Hello Workd

```python
from openai import OpenAI 
client = OpenAI(
    base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    api_key=Bedrcok_API_Key,   
)
resp = client.responses.create(
    model="openai.gpt-5.4",
    input="Hello from Bedrock",
)
print(resp.output_text)
```



[OpenAI models in Amazon Bedrock](https://developers.openai.com/api/docs/guides/amazon-bedrock)
