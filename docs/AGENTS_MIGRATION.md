# OpenAI Agents SDK 아키텍처

[OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents)와 [Amazon Bedrock OpenAI 모델](https://developers.openai.com/api/docs/guides/amazon-bedrock)만 사용합니다.

**제거됨:** AWS Strands, LangChain, LangGraph, Anthropic Claude / Amazon Nova (앱 LLM 경로)

## 스택

| 레이어 | 구현 |
|--------|------|
| UI | `app.py` (Streamlit) |
| 일상 대화 | `chat.general_conversation` → `BedrockOpenAI.responses` |
| RAG / 이미지 / 요약 | `chat.py` + `bedrock_responses.py` |
| Agent | `openai_agent.py` (Agents SDK + MCP) |
| 모델 프로필 | `info.py` (OpenAI GPT 5.5, OSS 120B/20B) |

## 모델 선택 (UI)

- OpenAI GPT 5.5 (`us-east-2`)
- OpenAI OSS 120B / 20B (`us-west-2`)

## 의존성

```bash
pip install -r requirements.txt
```

- `openai` + `openai-agents`
- `aws-bedrock-token-generator`
- **없음:** `langchain*`, `strands*`, `langgraph`

## MCP

로컬 stdio MCP (`mcp_server_*.py`)는 Agents SDK `MCPServerStdio`로 연결합니다. 실행 전 `MCPServerManager`가 `connect()`를 호출합니다.
