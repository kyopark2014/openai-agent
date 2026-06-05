"""Bedrock OpenAI model profiles (no Claude / Nova / LangChain)."""

openai_gpt_5_5_models = [
    {
        "bedrock_region": "us-east-2",
        "model_type": "openai",
        "model_id": "openai.gpt-5.5",
    },
]

openai_oss_120b_models = [
    {
        "bedrock_region": "us-west-2",
        "model_type": "openai",
        "model_id": "openai.gpt-oss-120b-1:0",
    }
]

openai_oss_20b_models = [
    {
        "bedrock_region": "us-west-2",
        "model_type": "openai",
        "model_id": "openai.gpt-oss-20b-1:0",
    }
]


def get_model_info(model_name: str) -> list[dict]:
    if model_name == "OpenAI GPT 5.5":
        return openai_gpt_5_5_models
    if model_name == "OpenAI OSS 120B":
        return openai_oss_120b_models
    if model_name == "OpenAI OSS 20B":
        return openai_oss_20b_models
    return openai_gpt_5_5_models


def get_stop_sequence(model_name: str) -> str:
    return ""
