"""Bedrock OpenAI model profiles (no Claude / Nova / LangChain)."""

openai_gpt_5_5_models = [
    {
        "bedrock_region": "us-east-2",
        "model_type": "openai",
        "model_id": "openai.gpt-5.5",
    },
]

openai_gpt_6_astra_models = [   # GPT-6 Astra via Bedrock Converse
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "openai",
        "model_id": "us.openai.gpt-6-astra",
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "openai",
        "model_id": "us.openai.gpt-6-astra",
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "openai",
        "model_id": "us.openai.gpt-6-astra",
    },
]

openai_gpt_5_6_sol_models = [   # GPT-5.6 Sol via Bedrock Converse
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-sol",
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-sol",
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-sol",
    },
]

openai_gpt_5_6_terra_models = [   # GPT-5.6 Terra via Bedrock Converse
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-terra",
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-terra",
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-terra",
    },
]

openai_gpt_5_6_luna_models = [   # GPT-5.6 Luna via Bedrock Converse
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-luna",
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-luna",
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "openai",
        "model_id": "us.openai.gpt-5.6-luna",
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
    if model_name == "OpenAI GPT 6 Astra":
        return openai_gpt_6_astra_models
    if model_name == "OpenAI GPT 5.6 Sol":
        return openai_gpt_5_6_sol_models
    if model_name == "OpenAI GPT 5.6 Terra":
        return openai_gpt_5_6_terra_models
    if model_name == "OpenAI GPT 5.6 Luna":
        return openai_gpt_5_6_luna_models
    if model_name == "OpenAI OSS 120B":
        return openai_oss_120b_models
    if model_name == "OpenAI OSS 20B":
        return openai_oss_20b_models
    return openai_gpt_5_5_models


def get_stop_sequence(model_name: str) -> str:
    return ""
