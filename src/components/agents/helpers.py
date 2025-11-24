from typing import Any

from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.generators.google_vertex import (
    VertexAIGeminiChatGenerator,
    VertexAIGeminiGenerator,
    VertexAIImageGenerator,
)
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

from src.components.generators.bedrock_chat_generator import (
    CustomBedrockChatGenerator,
)
from src.config.types import AgentSettings
from src.utils import get_boto3_creds


# TODO: override option
def get_generator(
    model_config: dict[str, Any],
    settings: AgentSettings,
    multimodal: bool = False,
    image: bool = True,
):
    values = model_config["values"]
    model_name = model_config["name"]
    if settings.vendor == "google":
        # TODO: Support needed for stream generation and toolconfig in VertexAI
        if multimodal:
            return model_name, VertexAIGeminiGenerator(
                project_id=values["project_id"],
                model=model_name,
            )
        elif image:
            return model_name, VertexAIImageGenerator(
                project_id=values["project_id"],
            )
        return model_name, VertexAIGeminiChatGenerator(
            **values,
            **settings.values,
            model=model_name,
        )

    elif settings.vendor == "aws":
        cv = values.copy()
        cv.update(get_boto3session_haystack(model_config))
        return model_name, CustomBedrockChatGenerator(
            model=model_name,
            **cv,
            **settings.values,
        )
    else:
        if multimodal:
            raise Exception("Not supported")
        return model_name, AzureOpenAIChatGenerator(
            **values,
        )


def get_generatorv2(model_config: dict[str, Any], settings: AgentSettings):
    values = model_config["values"]
    model_name = model_config["name"]
    cv = values.copy()
    # TODO:
    cv.pop("guardrailConfig")
    if ca := cv.pop("crossAccount", None):
        cv.update(get_boto3session_haystack(model_config))
        print(f"arn:aws:iam::{ca['id']}:role/{ca['role']}")
    return model_name, AmazonBedrockChatGenerator(
        model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        **cv,
    )


def get_boto3session_haystack(config: dict[str, Any]):
    sts = get_boto3_creds(config)
    if len(sts) > 0:
        return {key: Secret.from_token(value) for key, value in sts.items()}
    return {}
