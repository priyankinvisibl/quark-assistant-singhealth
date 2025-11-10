
from src.components.generators.bedrock_chat_generator import CustomBedrockChatGenerator
from haystack.dataclasses import ChatMessage
import re
import json


def analyze_log_with_custom_bedrock(error_log: str, pipeline_type: str) -> dict[str, str]:
    prompt = f'''
You are an expert in bioinformatics and {pipeline_type} pipeline debugging.
Given the following error log from a {pipeline_type} pipeline, analyze the issue and suggest a likely cause and possible fix.

Error Log:
{error_log}

Respond ONLY with a JSON object in this format:
{{
  "Explanation": "<explanation>",
  "Suggested Fix": "<suggested_fix>"
}}
'''  

    messages = [
        ChatMessage.from_user(prompt)
    ]

    chat_gen = CustomBedrockChatGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        generation_kwargs={"maxTokens": 1000}
    )

    output = chat_gen.run(messages)
    # Combine all reply contents if multiple replies
    # return "\n\n".join(reply.content for reply in output["replies"])
    combined = "\n\n".join(reply.content for reply in output["replies"])

    # Extract JSON object from the combined output
    match = re.search(r'\{.*\}', combined, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # fallback: return as string if parsing fails
    return {"error": "Could not parse JSON", "raw_output": combined}

