import copy
import os
import json

import pipmaster as pm  # Pipmaster for dynamic library install

if not pm.is_installed("aioboto3"):
    pm.install("aioboto3")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")
import aioboto3
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    locate_json_string_body_from_string,
)


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id="AKIATWBJZ4G6IN7JILVW",
    aws_secret_access_key="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",
    region_name='us-east-1',
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key

    )
    os.environ["AWS_REGION"] = os.environ.get("AWS_REGION", region_name)
    kwargs.pop("hashing_kv", None)
    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

    return response["output"]["message"]["content"][0]["text"]


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await bedrock_complete_if_cache(
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


# @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),  # TODO: fix exceptions
# )
async def bedrock_embed(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id="AKIATWBJZ4G6IN7JILVW",
    aws_secret_access_key="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",
    aws_session_token="IQoJb3JpZ2luX2VjEKr//////////wEaDGV1LWNlbnRyYWwtMSJHMEUCIQCMlINLrrdykQCl9M9mNMGuEKM8YGdCFBIm3PUWtj2iUQIgera8PIB54mT651zwVakdf+60sYmZsXAQM49BTwTkpfQqngMIExAAGgwyNTM0OTA3NDk4ODQiDCorU23TWV99QhsIsCr7AsmEfsgqcJjHZuXbJCCR/CPakwSF7VDY06NxW6UZJ42vvox6zU5Zx42Z0Cu9hno7s71vGEZK608zQ3nGat2nOd+6TgFmMcIoQnItjVZ20DF3oo9N/On5TzY+nFnJV2FBw1JL4EuGeN9OxTCrtx0U6Aqs8WljSkWSJZQ0YMgK5ytBWXcGWDI6TEo0SjMGpSLJqtAV77W8rql3QS0BPiO3nJHFc2jDjN/MsN/7nilkIH76hmgXOfHHPO2Il5qLEDUGnBrhcgyx7JvdPilK/+6Aob7lgiLSbcDXyW/hO5THxRReGXG3dvceHY7MrPedhAdhxc0fy+/GW+btg1O2vix71Z9+p5pCW3KJK3VqwkgvHQNDQ6qRBQmBVX5BbGGQnMa+W+Uc8bTR5asHJ25D3vZnZw/8Awv1KiiBpA9JA0A0zR0Aj94vHL3P2yNDFElKft9GxVf0oUqFpEA9j8Xl+kaOmnnlC4SnEmAhAQFAko4WYYuCIO53P0EQEHJ+Pr0wsoOKvwY6pgEVkAhMjKtMKinbjOm5TdZtQcZTwHdWqakseKvxTdE5x1cWWsHkgPRpkBV5HkS836nc1G2uowQ0/qiYoAqk3DW8VH78woZuwY+VWN9SYFaFcZtRU3OBHjzn/5fxFeE8YPR1wEsdobop9vy6ChNcHos797JarUu5E2tb0aUs7g3CXIYaFTprJq/enEF0QMWroWh6XiDTB2VilsEmlqvO12iJ15+XnRTF",
    region_name='us-east-3',
) -> np.ndarray:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )
    os.environ["AWS_REGION"] = os.environ.get("AWS_REGION", region_name)

    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        if (model_provider := model.split(".")[0]) == "amazon":
            embed_texts = []
            for text in texts:
                if "v2" in model:
                    body = json.dumps(
                        {
                            "inputText": text,
                            # 'dimensions': embedding_dim,
                            "embeddingTypes": ["float"],
                        }
                    )
                elif "v1" in model:
                    body = json.dumps({"inputText": text})
                else:
                    raise ValueError(f"Model {model} is not supported!")

                response = await bedrock_async_client.invoke_model(
                    modelId=model,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = await response.get("body").json()

                embed_texts.append(response_body["embedding"])
        elif model_provider == "cohere":
            body = json.dumps(
                {"texts": texts, "input_type": "search_document", "truncate": "NONE"}
            )

            response = await bedrock_async_client.invoke_model(
                model=model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            embed_texts = response_body["embeddings"]
        else:
            raise ValueError(f"Model provider '{model_provider}' is not supported!")

        return np.array(embed_texts)

