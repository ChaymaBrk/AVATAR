import copy
import os
import json
import logging
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from openai import AzureOpenAI
import numpy as np
from bedrock import bedrock_complete, bedrock_embed
import aioboto3
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Directly set AWS credentials here
AWS_ACCESS_KEY_ID = "AKIATWBJZ4G6PNI7SOMB"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "us-east-1"

# Set the AWS credentials in the environment variables
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_REGION"] = AWS_REGION

# Set up logging
logging.basicConfig(level=logging.INFO)

# Directories and configurations
WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Bedrock error class
class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""

# Retry logic for Bedrock API calls
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
    **kwargs,
) -> str:
    # No need to load AWS credentials from environment variables anymore as they're hardcoded
    kwargs.pop("hashing_kv", None)

    # Format message history
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    if system_prompt:
        args["system"] = [{"text": system_prompt}]  # Ensure system prompt is a list of dicts with text key

    # Map inference parameters
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

# Bedrock complete function for your specific use case
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
    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    return result

# Bedrock embed function for embedding texts
async def bedrock_embed(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    region_name='us-east-1',
) -> np.ndarray:
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        if (model_provider := model.split(".")[0]) == "amazon":
            embed_texts = []
            for text in texts:
                if "v2" in model:
                    body = json.dumps(
                        {
                            "inputText": text,
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

# Function to initialize RAG (Retrieve and Generate)
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=bedrock_complete,
        llm_model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192, func=bedrock_embed
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

# Main function to run the application
def main():
    rag = asyncio.run(initialize_rag())

    with open(r"C:\Users\pc\Desktop\e-Tafakna\project\LightRAG\AVATAR\code de statut personnel.pdf", "rb") as f:
        rag.insert(f.read())

    for mode in ["naive", "local", "global", "hybrid"]:
        print("\n+-" + "-" * len(mode) + "-+")
        print(f"| {mode.capitalize()} |")
        print("+-" + "-" * len(mode) + "-+\n")
        print(
            rag.query(
                "HOW TO BE A TUNISIAN?", param=QueryParam(mode=mode)
            )
        )

if __name__ == "__main__":
    main()

# Updated clean_text function to handle non-string data
def clean_text(text):
    # Ensure 'text' is a string before calling strip
    if isinstance(text, str):
        return text.strip().replace("\x00", "")
    else:
        # Handle cases where text is not a string (e.g., integers, None)
        return str(text).strip().replace("\x00", "")  # Convert to string if it's not already
