import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
import numpy as np
from bedrock import bedrock_complete, bedrock_embed
from lightrag.utils import EmbeddingFunc

logging.basicConfig(level=logging.INFO)

load_dotenv()



WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)




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



def main():
    rag = asyncio.run(initialize_rag())

    with open("./book_1.txt", "r", encoding="utf-8") as f:
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