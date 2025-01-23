import datetime
import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

print("start time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    # llm_model_name="gemma2:2b",
    # llm_model_name="qwen2m",
    llm_model_name="llama3.2",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

with open("./data/posts_prod_sample_cleaned.csv", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print("----------------------")
print("naive")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print("----------------------")
print("local")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print("----------------------")
print("global")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print("----------------------")
print("hybrid")
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)

# stream response
resp = rag.query(
    "What are the top themes in this story?",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)

print("end time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))