import os
import faiss
import numpy as np
import time

from usearch.index import Index
from datasets import load_dataset
from dotenv import load_dotenv
from mixedbread_ai.client import AsyncMixedbreadAI

load_dotenv()
api_key = os.getenv("MXBAI_API_KEY")
if not api_key:
    raise ValueError("MXBAI_API_KEY not found in environment variables")

mxbai = AsyncMixedbreadAI(api_key=api_key)
mxbai_req_options={"timeout": 60, "max_retries": 3}


def load_index():
    title_text_dataset = load_dataset(
        "mixedbread-ai/wikipedia-data-en-2023-11", split="train", num_proc=4
    ).select_columns(["title", "text", "url"])

    int8_view = Index.restore("wikipedia_int8_usearch_50m.index", view=True)
    binary_index: faiss.IndexBinaryFlat = faiss.read_index_binary(
        "wikipedia_ubinary_faiss_50m.index"
    )
    binary_ivf: faiss.IndexBinaryIVF = faiss.read_index_binary(
        "wikipedia_ubinary_ivf_faiss_50m.index"
    )
    return title_text_dataset, int8_view, binary_index, binary_ivf

title_text_dataset, int8_view, binary_index, binary_ivf = load_index()


async def embed_query(query: str) -> tuple:
    start_time = time.time()
    query_embedding = (await mxbai.embeddings(
        model="mxbai-embed-large-v1",
        prompt="Represent this sentence for searching relevant passages:",
        input=query,
        encoding_format=["int8", "ubinary"],
        normalized=True,
        request_options=mxbai_req_options,
    )).data[0].embedding
    emb_time = time.time() - start_time

    int8_embedding = np.asarray(query_embedding.int_8, dtype=np.int8)
    binary_embedding = np.asarray([query_embedding.ubinary], dtype=np.uint8)

    return int8_embedding, binary_embedding, emb_time

def search_index(binary_embedding: np.ndarray, top_k: int, rescore_multiplier: int, search_algorithm: str) -> tuple:
    index = binary_ivf if search_algorithm == "approx" else binary_index

    start_time = time.time()
    scores, binary_ids = index.search(binary_embedding, top_k * rescore_multiplier)
    search_time = time.time() - start_time

    binary_ids = binary_ids[0]
    scores = scores[0]

    return scores, binary_ids, search_time

def rescore(binary_ids: np.ndarray, int8_embedding: np.ndarray) -> tuple:
    start_time = time.time()
    int8_embeddings = int8_view[binary_ids].astype(int)
    load_time = time.time() - start_time

    start_time = time.time()
    scores = int8_embedding @ int8_embeddings.T
    rescore_time = time.time() - start_time

    return scores, load_time, rescore_time

def get_results(scores: np.ndarray, binary_ids: np.ndarray, top_k: int) -> tuple:
    start_time = time.time()
    indices = scores.argsort()[::-1][:top_k]
    top_k_indices = binary_ids[indices]
    top_k_scores = scores[indices]
    top_k_results = [
        (score, title_text_dataset[idx]["title"], title_text_dataset[idx]["text"], title_text_dataset[idx]["url"])
        for score, idx in zip(top_k_scores.tolist(), top_k_indices.tolist())
    ]
    sort_time = time.time() - start_time

    return top_k_results, sort_time

async def rerank(query: str, _results: list) -> tuple:
    start_time = time.time()
    reranked_results = (await mxbai.reranking(
        model="mixedbread-ai/mxbai-rerank-large-v1",
        query=query,
        input=[
            {"title": title, "text": text, "source": source}
            for _, title, text, source in _results
        ],
        request_options=mxbai_req_options,
    )).data
    rerank_time = time.time() - start_time

    reranked_results = [
        (doc.score, _results[doc.index][1], _results[doc.index][2], _results[doc.index][3])
        for doc in reranked_results
    ]

    return reranked_results, rerank_time

async def search(
    query: str,
    top_k: int,
    rescore_multiplier: int,
    search_algorithm: str,
    reranking: bool,
    rescoring: bool,
) -> dict:
    """Performs a semantic search on the Wikipedia dataset.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.
        rescore_multiplier (int): The multiplier for rescoring.
        search_algorithm (str): The search algorithm to use. Can be 'exact' or 'approx'.
        reranking (bool): Whether to perform reranking of the results.
        rescoring (bool): Whether to perform rescoring of the results.

    Returns:
        dict: The search results and timing information.
    """
    int8_embedding, binary_embedding, emb_time = await embed_query(query)

    scores, binary_ids, search_time = search_index(binary_embedding, top_k, rescore_multiplier, search_algorithm)

    load_time = 0
    rescore_time = 0
    if rescoring:
        scores, load_time, rescore_time = rescore(binary_ids, int8_embedding)

    results, sort_time = get_results(scores, binary_ids, top_k)

    rerank_time = 0
    if reranking:
        results, rerank_time = await rerank(query, results)

    return {
        "results": [
            {"score": float(round(score, 2)), "title": title, "content": content, "source": source}
            for score, title, content, source in results
        ],
        "embedding_time": emb_time,
        "search_time": search_time,
        "sort_time": sort_time,
        "load_time": load_time,
        "rescore_time": rescore_time,
        "reranking_time": rerank_time,
        "total_time": search_time + load_time + rescore_time + sort_time + rerank_time + emb_time,
    }
