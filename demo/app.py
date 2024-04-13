
import enum
import logging
from typing_extensions import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from demo.search import search

app = FastAPI()

class SearchAlgorithm(str, enum.Enum):
    exact = "exact"
    approx = "approx"

class SearchQuery(BaseModel):
    search_query: str = Field(..., min_length=1, max_length=512)
    num_documents: Annotated[int, Field(..., ge=1, le=100)] = 20
    rescore_multiplier: Annotated[int, Field(..., ge=1, le=10)] = 1
    search_algorithm: SearchAlgorithm = SearchAlgorithm.approx
    reranking: bool = True
    rescoring: bool = False


@app.post("/search/")
async def perform_search(query: SearchQuery):
    """Endpoint to perform a semantic search on the Wikipedia dataset."""
    try:
        results = await search(
            query=query.search_query,
            top_k=query.num_documents,
            search_algorithm=query.search_algorithm,
            rescore_multiplier=query.rescore_multiplier,
            reranking=query.reranking,
            rescoring=query.rescoring,
        )
        return results
    except Exception as e:
        logging.error(f"An error occurred during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)