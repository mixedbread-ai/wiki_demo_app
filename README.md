# Wiki Demo App

The Wiki Demo App is a FastAPI application that allows you to perform semantic search on a Wikipedia dataset using [mixedbread ai's](https://www.mixedbread.ai) embedding and reranking models. Check out the [blog post](https://www.mixedbread.ai/blog/binary-mrl).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/mixedbread-ai/wiki_demo_app
    ```

2. Navigate to the project directory:

    ```bash
    cd wiki-demo-app
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the environment variables:

    - Create a copy of the `.env.example` file and rename it to `.env`.

    - Open the `.env` file and fill in the required values for the environment variables.

5. Download the Indexes

    ```bash
    python download_index.py
    ```

## Usage

1. Start the FastAPI server:

    ```bash
    uvicorn demo.app:app --reload
    ```

2. Open your web browser and go to [http://localhost:8000/docs](http://localhost:8000/docs) to access the API documentation and test the endpoints.

### Endpoints

- `POST /search/`: Performs a semantic search on the Wikipedia dataset.

    Request body:

  - `search_query` (string): The search query.

  - `search_algorithm` (string, optional): The search algorithm to use. Can be `"exact"` or `"approx"`. Default: `"exact"`.

  - `num_documents` (integer, optional): The number of top results to return. Default: `100`.

  - `rescore_multiplier` (integer, optional): The multiplier for rescoring. Default: `1`.

  - `reranking` (boolean, optional): Whether to perform reranking of the results. Default: `true`.

  - `rescoring` (boolean, optional): Whether to perform rescoring of the results. Default: `false`.

    Response:

  - `results` (array): The search results, including the score, title, content, and source URL for each result.

  - `embedding_time` (float): The time taken for embedding the query.

  - `search_time` (float): The time taken for searching the index.

  - `sort_time` (float): The time taken for sorting the results.

  - `load_time` (float): The time taken for loading the embeddings (only applicable when `rescoring` is `true`).

  - `reranking_time` (float): The time taken for reranking the results (only applicable when `reranking` is `true`).

  - `total_time` (float): The total time taken for the search process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
