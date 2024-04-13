from huggingface_hub import hf_hub_download

files = [
    "wikipedia_int8_usearch_50m.index",
    "wikipedia_ubinary_faiss_50m.index",
    "wikipedia_ubinary_ivf_faiss_50m.index",
]

for file in files:
    hf_hub_download(
        repo_id="sentence-transformers/quantized-retrieval",
        filename=file,
        local_dir_use_symlinks=False,
        local_dir=".",
        repo_type="space",
    )
