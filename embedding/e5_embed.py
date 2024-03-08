from sentence_transformers import SentenceTransformer


def embed(chunks: list[str]) -> list[list[float]]:
    """Simple wrapper for e5 embedding.
    Example usage:
    ```python
    data = embed(
        ["What is the meaning of life?", "Why is the sky blue?"],
    )
    ```
    """
    model = SentenceTransformer("intfloat/e5-large-v2", trust_remote_code=True)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

    return embeddings
