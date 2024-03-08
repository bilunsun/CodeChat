import torch.nn.functional as F
from enum import StrEnum
from sentence_transformers import SentenceTransformer


class TaskType(StrEnum):
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"


def embed(
    chunks: list[str], task_type: TaskType, matryoshka_dim: int = 768
) -> list[list[float]]:
    """Simple wrapper for nomic embedding v1.5.
    Example usage:
    ```python
    data = embed(
        ["What is the meaning of life?", "Why is the sky blue?"],
        task_type=TaskType.SEARCH_QUERY,
    )
    ```
    """
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )  # Requires einops
    sentences = [f"{task_type}: {chunk}" for chunk in chunks]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :matryoshka_dim]
    embeddings = F.normalize(embeddings, p=2, dim=1).tolist()

    return embeddings
