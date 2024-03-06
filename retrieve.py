import dspy
from dataclasses import dataclass

from nomic_embed import embed, TaskType
from weaviate_db import query_weaviate_db


# Necessary since DSPy will access the long_text attribute of the passages
@dataclass
class Passage:
    long_text: str


class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, k: int = 3):
        super().__init__(k=k)

        self.k = k

    def forward(self, query: str, k: int | None = None) -> dspy.Prediction:
        query_vector = embed(query, task_type=TaskType.SEARCH_QUERY)[0]
        returned_chunks = query_weaviate_db(query_vector, k=k or self.k)

        return [Passage(chunk) for chunk in returned_chunks]
        # return dspy.Prediction(passages=[Passage(chunk) for chunk in returned_chunks])
