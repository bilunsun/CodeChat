import dspy

from retrieve import DSPythonicRMClient
from chunking import chunk_directory
from nomic_embed import embed, TaskType
from weaviate_db import create_from_chunks_and_embeddings

# Setup
# chunks = chunk_directory(".", required_exts=[".py"])
chunks = chunk_directory("demo_dir", required_exts=[".txt"])
embeddings = embed(chunks, task_type=TaskType.SEARCH_DOCUMENT)
create_from_chunks_and_embeddings(chunks, embeddings)

lm = dspy.OllamaLocal(model="dolphin-mixtral:latest")
rm = DSPythonicRMClient(k=3)
dspy.settings.configure(lm=lm, rm=rm)


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


rag = RAG()
# pred = rag("How many properties did the National Accountability Bureau freeze?")
pred = rag("What is the scientific name of the Ifac silene?")
for c in pred.context:
    print("-" * 80)
    print(c)
print("=" * 80)
print("ANSWER:", pred.answer)
