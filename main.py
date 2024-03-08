import dspy
from dspy.functional import TypedPredictor
from pydantic import BaseModel
from dsp.utils import deduplicate
from rich import print
from rich.syntax import Syntax

from retrieve import DSPythonicRMClient
from chunking import chunk_directory
from embedding.e5_embed import embed

# from embedding.nomic_embed import embed, TaskType
from weaviate_db import create_from_chunks_and_embeddings

# # Setup
# chunks = chunk_directory(".", required_exts=[".py"])
# # chunks = chunk_directory("demo_dir", required_exts=[".txt"])
# embeddings = embed(chunks)
# # embeddings = embed(chunks, task_type=TaskType.SEARCH_DOCUMENT)
# create_from_chunks_and_embeddings(chunks, embeddings)

lm = dspy.OllamaLocal(model="dolphin-mixtral:latest", temperature=0.2)
rm = DSPythonicRMClient(k=3)
dspy.settings.configure(lm=lm, rm=rm)


def print_context_answer(pred):
    print("\nCONTEXT:")
    for c in pred.context:
        print("-" * 80)
        syntax = Syntax(c, "python", theme="one-dark", line_numbers=True)
        print(syntax)
    print("=" * 80)
    print("ANSWER:", pred.answer)


def base_rag():
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
    pred = rag("Which LLM is used?")
    print_context_answer(pred)


def relevancy_rag():
    """
    Adds a RelevancyCheck module for the basic RAG example.
    Also tests the typed output capabilities with BoolModel.

    For the question "Which LLM is used?", base_rag() outputs:
    - "The LLM used in this project is OpenLLAMA."
    - "An OpenAI model (such as GPT-3 or GPT-4) is likely being used for this project."
    - "It cannot be determined which LLM is used in this project based on the provided information."

    whereas relevancy_rag() correctly keeps only the single relevant chunk,
    and consistently outputs "ANSWER: Dolphin-mixtral".

    New output:

    Passage 0: Irrelevant.
    Passage 1: Irrelevant.
    Passage 2: Relevant.

    CONTEXT:
    --------------------------------------------------------------------------------
     1 import dspy
     2
     3 from retrieve import DSPythonicRMClient
     4 from chunking import chunk_directory
     5 from embedding.e5_embed import embed
     6
     7 # from embedding.nomic_embed import embed, TaskType
     8 from weaviate_db import create_from_chunks_and_embeddings
     9
    10 # Setup
    11 chunks = chunk_directory(".", required_exts=[".py"])
    12 # chunks = chunk_directory("demo_dir", required_exts=[".txt"])
    13 embeddings = embed(chunks)
    14 # embeddings = embed(chunks, task_type=TaskType.SEARCH_DOCUMENT)
    15 create_from_chunks_and_embeddings(chunks, embeddings)
    16
    17 lm = dspy.OllamaLocal(model="dolphin-mixtral:latest")
    18 rm = DSPythonicRMClient(k=3)
    19 dspy.settings.configure(lm=lm, rm=rm)
    ================================================================================
    ANSWER: dolphin-mixtral
    """

    class BoolModel(BaseModel):
        value: bool

    class GenerateAnswer(dspy.Signature):
        """Answer questions with short factoid answers."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RelevancyCheck(dspy.Signature):
        """Determine whether the provided context is relevant in answering the question."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        relevant: BoolModel = dspy.OutputField()

    class RelevancyRAG(dspy.Module):
        def __init__(self, k: int = 5):
            super().__init__()

            self.relevancy_check = TypedPredictor(RelevancyCheck)
            self.retrieve = dspy.Retrieve(k=k)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        def forward(self, question):
            context = []

            passages = self.retrieve(question).passages
            for i, p in enumerate(passages):
                relevant = self.relevancy_check(
                    context=p, question=question
                ).relevant.value

                print(f"Passage {i}:", end=" ")
                if relevant:
                    print("Relevant.")
                    context = deduplicate(context + [p])
                else:
                    print("Irrelevant.")

            pred = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=pred.answer)

    rag = RelevancyRAG()
    pred = rag("Which LLM is used?")

    print_context_answer(pred)


if __name__ == "__main__":
    base_rag()
    relevancy_rag()
