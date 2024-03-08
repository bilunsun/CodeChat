import dspy
from dspy.functional import TypedChainOfThought
from pydantic import BaseModel

lm = dspy.OllamaLocal(model="dolphin-mixtral:latest", model_type="chat")
dspy.settings.configure(lm=lm)


class BooleanModel(BaseModel):
    bool_value: bool


def test_dspy():
    class GenerateAnswer(dspy.Signature):
        """Determine whether a fact is True or False."""

        fact = dspy.InputField()
        answer: BooleanModel = dspy.OutputField(desc="True or False")

    class Model(dspy.Module):
        def __init__(self):
            super().__init__()

            self.generate_answer = TypedChainOfThought(GenerateAnswer)

        def forward(self, fact):
            pred = self.generate_answer(fact=fact)
            return dspy.Prediction(answer=pred.answer)

    model = Model()
    print(model("Fire is cold."))


if __name__ == "__main__":
    test_dspy()
