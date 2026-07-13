from engine.reasoning import ClinicalReasoningEngine


class ReasoningAgent:

    def __init__(self):

        self.engine = ClinicalReasoningEngine()

    def run(

        self,

        prediction,

        shap,

        signals,

    ):

        return self.engine.generate(

            prediction["prediction"],

            prediction["confidence"],

            shap["features"],

            signals,

        )