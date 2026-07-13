from engine.prediction import PredictionEngine


class PredictionAgent:

    def __init__(self):

        self.engine = PredictionEngine()

    def run(self, signals):

        result = self.engine.predict(
            signals["signal"]
        )

        return {
            "prediction": result["prediction"],
            "confidence": float(result["confidence"]),
        }