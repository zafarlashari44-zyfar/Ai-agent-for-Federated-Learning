from engine.prediction import PredictionEngine


class PredictionAgent:

    def __init__(self):

        self.engine = PredictionEngine()

    def run(self, signals):

        signal = signals["signal"]

        prediction, confidence = self.engine.predict(signal)

        labels = {

            0: "Normal",

            1: "Supraventricular",

            2: "Ventricular",

            3: "Fusion",

            4: "Unknown"

        }

        return {

            "prediction":

            labels.get(

                prediction,

                "Unknown"

            ),

            "confidence":

            confidence

        }