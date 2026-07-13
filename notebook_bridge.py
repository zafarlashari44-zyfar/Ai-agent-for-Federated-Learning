import joblib
import numpy as np


class NotebookBridge:

    def __init__(self):

        self.model = None

    def load_model(self, path):

        print("Loading trained model...")

        self.model = joblib.load(path)

        print("Model loaded.")

    def predict(self, sample):

        prediction = self.model.predict(sample)

        probability = self.model.predict_proba(sample)

        return {

            "prediction": int(prediction[0]),

            "confidence": float(

                np.max(probability)

            )

        }