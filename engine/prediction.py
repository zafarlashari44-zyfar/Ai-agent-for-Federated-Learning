import numpy as np


class PredictionEngine:

    def predict(self, signal):

        signal = np.asarray(signal, dtype=float)

        prediction_index = int(np.argmax(signal[:5]))
        confidence = float(np.max(signal[:5]))

        labels = {
            0: "Normal Sinus Rhythm",
            1: "Supraventricular Ectopic",
            2: "Ventricular Arrhythmia",
            3: "Fusion Beat",
            4: "Unknown/Unclassifiable",
        }

        return {
            "prediction": labels.get(
                prediction_index,
                "Unknown/Unclassifiable",
            ),
            "confidence": confidence,
        }