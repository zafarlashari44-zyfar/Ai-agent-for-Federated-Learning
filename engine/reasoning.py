from datetime import datetime


class ClinicalReasoningEngine:

    def generate(
        self,
        prediction,
        confidence,
        shap_features,
        signal_features,
    ):

        confidence = float(confidence)

        if confidence > 1:
            confidence = confidence / 100

        if confidence >= 0.90:
            risk = "Low"
        elif confidence >= 0.70:
            risk = "Medium"
        else:
            risk = "High"

        reasoning = {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "risk": risk,
            "important_features": shap_features,
            "heart_rate": signal_features["heart_rate"],
            "rr_interval": signal_features["rr_interval"],
            "clinical_summary": (
                f"The ECG is classified as {prediction} with "
                f"{round(confidence * 100, 2)}% confidence. "
                f"The most important ECG regions are "
                f"{', '.join(shap_features)}."
            ),
            "generated_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        return reasoning