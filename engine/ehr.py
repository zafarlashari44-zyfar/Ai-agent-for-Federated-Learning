import json
from pathlib import Path
from engine.validation import MedicalValidationEngine


class EHREngine:

    def __init__(self):
        self.validator = MedicalValidationEngine()

    def generate(self, reasoning):

        ehr = {

            "Patient_ID": "PT-0001",

            "Diagnosis":
            reasoning["prediction"],

            "Confidence":
            reasoning["confidence"],

            "Risk":
            reasoning["risk"],

            "Heart_Rate":
            reasoning["heart_rate"],

            "RR_Interval":
            reasoning["rr_interval"],

            "Important_Features":
            reasoning["important_features"],

            "Clinical_Summary":
            reasoning["clinical_summary"],

            "Doctor_Recommendation":
            "Further ECG review is recommended.",

            "Patient_Advice":
            "Please consult your cardiologist if symptoms continue."

        }

        for key in ehr:

            if isinstance(ehr[key], str):

                ehr[key] = self.validator.clean(ehr[key])

        Path("reports").mkdir(exist_ok=True)

        with open(
            "reports/ehr_report.json",
            "w"
        ) as f:

            json.dump(
                ehr,
                f,
                indent=4
            )

        return ehr