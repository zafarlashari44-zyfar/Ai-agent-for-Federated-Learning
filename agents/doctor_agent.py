class DoctorAgent:

    def run(self, ehr):

        return {

            "Diagnosis": ehr["Diagnosis"],

            "Confidence": ehr["Confidence"],

            "Risk": ehr["Risk"],

            "Clinical Summary": ehr["Clinical_Summary"],

            "Recommendation": ehr["Doctor_Recommendation"]

        }