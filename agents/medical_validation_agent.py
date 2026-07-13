from engine.validation import MedicalValidationEngine


class MedicalValidationAgent:

    def __init__(self):

        self.engine = MedicalValidationEngine()

    def run(self, ehr):

        cleaned = {}

        for key, value in ehr.items():

            if isinstance(value, str):

                cleaned[key] = self.engine.clean(value)

            else:

                cleaned[key] = value

        return cleaned