import re


class MedicalValidationEngine:

    def clean(self, report):

        replacements = {
            "##th": "",
            "##zziness": "dizziness",
            "sync": "syncope",
            "di": "dizziness",
            "resting": "",
            "vital": "",
            "tri": "",
            "Pipeline Serialization Alert": "",
            "Automated formatting error.": "",
        }

        text = str(report)

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text)

        return text.strip()