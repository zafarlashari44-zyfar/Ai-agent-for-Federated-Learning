from engine.ehr import EHREngine


class EHRAgent:

    def __init__(self):

        self.engine = EHREngine()

    def run(self, reasoning):

        return self.engine.generate(reasoning)