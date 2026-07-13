from engine.dataset import DatasetEngine


class DatasetAgent:

    def __init__(self):

        self.engine = DatasetEngine()

    def run(self):

        print("Loading ECG Dataset...")

        train, test = self.engine.load()

        return {

            "train": train,

            "test": test

        }