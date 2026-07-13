import pandas as pd
from config import MITBIH_TRAIN, MITBIH_TEST


class DatasetEngine:

    def load(self):

        train = pd.read_csv(
            MITBIH_TRAIN,
            header=None,
        )

        test = pd.read_csv(
            MITBIH_TEST,
            header=None,
        )

        return train, test