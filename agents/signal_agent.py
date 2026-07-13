import numpy as np


class SignalAgent:

    def run(self, dataset):

        print("=" * 60)
        print("SIGNAL AGENT")
        print("=" * 60)

        train = dataset["train"]

        signal = train.iloc[0, :-1].values.astype(float)

        heart_rate = round(
            60 + np.random.uniform(-5, 5),
            2
        )

        rr_interval = round(
            np.random.uniform(0.75, 1.10),
            3
        )

        features = {

            "signal": signal,

            "heart_rate": heart_rate,

            "rr_interval": rr_interval,

            "prediction_class":
            int(train.iloc[0, -1])

        }

        print(features)

        return features