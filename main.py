import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
INPUT_SIZE = 187
NUM_CLASSES = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def generate_synthetic_ecg(samples=6000):
    print("Generating synthetic ECG dataset...")

    x = np.random.normal(0, 0.25, (samples, INPUT_SIZE)).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, samples)

    time_axis = np.linspace(0, 1, INPUT_SIZE)

    for i in range(samples):
        label = y[i]

        if label == 0:
            signal = np.sin(2 * np.pi * 5 * time_axis)

        elif label == 1:
            signal = np.sin(2 * np.pi * 7 * time_axis)
            signal[50:60] += 1.2

        elif label == 2:
            signal = np.sin(2 * np.pi * 3 * time_axis)
            signal[90:110] += 1.8

        elif label == 3:
            signal = np.sin(2 * np.pi * 6 * time_axis)
            signal[120:135] -= 1.4

        else:
            signal = np.sin(2 * np.pi * 9 * time_axis)
            signal += np.random.normal(0, 0.4, INPUT_SIZE)

        x[i] += signal.astype(np.float32)

    return x, y.astype(np.int64)


def create_clients(x_train, y_train):
    indexes = np.arange(len(x_train))
    np.random.shuffle(indexes)

    splits = np.array_split(indexes, NUM_CLIENTS)
    clients = []

    for client_id, client_indexes in enumerate(splits, start=1):
        client_x = torch.tensor(x_train[client_indexes], dtype=torch.float32)
        client_y = torch.tensor(y_train[client_indexes], dtype=torch.long)

        dataset = TensorDataset(client_x, client_y)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        clients.append(loader)

        print(
            f"Hospital {client_id}: "
            f"{len(client_indexes)} private ECG records"
        )

    return clients


def train_local_model(global_model, train_loader):
    local_model = copy.deepcopy(global_model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    local_model.train()

    total_loss = 0.0

    for _ in range(LOCAL_EPOCHS):
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            predictions = local_model(features)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    average_loss = total_loss / max(len(train_loader) * LOCAL_EPOCHS, 1)

    return local_model.state_dict(), average_loss


def federated_average(client_weights):
    averaged_weights = copy.deepcopy(client_weights[0])

    for key in averaged_weights.keys():
        averaged_weights[key] = torch.zeros_like(averaged_weights[key])

        for weights in client_weights:
            averaged_weights[key] += weights[key]

        averaged_weights[key] /= len(client_weights)

    return averaged_weights


def evaluate_model(model, x_test, y_test):
    model.eval()

    features = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        predictions = model(features)
        predicted_classes = torch.argmax(predictions, dim=1)

    predicted_classes = predicted_classes.cpu().numpy()

    accuracy = accuracy_score(y_test, predicted_classes)
    f1 = f1_score(
        y_test,
        predicted_classes,
        average="weighted"
    )

    return accuracy, f1, predicted_classes


def save_confusion_matrix(y_test, predictions):
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Federated ECG Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))
    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()


def federated_learning_agent():
    print("=" * 60)
    print("FEDERATED ECG LEARNING AGENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("Raw patient data will remain inside simulated hospitals.")
    print()

    x, y = generate_synthetic_ecg()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    clients = create_clients(x_train, y_train)

    global_model = ECGModel().to(DEVICE)

    training_history = []

    for round_number in range(1, NUM_ROUNDS + 1):
        print()
        print(f"Federated Round {round_number}/{NUM_ROUNDS}")

        client_weights = []
        client_losses = []

        for client_id, client_loader in enumerate(clients, start=1):
            weights, loss = train_local_model(
                global_model,
                client_loader
            )

            client_weights.append(weights)
            client_losses.append(loss)

            print(
                f"Hospital {client_id} local loss: "
                f"{loss:.4f}"
            )

        global_weights = federated_average(client_weights)
        global_model.load_state_dict(global_weights)

        accuracy, f1, _ = evaluate_model(
            global_model,
            x_test,
            y_test
        )

        round_result = {
            "round": round_number,
            "average_client_loss": float(np.mean(client_losses)),
            "global_accuracy": float(accuracy),
            "global_f1_score": float(f1),
        }

        training_history.append(round_result)

        print(f"Global accuracy: {accuracy * 100:.2f}%")
        print(f"Global F1 score: {f1 * 100:.2f}%")

    final_accuracy, final_f1, predictions = evaluate_model(
        global_model,
        x_test,
        y_test
    )

    torch.save(
        global_model.state_dict(),
        "models/federated_ecg_model.pth"
    )

    save_confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    final_results = {
        "number_of_clients": NUM_CLIENTS,
        "federated_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "final_accuracy": float(final_accuracy),
        "final_f1_score": float(final_f1),
        "training_history": training_history,
        "classification_report": report,
    }

    with open(
        "results/federated_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(final_results, file, indent=4)

    print()
    print("=" * 60)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 60)
    print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Final F1 Score: {final_f1 * 100:.2f}%")
    print()
    print("Saved files:")
    print("models/federated_ecg_model.pth")
    print("results/federated_results.json")
    print("results/confusion_matrix.png")


if __name__ == "__main__":
    federated_learning_agent()
