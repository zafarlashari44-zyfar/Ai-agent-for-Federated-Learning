import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


class FederatedECGModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        hidden_size_1 = min(256, max(64, input_size))
        hidden_size_2 = min(128, max(32, input_size // 2))

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(0.25),

            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(hidden_size_2, num_classes),
        )

    def forward(self, features):
        return self.network(features)


def clean_dataset(dataframe):
    dataframe = dataframe.copy()

    dataframe = dataframe.dropna(axis=1, how="all")
    dataframe = dataframe.dropna(axis=0, how="all")

    if dataframe.empty:
        raise ValueError("Dataset empty hai.")

    label_column = dataframe.columns[-1]

    features = dataframe.iloc[:, :-1].copy()
    labels = dataframe.iloc[:, -1].copy()

    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median())
    features = features.fillna(0)

    valid_rows = labels.notna()
    features = features.loc[valid_rows]
    labels = labels.loc[valid_rows]

    if len(features) < 20:
        raise ValueError("Dataset mein kam az kam 20 valid records chahiye.")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels.astype(str))

    if len(label_encoder.classes_) < 2:
        raise ValueError("Dataset mein kam az kam 2 classes honi chahiye.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return (
        scaled_features.astype(np.float32),
        encoded_labels.astype(np.int64),
        label_column,
        label_encoder,
    )


def create_federated_clients(
    features,
    labels,
    number_of_clients,
    batch_size,
):
    indexes = np.arange(len(features))
    np.random.shuffle(indexes)

    client_indexes = np.array_split(indexes, number_of_clients)
    clients = []

    for indexes_for_client in client_indexes:
        client_features = torch.tensor(
            features[indexes_for_client],
            dtype=torch.float32,
        )

        client_labels = torch.tensor(
            labels[indexes_for_client],
            dtype=torch.long,
        )

        dataset = TensorDataset(
            client_features,
            client_labels,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        clients.append(
            {
                "loader": loader,
                "records": len(indexes_for_client),
            }
        )

    return clients


def train_local_client(
    global_model,
    train_loader,
    local_epochs,
    learning_rate,
):
    local_model = copy.deepcopy(global_model).to(DEVICE)

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    local_model.train()

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / max(total_batches, 1)

    return (
        copy.deepcopy(local_model.state_dict()),
        average_loss,
    )


def federated_average(client_results):
    total_records = sum(
        client["records"] for client in client_results
    )

    averaged_weights = {}

    first_state = client_results[0]["weights"]

    for parameter_name in first_state:
        averaged_weights[parameter_name] = torch.zeros_like(
            first_state[parameter_name]
        )

        for client in client_results:
            client_weight = client["records"] / total_records

            averaged_weights[parameter_name] += (
                client["weights"][parameter_name]
                * client_weight
            )

    return averaged_weights


def evaluate_model(model, features, labels):
    model.eval()

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(feature_tensor)
        predictions = torch.argmax(
            outputs,
            dim=1,
        )

    predictions = predictions.cpu().numpy()

    accuracy = accuracy_score(labels, predictions)

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return accuracy, weighted_f1, predictions


def create_confusion_matrix_figure(
    true_labels,
    predictions,
    class_names,
):
    matrix = confusion_matrix(
        true_labels,
        predictions,
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
    )

    figure.colorbar(image, ax=axis)

    axis.set_title("Federated Learning Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))

    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(class_names)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    return figure


def train_federated_agent(
    dataframe,
    number_of_clients,
    federated_rounds,
    local_epochs,
    batch_size,
    learning_rate,
    progress_callback,
):
    (
        features,
        labels,
        label_column,
        label_encoder,
    ) = clean_dataset(dataframe)

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )

    input_size = train_features.shape[1]
    number_of_classes = len(label_encoder.classes_)

    clients = create_federated_clients(
        train_features,
        train_labels,
        number_of_clients,
        batch_size,
    )

    global_model = FederatedECGModel(
        input_size=input_size,
        num_classes=number_of_classes,
    ).to(DEVICE)

    training_history = []

    for round_number in range(1, federated_rounds + 1):
        client_results = []
        client_losses = []

        for client_id, client in enumerate(clients, start=1):
            weights, loss = train_local_client(
                global_model=global_model,
                train_loader=client["loader"],
                local_epochs=local_epochs,
                learning_rate=learning_rate,
            )

            client_results.append(
                {
                    "client_id": client_id,
                    "weights": weights,
                    "records": client["records"],
                    "loss": loss,
                }
            )

            client_losses.append(loss)

        aggregated_weights = federated_average(client_results)

        global_model.load_state_dict(aggregated_weights)

        accuracy, f1, _ = evaluate_model(
            global_model,
            test_features,
            test_labels,
        )

        round_result = {
            "round": round_number,
            "average_client_loss": float(
                np.mean(client_losses)
            ),
            "global_accuracy": float(accuracy),
            "global_f1_score": float(f1),
        }

        training_history.append(round_result)

        progress_callback(
            round_number,
            federated_rounds,
            round_result,
        )

    final_accuracy, final_f1, predictions = evaluate_model(
        global_model,
        test_features,
        test_labels,
    )

    class_names = [
        str(class_name)
        for class_name in label_encoder.classes_
    ]

    report = classification_report(
        test_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "input_size": input_size,
            "number_of_classes": number_of_classes,
            "class_names": class_names,
        },
        "models/federated_ecg_model.pth",
    )

    results = {
        "dataset_records": int(len(features)),
        "training_records": int(len(train_features)),
        "testing_records": int(len(test_features)),
        "input_features": int(input_size),
        "label_column": str(label_column),
        "classes": class_names,
        "number_of_clients": number_of_clients,
        "federated_rounds": federated_rounds,
        "local_epochs": local_epochs,
        "final_accuracy": float(final_accuracy),
        "final_f1_score": float(final_f1),
        "training_history": training_history,
        "classification_report": report,
    }

    with open(
        "results/federated_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    confusion_figure = create_confusion_matrix_figure(
        test_labels,
        predictions,
        class_names,
    )

    confusion_figure.savefig(
        "results/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    return {
        "results": results,
        "clients": clients,
        "history": pd.DataFrame(training_history),
        "confusion_figure": confusion_figure,
    }


st.set_page_config(
    page_title="Federated ECG Learning Agent",
    page_icon="🫀",
    layout="wide",
)

st.title("Federated ECG Learning Agent")

st.write(
    "Upload an ECG CSV dataset. Each simulated hospital trains "
    "locally, while the central server aggregates only model weights."
)

st.warning(
    "Dataset ki last column class label honi chahiye. "
    "Baaki columns numeric ECG features honi chahiye."
)

with st.sidebar:
    st.header("Training Configuration")

    number_of_clients = st.slider(
        "Simulated hospitals",
        min_value=2,
        max_value=10,
        value=3,
    )

    federated_rounds = st.slider(
        "Federated rounds",
        min_value=1,
        max_value=20,
        value=5,
    )

    local_epochs = st.slider(
        "Local epochs",
        min_value=1,
        max_value=10,
        value=2,
    )

    batch_size = st.selectbox(
        "Batch size",
        options=[16, 32, 64, 128],
        index=2,
    )

    learning_rate = st.selectbox(
        "Learning rate",
        options=[0.0001, 0.0005, 0.001, 0.005],
        index=2,
    )

    st.write(f"Device: `{DEVICE}`")


uploaded_file = st.file_uploader(
    "Upload ECG CSV dataset",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )

        column_1, column_2, column_3 = st.columns(3)

        column_1.metric(
            "Records",
            f"{len(dataframe):,}",
        )

        column_2.metric(
            "Feature columns",
            max(len(dataframe.columns) - 1, 0),
        )

        column_3.metric(
            "Label column",
            str(dataframe.columns[-1]),
        )

        label_counts = dataframe.iloc[:, -1].value_counts()

        st.subheader("Class Distribution")
        st.bar_chart(label_counts)

        start_training = st.button(
            "Start Federated Training",
            type="primary",
            use_container_width=True,
        )

        if start_training:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            round_placeholder = st.empty()

            def update_progress(
                current_round,
                total_rounds,
                round_result,
            ):
                progress = current_round / total_rounds
                progress_bar.progress(progress)

                status_placeholder.info(
                    f"Federated round {current_round} "
                    f"of {total_rounds} completed"
                )

                round_placeholder.write(
                    {
                        "round": current_round,
                        "accuracy": (
                            f"{round_result['global_accuracy'] * 100:.2f}%"
                        ),
                        "f1_score": (
                            f"{round_result['global_f1_score'] * 100:.2f}%"
                        ),
                        "average_client_loss": (
                            f"{round_result['average_client_loss']:.4f}"
                        ),
                    }
                )

            with st.spinner(
                "Hospitals are training locally..."
            ):
                output = train_federated_agent(
                    dataframe=dataframe,
                    number_of_clients=number_of_clients,
                    federated_rounds=federated_rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0)
            status_placeholder.success(
                "Federated training completed."
            )

            results = output["results"]

            st.subheader("Final Performance")

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)

            metric_1.metric(
                "Accuracy",
                f"{results['final_accuracy'] * 100:.2f}%",
            )

            metric_2.metric(
                "Weighted F1",
                f"{results['final_f1_score'] * 100:.2f}%",
            )

            metric_3.metric(
                "Hospitals",
                results["number_of_clients"],
            )

            metric_4.metric(
                "Federated rounds",
                results["federated_rounds"],
            )

            st.subheader("Hospital Data Distribution")

            hospital_data = pd.DataFrame(
                {
                    "Hospital": [
                        f"Hospital {index + 1}"
                        for index in range(len(output["clients"]))
                    ],
                    "Private records": [
                        client["records"]
                        for client in output["clients"]
                    ],
                }
            )

            st.dataframe(
                hospital_data,
                use_container_width=True,
            )

            st.subheader("Training History")

            history = output["history"].copy()

            history["global_accuracy"] = (
                history["global_accuracy"] * 100
            )

            history["global_f1_score"] = (
                history["global_f1_score"] * 100
            )

            st.line_chart(
                history.set_index("round")[
                    [
                        "global_accuracy",
                        "global_f1_score",
                    ]
                ]
            )

            st.dataframe(
                history,
                use_container_width=True,
            )

            st.subheader("Confusion Matrix")

            st.pyplot(
                output["confusion_figure"],
                use_container_width=False,
            )

            with open(
                "results/federated_results.json",
                "rb",
            ) as file:
                st.download_button(
                    "Download Results JSON",
                    data=file,
                    file_name="federated_results.json",
                    mime="application/json",
                )

            with open(
                "models/federated_ecg_model.pth",
                "rb",
            ) as file:
                st.download_button(
                    "Download Federated Model",
                    data=file,
                    file_name="federated_ecg_model.pth",
                    mime="application/octet-stream",
                )

            st.success(
                "Raw records remained inside simulated hospital "
                "partitions. Only model parameters were aggregated."
            )

    except Exception as error:
        st.error(f"Dataset error: {error}")

else:
    st.info(
        "ECG CSV upload karo. Training controls sidebar mein hain."
    )