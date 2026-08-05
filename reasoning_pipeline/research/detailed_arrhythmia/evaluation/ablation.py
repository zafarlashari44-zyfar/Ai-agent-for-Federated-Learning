from typing import Any


def compare_experiments(
    weighted_only: dict[str, Any],
    augmented: dict[str, Any],
) -> dict[str, Any]:
    labels = tuple(weighted_only["per_class"])
    class_changes = {
        label: {
            "f1_change": (
                augmented["per_class"][label]["f1"]
                - weighted_only["per_class"][label]["f1"]
            ),
            "recall_change": (
                augmented["per_class"][label]["recall"]
                - weighted_only["per_class"][label]["recall"]
            ),
            "false_negative_change": (
                augmented["per_class"][label]["false_negatives"]
                - weighted_only["per_class"][label]["false_negatives"]
            ),
        }
        for label in labels
    }
    return {
        "macro_f1_change": augmented["macro_f1"] - weighted_only["macro_f1"],
        "weighted_f1_change": (
            augmented["weighted_f1"] - weighted_only["weighted_f1"]
        ),
        "balanced_accuracy_change": (
            augmented["balanced_accuracy"] - weighted_only["balanced_accuracy"]
        ),
        "augmentation_improved_macro_f1": (
            augmented["macro_f1"] > weighted_only["macro_f1"]
        ),
        "per_class_changes": class_changes,
    }
