from types import MappingProxyType

CLASS_LABELS = MappingProxyType(
    {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q",
    }
)


def get_class_label(class_index: int) -> str:
    try:
        return CLASS_LABELS[class_index]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ECG class index: {class_index}"
        ) from exc
