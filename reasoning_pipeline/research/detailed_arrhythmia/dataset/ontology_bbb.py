from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBBLabelOntology:
    version: str
    labels: tuple[str, ...]
    symbol_mapping: dict[str, str]

    def map_symbol(
        self,
        symbol: str,
    ) -> str | None:
        return self.symbol_mapping.get(symbol)


BBB_ONTOLOGY = BBBLabelOntology(
    version="mit_bih_detailed_bbb_v1",
    labels=(
        "N",
        "BBB",
        "A",
        "V",
        "F",
    ),
    symbol_mapping={
        "N": "N",
        "L": "BBB",
        "R": "BBB",
        "A": "A",
        "V": "V",
        "F": "F",
    },
)