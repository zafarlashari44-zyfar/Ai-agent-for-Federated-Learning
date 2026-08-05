from dataclasses import dataclass


@dataclass(frozen=True)
class DetailedLabelOntology:
    version: str
    labels: tuple[str, ...]
    symbol_to_label: tuple[tuple[str, str | None], ...]
    decisions: tuple[tuple[str, str], ...]

    @property
    def mapping(self) -> dict[str, str | None]:
        return dict(self.symbol_to_label)

    def map_symbol(self, symbol: str) -> str | None:
        return self.mapping.get(symbol)


DEFAULT_ONTOLOGY = DetailedLabelOntology(
    version="detailed-mit-bih-ontology-v1",
    labels=("N", "L", "R", "A", "V", "F"),
    symbol_to_label=(
        ("N", "N"),
        ("L", "L"),
        ("R", "R"),
        ("A", "A"),
        ("V", "V"),
        ("F", "F"),
        ("E", None),
        ("J", None),
        ("j", None),
        ("/", None),
    ),
    decisions=(
        ("N,L,R,A,V,F", "included for detailed, clinically interpretable evaluation"),
        ("E,J,j", "excluded pending sufficient beat and patient diversity"),
        ("/", "paced beats excluded pending patient-independent coverage audit"),
        ("other symbols", "excluded rather than merged into a heterogeneous class"),
    ),
)
