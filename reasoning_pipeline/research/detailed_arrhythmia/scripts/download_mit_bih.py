import argparse
from pathlib import Path

import wfdb

from research.detailed_arrhythmia.scripts.run_baseline import MIT_BIH_RECORDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    arguments.destination.mkdir(parents=True, exist_ok=True)
    files = [
        f"{record_id}.{extension}"
        for record_id in MIT_BIH_RECORDS
        for extension in ("hea", "dat", "atr")
    ]
    wfdb.dl_files(
        "mitdb",
        str(arguments.destination),
        files,
        keep_subdirs=False,
        overwrite=False,
    )


if __name__ == "__main__":
    main()
