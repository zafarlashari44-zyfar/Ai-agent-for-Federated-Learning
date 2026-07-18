from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fl_ecg_orchestrator.agents.reasoning_agent import ReasoningAgent


def main() -> None:

    context = {
        "project_root": str(PROJECT_ROOT),
    }

    result = ReasoningAgent().run(context)

    print()
    print("=" * 60)
    print("Reasoning Agent")
    print("=" * 60)
    print(result.to_dict())


if __name__ == "__main__":
    main()
