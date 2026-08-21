"""
verify_audit.py

Checks the n8n governance audit log against the exact contract that
Healthcare-project/scripts/import-governance-audit.mjs expects.

Every key checked below is one the import script actually reads. A missing
or misnamed key does NOT raise an error at import time -- it silently
becomes null, false, or "Low". This script surfaces those before they
reach Supabase.

Usage:
    python verify_audit.py [path_to_audit_log.json]
"""

import json
import sys
from collections import Counter

DEFAULT_PATH = "Healthcare-project/scripts/ecg_audit_log.json"

# Keys read by import-governance-audit.mjs, grouped by failure mode.

# Missing -> column becomes NULL. Visible as blank in the dashboard.
NULLABLE_KEYS = [
    "sample_index",
    "true_class_aami",
    "prediction",
    "confidence",
    "calibrated_confidence",
    "uncertainty_level",
    "normalized_entropy",
    "dominant_ecg_region",
    "urgency_level",
    "suggested_next_action",
    "review_reason",
    "governance_policy",
    "route",                 # NB: maps to governance_route column
    "clinical_review_status",
    "review_mode",
    "hitl_released_at",
]

# Missing -> column becomes FALSE. Indistinguishable from a real false.
BOOLEAN_KEYS = [
    "human_review_required",
    "low_confidence_flag",
    "disagreement_flag",
    "blind_class_risk",
]

# Missing -> every record marked incorrect / Low risk. Silent and severe.
CRITICAL_KEYS = [
    "ground_truth_miss",
    "risk_score",
]

REQUIRED_KEYS = ["patient_id"]

# Read by the schema but hardcoded to null in the import script.
DROPPED_BY_IMPORT = ["exclusion_statement"]

CATEGORICAL_KEYS = [
    "uncertainty_level",
    "route",
    "clinical_review_status",
    "review_mode",
    "governance_policy",
]

problems = []
warnings = []


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    if "records" not in audit:
        print(f"FAIL: no top-level 'records' key in {path}.")
        print("      The import script does audit.records.map(...) and will crash.")
        sys.exit(1)
    return audit


def check_presence(records, keys, label):
    print(f"\n{label}")
    total = len(records)
    for key in keys:
        missing = sum(1 for r in records if key not in r or r.get(key) is None)
        if missing == total:
            print(f"  [MISSING]  {key:26s} absent in ALL {total} records")
            problems.append(f"{key} missing from every record")
        elif missing:
            print(f"  [PARTIAL]  {key:26s} absent in {missing}/{total}")
            warnings.append(f"{key} missing in {missing} records")
        else:
            print(f"  [ok]       {key:26s} present in all {total}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    audit = load(path)
    records = audit["records"]
    total = len(records)

    print(f"Loaded {total} records from {path}")
    if total != 250:
        print(f"  WARNING: expected 250 records, found {total}")
        warnings.append(f"record count is {total}, not 250")

    if "generated_at" not in audit:
        warnings.append("no top-level generated_at (falls back to import time)")

    check_presence(records, REQUIRED_KEYS, "REQUIRED")
    check_presence(records, CRITICAL_KEYS,
                   "CRITICAL -- missing here corrupts accuracy and risk_flag")
    check_presence(records, BOOLEAN_KEYS,
                   "BOOLEANS -- missing here silently becomes false")
    check_presence(records, NULLABLE_KEYS,
                   "NULLABLE -- missing here shows blank in the dashboard")

    # ground_truth_miss must be a real boolean for `correct` to be meaningful.
    print("\nGROUND TRUTH")
    gtm_types = {type(r.get("ground_truth_miss")).__name__ for r in records}
    print(f"  ground_truth_miss types: {gtm_types}")
    if gtm_types != {"bool"}:
        print("  FAIL: not all booleans. `correct: item.ground_truth_miss === false`")
        print("        uses strict equality -- non-bool values mark records INCORRECT.")
        problems.append("ground_truth_miss is not consistently boolean")
    misses = sum(1 for r in records if r.get("ground_truth_miss") is True)
    print(f"  misclassifications in audit: {misses}/{total}")

    # risk_score drives the risk_flag tiers in the import script.
    print("\nRISK SCORE")
    scores = [r.get("risk_score") for r in records
              if isinstance(r.get("risk_score"), (int, float))]
    if len(scores) != total:
        print(f"  FAIL: only {len(scores)}/{total} have numeric risk_score.")
        print("        Non-numeric values silently become risk_flag='Low'.")
        problems.append("risk_score not numeric in all records")
    if scores:
        print(f"  range: {min(scores):.4f} to {max(scores):.4f}")
        tiers = Counter(
            "High" if s >= 0.45 else "Medium" if s >= 0.25 else "Low"
            for s in scores
        )
        print(f"  risk_flag tiers the import will assign: {dict(tiers)}")

    # Confidence scale: 0-1 vs 0-100 is the classic silent mismatch.
    print("\nCONFIDENCE SCALES")
    for key in ["confidence", "calibrated_confidence", "normalized_entropy"]:
        vals = [r.get(key) for r in records
                if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        print(f"  {key:24s} {lo:.4f} to {hi:.4f}", end="")
        if hi > 1.0:
            print("   <-- above 1.0, check the dashboard expects 0-100")
            warnings.append(f"{key} exceeds 1.0")
        else:
            print()

    # Exact string values matter -- the columns have no check constraints,
    # so a case mismatch is accepted by Postgres and shows as nothing.
    print("\nCATEGORICAL VALUES (must match the dashboard's expected strings)")
    for key in CATEGORICAL_KEYS:
        vals = [r.get(key) for r in records if isinstance(r.get(key), str)]
        if vals:
            print(f"  {key}: {dict(Counter(vals))}")
        else:
            print(f"  {key}: (none present)")

    # The headline governance result.
    print("\nGOVERNANCE COUNTS -- compare these against Supabase after import")
    for key in BOOLEAN_KEYS:
        true_count = sum(1 for r in records if r.get(key) is True)
        print(f"  {key:26s} true in {true_count}/{total}")

    caught = sum(1 for r in records
                 if r.get("ground_truth_miss") is True
                 and r.get("human_review_required") is True)
    if misses:
        print(f"\n  misclassifications caught by review routing: "
              f"{caught}/{misses} ({100 * caught / misses:.1f}%)")

    # Fields present in your output that the import silently discards.
    print("\nDROPPED BY THE IMPORT SCRIPT")
    for key in DROPPED_BY_IMPORT:
        present = sum(1 for r in records if r.get(key) not in (None, ""))
        if present:
            print(f"  {key}: populated in {present}/{total} records, but the")
            print(f"      import hardcodes this column to null. Data is lost.")
            problems.append(f"{key} populated but discarded by import script")
        else:
            print(f"  {key}: not populated in the audit output either")

    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    if not problems and not warnings:
        print("All checks passed. Audit output matches the import contract.")


if __name__ == "__main__":
    main()
