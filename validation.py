"""Data-quality gate: reports issues, never raises on bad data."""
from generator import CHANNELS

REQUIRED = ["week", "sales"] + [f"spend_{c}" for c in CHANNELS]


def validate(rows):
    issues = []
    seen_weeks = set()
    prev_week = None
    for idx, r in enumerate(rows):
        for field in REQUIRED:
            if field not in r:
                issues.append(f"row {idx}: missing field {field}")
        if len(issues) > 50:
            break
        if "week" in r:
            if r["week"] in seen_weeks:
                issues.append(f"row {idx}: duplicate week {r['week']}")
            seen_weeks.add(r["week"])
            if prev_week is not None and r["week"] != prev_week + 1:
                issues.append(f"row {idx}: week gap {prev_week} -> {r['week']}")
            prev_week = r["week"]
        if "sales" in r and isinstance(r["sales"], (int, float)) and r["sales"] < 0:
            issues.append(f"row {idx}: negative sales")
        for c in CHANNELS:
            v = r.get(f"spend_{c}")
            if isinstance(v, (int, float)) and v < 0:
                issues.append(f"row {idx}: negative spend_{c}")
    return issues
