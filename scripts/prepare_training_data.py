import csv
import io
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com/repos/Azure/Azure-Sentinel/contents"
TARGET_PATH = "Solutions/Training/Azure-Sentinel-Training-Lab/Artifacts/Telemetry"
RAW_BASE = "https://raw.githubusercontent.com/Azure/Azure-Sentinel/master"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RANDOM_SEED = 42

KEYWORD_RULES: List[Tuple[str, str, str]] = [
    ("security", "security_event", "suspicious"),
    ("solargate", "threat_ioc", "suspicious"),
    ("evasion", "threat_evasion", "suspicious"),
    ("hunting", "threat_hunting", "suspicious"),
    ("highrisk", "high_risk_indicator", "suspicious"),
    ("pentest", "network_scan", "suspicious"),
    ("disable_accounts", "identity_admin_change", "suspicious"),
    ("inbox_rule", "mailbox_rule_change", "suspicious"),
    ("sign-in_adelete", "signin", "suspicious"),
    ("signin", "signin", "normal"),
    ("sign-in", "signin", "normal"),
    ("audit", "audit", "normal"),
    ("office", "office_activity", "normal"),
    ("azure", "azure_activity", "normal"),
    ("abap", "sap_abap", "normal"),
]

MEDIUM_MAP: Dict[str, str] = {
    "security_event": "identity_attack",
    "signin": "identity_attack",
    "identity_admin_change": "privilege_escalation",
    "mailbox_rule_change": "privilege_escalation",
    "threat_evasion": "privilege_escalation",
    "threat_hunting": "network_activity",
    "network_scan": "network_activity",
    "threat_ioc": "network_activity",
    "high_risk_indicator": "network_activity",
    "azure_activity": "benign_activity",
    "office_activity": "benign_activity",
    "audit": "benign_activity",
    "sap_abap": "benign_activity",
}


def github_json(url: str):
    req = Request(url, headers={"User-Agent": "azure-log-analytic-ai-prep"})
    with urlopen(req, timeout=30) as resp:
        return json.load(resp)


def download_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "azure-log-analytic-ai-prep"})
    with urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def clean_cell(value: str) -> str:
    return (value or "").replace("\ufeff", "").strip()


def infer_labels(filename: str) -> Tuple[str, str]:
    lower = filename.lower()
    for kw, log_type, risk in KEYWORD_RULES:
        if kw in lower:
            return log_type, risk
    return "other", "unknown"


def row_to_text(row: Dict[str, str], priority_fields: List[str]) -> str:
    parts: List[str] = []
    used = set()

    for key in priority_fields:
        if key in row and row[key] not in (None, ""):
            parts.append(f"{key}={row[key]}")
            used.add(key)

    for key, val in row.items():
        if key in used:
            continue
        if val in (None, ""):
            continue
        parts.append(f"{key}={val}")

    text = " | ".join(parts).strip()
    return text[:4000]


def fetch_csv_files() -> List[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api_url = f"{GITHUB_API}/{TARGET_PATH}"
    items = github_json(api_url)

    csv_paths: List[Path] = []
    for item in items:
        if item.get("type") != "file":
            continue
        name = item.get("name", "")
        if not name.lower().endswith(".csv"):
            continue

        raw_url = item.get("download_url") or f"{RAW_BASE}/{TARGET_PATH}/{name}"
        local_path = RAW_DIR / safe_name(name)
        try:
            content = download_text(raw_url)
        except (HTTPError, URLError) as exc:
            print(f"WARN: failed to download {name}: {exc}", file=sys.stderr)
            continue

        local_path.write_text(content, encoding="utf-8")
        csv_paths.append(local_path)

    return csv_paths


def parse_csv(path: Path) -> List[Dict[str, str]]:
    def parse_with_dict_reader(content: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        stream = io.StringIO(content)
        sample = content[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            return out
        for raw in reader:
            row = {clean_cell(str(k)): clean_cell("" if v is None else str(v)) for k, v in raw.items() if k is not None}
            if any(row.values()):
                out.append(row)
        return out

    def parse_with_plain_reader(content: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        reader = csv.reader(io.StringIO(content))
        try:
            headers = [clean_cell(h) for h in next(reader)]
        except StopIteration:
            return out

        for values in reader:
            if not values:
                continue
            normalized = [clean_cell(v) for v in values]
            row = {headers[i]: normalized[i] if i < len(normalized) else "" for i in range(len(headers))}
            if any(row.values()):
                out.append(row)
        return out

    content = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_with_dict_reader(content)
    if not parsed:
        return parsed

    fieldnames = list(parsed[0].keys())
    if len(fieldnames) == 1 and "," in fieldnames[0]:
        reparsed = parse_with_plain_reader(content)
        if reparsed:
            return reparsed

    return parsed


def build_dataset(csv_files: List[Path]) -> List[Dict[str, str]]:
    dataset: List[Dict[str, str]] = []

    priority = [
        "TimeGenerated",
        "Timestamp",
        "EventTime",
        "OperationName",
        "ActivityDisplayName",
        "ActionType",
        "ResultType",
        "UserPrincipalName",
        "IPAddress",
        "Computer",
        "Account",
    ]

    for path in csv_files:
        log_type, risk_label = infer_labels(path.name)
        rows = parse_csv(path)

        for idx, row in enumerate(rows):
            text = row_to_text(row, priority)
            if not text:
                continue

            dataset.append(
                {
                    "id": f"{path.stem}_{idx}",
                    "source_file": path.name,
                    "log_type": log_type,
                    "risk_label": risk_label,
                    "text": text,
                    "target": f"{log_type}__{risk_label}",
                }
            )

    return dataset


def split_dataset(rows: List[Dict[str, str]]) -> None:
    random.seed(RANDOM_SEED)
    random.shuffle(rows)

    n = len(rows)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    for i, row in enumerate(rows):
        if i < train_end:
            row["split"] = "train"
        elif i < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"


def write_outputs(rows: List[Dict[str, str]], csv_name: str, jsonl_name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = PROCESSED_DIR / csv_name
    jsonl_out = PROCESSED_DIR / jsonl_name

    fieldnames = sorted({k for row in rows for k in row.keys() if k != "text"})
    fieldnames.append("text")

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(rows: List[Dict[str, str]], summary_name: str, label_key: str) -> None:
    summary_path = PROCESSED_DIR / summary_name

    by_label: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    for row in rows:
        label = row.get(label_key, "missing")
        by_label[label] = by_label.get(label, 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1

    summary = {
        "total_rows": len(rows),
        "label_key": label_key,
        "labels": dict(sorted(by_label.items(), key=lambda x: (-x[1], x[0]))),
        "splits": by_split,
        "source": "Azure/Azure-Sentinel",
        "path": TARGET_PATH,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_binary_dataset(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if row["risk_label"] == "unknown":
            continue

        binary_label = "suspicious" if row["risk_label"] == "suspicious" else "normal"
        out.append(
            {
                **row,
                "binary_label": binary_label,
                "y": 1 if binary_label == "suspicious" else 0,
            }
        )
    return out


def build_medium_dataset(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        medium_label = MEDIUM_MAP.get(row["log_type"], "unknown")
        out.append({**row, "medium_label": medium_label})
    return out


def main() -> int:
    try:
        csv_files = fetch_csv_files()
    except Exception as exc:
        print(f"ERROR: failed to list/download files: {exc}", file=sys.stderr)
        return 1

    if not csv_files:
        print("ERROR: no CSV files fetched", file=sys.stderr)
        return 2

    dataset = build_dataset(csv_files)
    if not dataset:
        print("ERROR: dataset is empty after parsing", file=sys.stderr)
        return 3

    split_dataset(dataset)
    write_outputs(dataset, "training_dataset.csv", "training_dataset.jsonl")
    write_summary(dataset, "summary.json", "target")

    binary_dataset = build_binary_dataset(dataset)
    write_outputs(binary_dataset, "training_dataset_binary.csv", "training_dataset_binary.jsonl")
    write_summary(binary_dataset, "summary_binary.json", "binary_label")

    medium_dataset = build_medium_dataset(dataset)
    write_outputs(medium_dataset, "training_dataset_medium.csv", "training_dataset_medium.jsonl")
    write_summary(medium_dataset, "summary_medium.json", "medium_label")

    print(f"Fetched CSV files: {len(csv_files)}")
    print(f"Prepared rows: {len(dataset)}")
    print(f"Output: {PROCESSED_DIR / 'training_dataset.csv'}")
    print(f"Output: {PROCESSED_DIR / 'training_dataset.jsonl'}")
    print(f"Output: {PROCESSED_DIR / 'summary.json'}")
    print(f"Output: {PROCESSED_DIR / 'training_dataset_binary.csv'}")
    print(f"Output: {PROCESSED_DIR / 'training_dataset_binary.jsonl'}")
    print(f"Output: {PROCESSED_DIR / 'summary_binary.json'}")
    print(f"Output: {PROCESSED_DIR / 'training_dataset_medium.csv'}")
    print(f"Output: {PROCESSED_DIR / 'training_dataset_medium.jsonl'}")
    print(f"Output: {PROCESSED_DIR / 'summary_medium.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
