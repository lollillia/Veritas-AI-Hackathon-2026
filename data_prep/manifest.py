"""Reading and writing the split manifest (data/splits.csv)."""
import csv
from pathlib import Path

COLUMNS = ["path", "md5", "source", "original_folder", "label", "group_id", "split"]
DROPPED_COLUMNS = ["path", "md5", "reason", "duplicate_of"]


def _write(columns: list[str], rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({c: row[c] for c in columns} for row in rows)


def write_manifest(rows: list[dict], path: Path) -> None:
    _write(COLUMNS, rows, path)


def write_dropped(rows: list[dict], path: Path) -> None:
    _write(DROPPED_COLUMNS, rows, path)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["group_id"] = int(row["group_id"])
    return rows


def samples_for_split(rows: list[dict], split: str, root: Path) -> list[tuple[Path, str]]:
    """(absolute image path, label) for one split, sorted by path.

    Raises FileNotFoundError naming the first missing file (by sorted path).
    """
    chosen = sorted((r for r in rows if r["split"] == split), key=lambda r: r["path"])
    for r in chosen:
        if not (Path(root) / r["path"]).is_file():
            raise FileNotFoundError(f"manifest file missing: {r['path']}")
    return [(Path(root) / r["path"], r["label"]) for r in chosen]
