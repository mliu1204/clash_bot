import argparse
import csv
import hashlib
from collections.abc import Callable
from pathlib import Path


BASE_DATA_DIR = Path("data")
CHECKPOINT_DIR = BASE_DATA_DIR / "check_point_data"
SCRAPED_DATA_DIR = BASE_DATA_DIR / "scraped_data"

CHECKPOINT_DATE_SRC = "2_25"
CHECKPOINT_DATE_DST = "3_3_2"

ALL_BATTLE_META_FILENAME = "all_battle_meta_data.csv"
ALL_WORKER_ROWS_FILENAME = "all_worker_rows.csv"

SCRAPED_BATTLE_META_PATH = SCRAPED_DATA_DIR / "battle_meta_data.csv"
BATTLE_CHUNKS_DIR = SCRAPED_DATA_DIR / "battle_chunks"
WORKER_0_PATH = BATTLE_CHUNKS_DIR / "worker_0_results.csv"
WORKER_1_PATH = BATTLE_CHUNKS_DIR / "worker_1_results.csv"

HEADERS_CSV = Path("data_scripts") / "headers.csv"
HEADERS_TIGHT_CSV = Path("data_scripts") / "headers_tight.csv"
PROPER_META_FILENAME = "proper_meta.csv"
PROPER_META_TIGHT_FILENAME = "proper_meta_tight.csv"


def _stable_row_hash64(row: list[str]) -> int:
    # Stable, low-collision 64-bit hash for deduping rows.
    h = hashlib.blake2b(digest_size=8)
    h.update("\x1f".join(row).encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest(), byteorder="big", signed=False)


def append_csv(
    source_path: Path | None,
    append_paths: list[Path],
    output_path: Path,
    *,
    dedupe_key_column: str | None = None,
    dedupe_key_cleaner: Callable[[object], object] | None = None,
) -> None:
    """
    Create an updated checkpoint CSV and write to `output_path`.
    - If `source_path` is set: take all rows from it, then append rows from each path in `append_paths`.
    - If `source_path` is None (scraped-only): merge only from `append_paths`; header from first existing path.
    Assumes all CSVs share the same header.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[object] | None = set() if dedupe_key_column else None
    header_ref_path = source_path  # for error messages

    def maybe_write_row(writer: csv.writer, header: list[str], row: list[str]) -> None:
        if not seen:
            writer.writerow(row)
            return
        try:
            idx = header.index(dedupe_key_column)  # type: ignore[arg-type]
        except ValueError:
            name = (header_ref_path or append_paths[0]).name if append_paths else "?"
            raise RuntimeError(f"Column {dedupe_key_column!r} not found in {name}")
        if idx >= len(row):
            return
        key: object = row[idx]
        if dedupe_key_cleaner:
            key = dedupe_key_cleaner(key)  # type: ignore[misc]
        if not key:
            return
        if key in seen:
            return
        seen.add(key)
        writer.writerow(row)

    with output_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)

        if source_path is not None and source_path.exists():
            with source_path.open("r", newline="", encoding="utf-8") as src_file:
                src_reader = csv.reader(src_file)
                header = next(src_reader)
                writer.writerow(header)
                for row in src_reader:
                    maybe_write_row(writer, header, row)
        else:
            # Scraped-only: get header from first existing path in append_paths
            first = next((p for p in append_paths if p.exists()), None)
            if first is None:
                raise FileNotFoundError(
                    f"No scraped files found among {[str(p) for p in append_paths]}"
                )
            with first.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            writer.writerow(header)

        for path in append_paths:
            if not path.exists():
                continue
            with path.open("r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                _ = next(r, None)  # drop header
                for row in r:
                    maybe_write_row(writer, header, row)


def _clean_replay_tag(tag: str) -> str | None:
    """
    Normalize a replayTag value to a canonical battle id:
    - Strip whitespace.
    - Remove all '#' characters (replay tags are like '#08YPULQYRPP9').
    Returns None if the cleaned value is empty.
    """
    if tag is None:
        return None
    cleaned = tag.strip().replace("#", "")
    return cleaned or None


def create_paired_files(
    meta_path: Path,
    worker_path: Path,
    paired_meta_out: Path,
    paired_worker_out: Path,
) -> None:
    """
    From the combined checkpoint CSVs, create two new files that only
    contain rows whose ids exist in both:
    - Use cleaned replayTag from meta_path.
    - Use battle_id from worker_path.
    """
    # Collect ids from meta (using cleaned replayTag)
    meta_ids: set[str] = set()
    with meta_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            replay_idx = header.index("replayTag")
        except ValueError:
            raise RuntimeError("Column 'replayTag' not found in battle meta data CSV")

        for row in reader:
            if replay_idx >= len(row):
                continue
            cleaned = _clean_replay_tag(row[replay_idx])
            if cleaned:
                meta_ids.add(cleaned)

    # Collect ids from worker (battle_id)
    worker_ids: set[str] = set()
    with worker_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            battle_id_idx = header.index("battle_id")
        except ValueError:
            raise RuntimeError("Column 'battle_id' not found in worker rows CSV")

        for row in reader:
            if battle_id_idx >= len(row):
                continue
            bid = row[battle_id_idx].strip()
            if bid:
                worker_ids.add(bid)

    # Intersection of ids present in both
    common_ids = meta_ids & worker_ids

    # Write filtered meta rows
    paired_meta_out.parent.mkdir(parents=True, exist_ok=True)
    seen_meta_ids: set[str] = set()
    with meta_path.open("r", newline="", encoding="utf-8") as src, paired_meta_out.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        try:
            replay_idx = header.index("replayTag")
        except ValueError:
            raise RuntimeError("Column 'replayTag' not found in battle meta data CSV")
        writer.writerow(header)
        for row in reader:
            if replay_idx >= len(row):
                continue
            cleaned = _clean_replay_tag(row[replay_idx])
            if cleaned and cleaned in common_ids and cleaned not in seen_meta_ids:
                seen_meta_ids.add(cleaned)
                writer.writerow(row)

    # Write filtered worker rows
    seen_worker_rows: set[int] = set()
    with worker_path.open("r", newline="", encoding="utf-8") as src, paired_worker_out.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        try:
            battle_id_idx = header.index("battle_id")
        except ValueError:
            raise RuntimeError("Column 'battle_id' not found in worker rows CSV")
        writer.writerow(header)
        for row in reader:
            if battle_id_idx >= len(row):
                continue
            bid = row[battle_id_idx].strip()
            if not (bid and bid in common_ids):
                continue
            row_hash = _stable_row_hash64(row)
            if row_hash in seen_worker_rows:
                continue
            seen_worker_rows.add(row_hash)
            writer.writerow(row)


def split_meta_by_mode(meta_path: Path, by_modes_dir: Path) -> None:
    """
    Split a meta CSV into per-mode files under by_modes_dir, using the
    `gameMode_name` column for the filename (e.g. 'Ladder.csv').
    Also normalizes replayTag values to cleaned ids when writing.
    """
    by_modes_dir.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            mode_idx = header.index("type")
        except ValueError:
            raise RuntimeError("Column 'type' not found in battle meta data CSV")
        try:
            replay_idx = header.index("replayTag")
        except ValueError:
            replay_idx = -1

        writers: dict[str, csv.writer] = {}
        files: dict[str, object] = {}

        for row in reader:
            if mode_idx >= len(row):
                continue
            mode = row[mode_idx].strip()
            if not mode:
                mode = "UNKNOWN"

            if replay_idx >= 0 and replay_idx < len(row):
                cleaned = _clean_replay_tag(row[replay_idx])
                row[replay_idx] = cleaned or ""

            writer = writers.get(mode)
            if writer is None:
                out_path = by_modes_dir / f"{mode}.csv"
                fh = out_path.open("w", newline="", encoding="utf-8")
                files[mode] = fh
                writer = csv.writer(fh)
                writers[mode] = writer
                writer.writerow(header)

            writer.writerow(row)

    # Close all opened files
    for fh in files.values():
        fh.close()


MAX_BAD_ROWS_SAVED = 100


def check_csv_integrity(
    csv_path: Path,
    bad_rows_path: Path | None = None,
    good_rows_path: Path | None = None,
    max_bad_rows_to_save: int = MAX_BAD_ROWS_SAVED,
) -> tuple[bool, int | None, int, int, int | None]:
    """
    Validate that every data row has the same number of columns as the header.
    Returns (ok, expected_cols, data_rows, bad_rows, most_common_bad_cols).
    If bad_rows_path is provided, writes up to max_bad_rows_to_save bad rows (header + data) there.
    """
    if not csv_path.exists():
        return False, None, 0, 0, None

    expected_cols: int | None = None
    total_rows = 0
    bad_rows = 0
    bad_col_counts: dict[int, int] = {}
    bad_rows_written = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return False, None, 0, 0, None
        expected_cols = len(header)

        bad_writer: csv.writer | None = None
        good_writer: csv.writer | None = None
        if bad_rows_path is not None:
            bad_rows_path.parent.mkdir(parents=True, exist_ok=True)
            bad_file = bad_rows_path.open("w", newline="", encoding="utf-8")
            try:
                bad_writer = csv.writer(bad_file)
                bad_writer.writerow(header)
            except Exception:
                bad_file.close()
                bad_writer = None

        if good_rows_path is not None:
            good_rows_path.parent.mkdir(parents=True, exist_ok=True)
            good_file = good_rows_path.open("w", newline="", encoding="utf-8")
            try:
                good_writer = csv.writer(good_file)
                good_writer.writerow(header)
            except Exception:
                good_file.close()
                good_writer = None

        for row_num, row in enumerate(reader, start=2):  # 1 is header
            total_rows += 1
            actual_cols = len(row)
            if actual_cols != expected_cols:
                bad_rows += 1
                bad_col_counts[actual_cols] = bad_col_counts.get(actual_cols, 0) + 1
                if bad_writer is not None and bad_rows_written < max_bad_rows_to_save:
                    bad_writer.writerow(row)
                    bad_rows_written += 1
            else:
                if good_writer is not None:
                    good_writer.writerow(row)

    # Ensure file is closed if we opened one
    # (If bad_writer is None, no extra file was opened.)

    ok = bad_rows == 0
    most_common_bad_cols: int | None = None
    if bad_rows > 0:
        # Pick the column count that appears most often among bad rows
        most_common_bad_cols = max(bad_col_counts.items(), key=lambda kv: kv[1])[0]

    return ok, expected_cols, total_rows, bad_rows, most_common_bad_cols


def check_group_csvs_integrity(
    by_modes_dir: Path,
    summary_csv: Path,
    *,
    save_bad_rows: bool = False,
    save_good_rows: bool = False,
) -> bool:
    """
    Run integrity checks for each group CSV in by_modes_dir.
    Writes a compact CSV summary per group and returns True only if all pass.
    When save_bad_rows is True, also writes per-file bad-row CSVs under
    by_modes_dir / "bad_rows".
    """
    if not by_modes_dir.exists():
        return False

    csv_paths = sorted(by_modes_dir.glob("*.csv"))
    if not csv_paths:
        return False

    # Collect per-file stats first so we can sort by bad_rows desc
    results: list[tuple[str, bool, int | None, int, int, int | None]] = []
    all_ok = True
    bad_rows_root = by_modes_dir / "bad_rows" if save_bad_rows else None
    good_rows_root = by_modes_dir / "good_rows" if save_good_rows else None
    for p in csv_paths:
        bad_rows_path = None
        good_rows_path = None
        if bad_rows_root is not None:
            bad_rows_path = bad_rows_root / p.name
        if good_rows_root is not None:
            good_rows_path = good_rows_root / p.name
        ok, expected_cols, data_rows, bad_rows, most_common_bad_cols = check_csv_integrity(
            p, bad_rows_path, good_rows_path
        )
        all_ok = all_ok and ok
        results.append(
            (p.name, ok, expected_cols, data_rows, bad_rows, most_common_bad_cols)
        )

    # Sort primarily by bad_rows descending, then by filename
    results.sort(key=lambda r: (-r[4], r[0]))

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "ok",
                "expected_cols",
                "data_rows",
                "bad_rows",
                "good_to_bad_ratio",
                "most_common_bad_cols",
            ]
        )
        for file_name, ok, expected_cols, data_rows, bad_rows, most_common_bad_cols in results:
            good_rows = data_rows - bad_rows
            ratio: str | float
            if bad_rows > 0:
                # Simple numeric ratio; leave empty when there are no bad rows
                ratio = round(good_rows / bad_rows, 4)
            else:
                ratio = ""

            writer.writerow(
                [
                    file_name,
                    "OK" if ok else "BAD",
                    expected_cols if expected_cols is not None else "",
                    data_rows,
                    bad_rows,
                    ratio,
                    most_common_bad_cols if most_common_bad_cols is not None else "",
                ]
            )

    return all_ok


def generate_complete_checkpoint(
    project_root: Path,
    *,
    scraped_only: bool = False,
) -> tuple[Path, Path]:
    """
    Generate the combined, deduped checkpoint CSVs in CHECKPOINT_DATE_DST.
    - If scraped_only is False: append new scraped data to the CHECKPOINT_DATE_SRC checkpoint.
    - If scraped_only is True: use only data under scraped_data (no previous checkpoint).
    Returns paths to (out_battle_meta, out_worker_rows).
    """
    new_battle_meta = project_root / SCRAPED_BATTLE_META_PATH
    worker_0 = project_root / WORKER_0_PATH
    worker_1 = project_root / WORKER_1_PATH

    out_battle_meta = (
        project_root
        / CHECKPOINT_DIR
        / CHECKPOINT_DATE_DST
        / ALL_BATTLE_META_FILENAME
    )
    out_worker_rows = (
        project_root
        / CHECKPOINT_DIR
        / CHECKPOINT_DATE_DST
        / ALL_WORKER_ROWS_FILENAME
    )

    if scraped_only:
        append_csv(
            None,
            [new_battle_meta],
            out_battle_meta,
            dedupe_key_column="replayTag",
            dedupe_key_cleaner=_clean_replay_tag,
        )
        append_csv(None, [worker_0, worker_1], out_worker_rows)
    else:
        src_battle_meta = (
            project_root
            / CHECKPOINT_DIR
            / CHECKPOINT_DATE_SRC
            / ALL_BATTLE_META_FILENAME
        )
        src_worker_rows = (
            project_root
            / CHECKPOINT_DIR
            / CHECKPOINT_DATE_SRC
            / ALL_WORKER_ROWS_FILENAME
        )
        append_csv(
            src_battle_meta,
            [new_battle_meta],
            out_battle_meta,
            dedupe_key_column="replayTag",
            dedupe_key_cleaner=_clean_replay_tag,
        )
        append_csv(src_worker_rows, [worker_0, worker_1], out_worker_rows)

    return out_battle_meta, out_worker_rows


def generate_paired_checkpoint(out_battle_meta: Path, out_worker_rows: Path) -> tuple[Path, Path]:
    """
    From the combined checkpoint CSVs, generate the paired+deduped
    meta and worker files and return their paths.
    """
    paired_meta_out = out_battle_meta.parent / "paired_meta_data.csv"
    paired_worker_out = out_worker_rows.parent / "paired_replay_data.csv"
    create_paired_files(out_battle_meta, out_worker_rows, paired_meta_out, paired_worker_out)
    return paired_meta_out, paired_worker_out


def _is_empty_cell(value: str) -> bool:
    """Treat None, empty string, or whitespace-only as empty."""
    if value is None:
        return True
    return str(value).strip() == ""


def filter_proper_meta(
    paired_meta_path: Path,
    headers_csv_path: Path,
    proper_meta_out_path: Path,
) -> Path:
    """
    Keep only rows that have data solely in columns listed in headers_csv.
    Rows may have missing/empty values in those columns, but must not have
    any non-empty value in a column not in headers_csv. Writes result to
    proper_meta_out_path with only the columns from headers_csv (others dropped).
    """
    with headers_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        allowed_cols_ordered = next(reader)  # preserve order from headers.csv
    allowed_cols = set(allowed_cols_ordered)

    proper_meta_out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0

    with paired_meta_path.open("r", newline="", encoding="utf-8") as src, proper_meta_out_path.open(
        "w", newline="", encoding="utf-8"
    ) as out:
        reader = csv.reader(src)
        header = next(reader)
        writer = csv.writer(out)
        # Output header: only allowed columns, in headers.csv order
        writer.writerow(allowed_cols_ordered)

        col_to_idx = {name: i for i, name in enumerate(header)}
        disallowed_indices = [i for i, col in enumerate(header) if col not in allowed_cols]

        for row in reader:
            if len(row) > len(header):
                dropped += 1
                continue
            has_data_outside = False
            for i in disallowed_indices:
                if i < len(row) and not _is_empty_cell(row[i]):
                    has_data_outside = True
                    break
            if has_data_outside:
                dropped += 1
                continue
            # Write only values for allowed columns, in headers.csv order
            out_row = []
            for col in allowed_cols_ordered:
                idx = col_to_idx.get(col)
                if idx is not None and idx < len(row):
                    out_row.append(row[idx])
                else:
                    out_row.append("")
            writer.writerow(out_row)
            kept += 1

    return proper_meta_out_path


def reduce_to_tight_columns(
    proper_meta_path: Path,
    headers_tight_csv_path: Path,
    proper_meta_tight_out_path: Path,
) -> Path:
    """
    Read proper_meta CSV and write a CSV with only the columns in headers_tight_csv.
    All rows are kept; no row filtering. Column order follows headers_tight_csv.
    """
    with headers_tight_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        tight_cols_ordered = next(reader)

    proper_meta_tight_out_path.parent.mkdir(parents=True, exist_ok=True)

    with proper_meta_path.open("r", newline="", encoding="utf-8") as src, proper_meta_tight_out_path.open(
        "w", newline="", encoding="utf-8"
    ) as out:
        reader = csv.reader(src)
        header = next(reader)
        writer = csv.writer(out)
        writer.writerow(tight_cols_ordered)
        col_to_idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            out_row = []
            for col in tight_cols_ordered:
                idx = col_to_idx.get(col)
                if idx is not None and idx < len(row):
                    out_row.append(row[idx])
                else:
                    out_row.append("")
            writer.writerow(out_row)

    return proper_meta_tight_out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build checkpoint from scraped and/or previous checkpoint data.")
    parser.add_argument(
        "--scraped-only",
        action="store_true",
        help="Use only data under data/scraped_data; do not depend on a previous checkpoint.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    # 1) Generate combined, deduped checkpoint data
    out_battle_meta, out_worker_rows = generate_complete_checkpoint(
        project_root, scraped_only=args.scraped_only
    )

    # 2) Generate paired + deduped subsets
    paired_meta_out, paired_worker_out = generate_paired_checkpoint(
        out_battle_meta, out_worker_rows
    )

    # 2b) Keep only meta rows with data solely in columns from headers.csv
    headers_csv_path = project_root / HEADERS_CSV
    if headers_csv_path.exists():
        proper_meta_path = paired_meta_out.parent / PROPER_META_FILENAME
        filter_proper_meta(
            paired_meta_out,
            headers_csv_path,
            proper_meta_path,
        )
        # 2c) From proper_meta, keep only columns in headers_tight.csv (no rows dropped)
        headers_tight_path = project_root / HEADERS_TIGHT_CSV
        if headers_tight_path.exists():
            reduce_to_tight_columns(
                proper_meta_path,
                headers_tight_path,
                paired_meta_out.parent / PROPER_META_TIGHT_FILENAME,
            )

    # 3) Split paired meta by mode into by_modes folder
    by_modes_dir = paired_meta_out.parent / "by_type"
    split_meta_by_mode(paired_meta_out, by_modes_dir)

    # 4) Integrity checks for group CSVs, saved as CSV summary
    integrity_summary = by_modes_dir.parent / "by_modes_integrity.csv"
    check_group_csvs_integrity(
        by_modes_dir,
        integrity_summary,
        save_bad_rows=True,
        save_good_rows=True,
    )


if __name__ == "__main__":
    main()

