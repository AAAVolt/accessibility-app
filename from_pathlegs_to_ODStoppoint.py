import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import gc
import os

# ========== CONFIG: exact, uppercase column names from your file ==========
# Keep this list tight to reduce memory. Add more cols only if you truly need them.
REQ_COLS = [
    "ORIGZONENO",
    "DESTZONENO",
    "PATHINDEX",
    "PATHLEGINDEX",
    "TIME",  # total time (seconds) on the aggregate path row
    "FROMSTOPPOINTNO",  # origin stop point
    "TOSTOPPOINTNO"     # destination stop point
]


def detect_delimiter(csv_file_path: str) -> str:
    """
    Detect the delimiter used in the CSV file by reading the first line.
    """
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    # Check for common delimiters
    if ';' in first_line and first_line.count(';') > first_line.count(','):
        return ';'
    elif ',' in first_line:
        return ','
    elif '\t' in first_line:
        return '\t'
    else:
        # Default to comma
        return ','


def find_min_time_trips_fast(csv_file_path: str, chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Find the minimum-time path per (ORIGZONENO, DESTZONENO) on very large CSVs (200M+ rows).

    Assumptions / strategy:
      - The CSV has both aggregate "path" rows and per-leg rows.
      - Aggregate rows have PATHLEGINDEX missing/blank (NaN after parsing),
        while leg rows have PATHLEGINDEX = 1,2,3,...
      - TIME on aggregate rows is the total travel time for the whole path (seconds).
      - We only need the minimum TIME per (ORIGZONENO, DESTZONENO) across paths.
      - We DO NOT force integer numpy dtypes on columns with missing values to avoid
        "Integer column has NA values in column 0" errors.

    Memory / speed tricks:
      - usecols limits columns read
      - na_values treats "" and " " as NA
      - avoid dtype=int32 for columns that have blanks; let pandas infer or use nullable ints later
      - process in chunks; only aggregate rows are kept
      - update a small in-memory dict of minima
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV not found: {csv_file_path}")

    t0 = time.time()
    print("🚀 Starting fast min-time extraction (with delimiter detection)…")

    # Detect delimiter
    delimiter = detect_delimiter(csv_file_path)
    print(f"📋 Detected delimiter: '{delimiter}'")

    # Validate required columns exist (reads header only)
    header_df = pd.read_csv(csv_file_path, nrows=0, delimiter=delimiter)
    header = set(map(str.upper, header_df.columns))
    print(f"📁 Found columns: {sorted(header)}")

    missing = [c for c in REQ_COLS if c not in header]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {sorted(header)}"
        )

    # Prepare read_csv kwargs
    read_kwargs = dict(
        usecols=REQ_COLS,
        chunksize=chunk_size,
        delimiter=delimiter,  # Add delimiter parameter
        na_values=["", " "],
        keep_default_na=True,
        low_memory=False
    )
    # If pandas >= 2.0 and pyarrow is installed, this reduces memory further
    try:
        read_kwargs["dtype_backend"] = "pyarrow"  # optional, safe to skip if unsupported
    except Exception:
        pass

    min_times = {}  # (orig, dest) -> dict with fields

    # We don't pre-count rows (expensive on 200M+). tqdm without total still shows speed & ETA.
    with pd.read_csv(csv_file_path, **read_kwargs) as it:
        pbar = tqdm(desc="Processing", unit="rows")
        for i, chunk in enumerate(it):
            # Keep only aggregate path rows (PATHLEGINDEX is NA)
            agg = chunk[chunk["PATHLEGINDEX"].isna()]

            # Drop rows without OD or TIME
            agg = agg.dropna(subset=["ORIGZONENO", "DESTZONENO", "TIME"])
            if not agg.empty:
                # TIME might be string; coerce safely (seconds)
                time_sec = pd.to_numeric(agg["TIME"], errors="coerce")
                agg = agg.assign(Time_Minutes=time_sec / 60.0)
                agg = agg.dropna(subset=["Time_Minutes"])

                # Group by OD and update global minima
                for (orig, dest), grp in agg.groupby(["ORIGZONENO", "DESTZONENO"], sort=False):
                    j = grp["Time_Minutes"].idxmin()
                    row = agg.loc[j]

                    # orig/dest arrive as numbers or strings; ensure ints if possible
                    try:
                        key = (int(float(orig)), int(float(dest)))
                    except Exception:
                        # Fallback: keep as-is (rare)
                        key = (orig, dest)

                    new_min = float(row["Time_Minutes"])
                    prev = min_times.get(key)
                    if prev is None or new_min < prev["Time_Minutes"]:
                        # Process origin stop point
                        from_stop = row.get("FROMSTOPPOINTNO")
                        try:
                            from_stop_val = int(float(from_stop)) if pd.notna(from_stop) else None
                        except Exception:
                            from_stop_val = None

                        # Process destination stop point
                        to_stop = row.get("TOSTOPPOINTNO")
                        try:
                            to_stop_val = int(float(to_stop)) if pd.notna(to_stop) else None
                        except Exception:
                            to_stop_val = None

                        # Process path index
                        path_index = row.get("PATHINDEX")
                        try:
                            path_index_val = int(float(path_index)) if pd.notna(path_index) else None
                        except Exception:
                            path_index_val = None

                        min_times[key] = {
                            "OrigZoneNo": key[0],
                            "DestZoneNo": key[1],
                            "Time": float(row["TIME"]),  # seconds
                            "Time_Minutes": new_min,  # minutes
                            "FromStopPointNo": from_stop_val,
                            "ToStopPointNo": to_stop_val,  # Added destination stop point
                            "PathIndex": path_index_val
                        }

            pbar.update(len(chunk))

            # Periodic GC for very large runs
            if i % 8 == 0:
                gc.collect()

        pbar.close()

    # Dict -> DataFrame
    df = pd.DataFrame.from_dict(min_times, orient="index").reset_index(drop=True)
    if not df.empty:
        df = df.sort_values(["OrigZoneNo", "DestZoneNo"]).reset_index(drop=True)

    dt = time.time() - t0
    print(f"✅ Done in {dt:.2f}s — OD pairs: {len(df):,}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 56)
    print("📋 MINIMUM TIME TRIPS PER (ORIGZONENO, DESTZONENO) — RESULTS")
    print("=" * 56)
    print(f"🎯 Total unique OD pairs: {len(df):,}")

    if len(df) == 0:
        return

    print(f"⚡ Average minimum time: {df['Time_Minutes'].mean():.2f} min")
    print(f"🏃 Fastest trip: {df['Time_Minutes'].min():.2f} min")
    print(f"🐢 Slowest trip: {df['Time_Minutes'].max():.2f} min")

    sample_cols = ["OrigZoneNo", "DestZoneNo", "Time_Minutes", "Time", "FromStopPointNo", "ToStopPointNo", "PathIndex"]
    print("\n📊 Sample (first 10):")
    print(df.head(10)[sample_cols].to_string(index=False))


def save_results(df: pd.DataFrame, output_file_path: str) -> None:
    print(f"💾 Saving results to {output_file_path}…")
    # For huge outputs, disable index and use default CSV settings
    df.to_csv(output_file_path, index=False)
    print("✅ Results saved.")


def auto_chunk_size(fallback: int = 200_000) -> int:
    """
    Pick a chunk size based on available RAM, if psutil is available.
    Heuristic: ~50k rows per GB, capped at 300k.
    """
    try:
        import psutil  # type: ignore
        gb = psutil.virtual_memory().available / (1024 ** 3)
        return int(min(300_000, max(100_000, gb * 50_000)))
    except Exception:
        return fallback


def diagnose_csv_structure(csv_file_path: str, num_rows: int = 5) -> None:
    """
    Helper function to diagnose CSV structure and show first few rows.
    """
    print("🔍 CSV DIAGNOSIS")
    print("=" * 40)

    # Try different delimiters
    delimiters = [',', ';', '\t', '|']

    for delim in delimiters:
        try:
            print(f"\nTrying delimiter: '{delim}'")
            df_sample = pd.read_csv(csv_file_path, nrows=num_rows, delimiter=delim)
            print(f"Columns ({len(df_sample.columns)}): {list(df_sample.columns)}")
            print("Sample data:")
            print(df_sample.head(2))
            print("-" * 40)
        except Exception as e:
            print(f"Failed with '{delim}': {e}")

    print("=" * 40)


if __name__ == "__main__":
    # ==== CHANGE THIS PATH ====
    csv_file_path = r"C:\Users\avoltan\Documents\put_path_legs_v2.csv"
    output_file = "min_time_trips_per_od_optimized_v3.csv"

    # First, let's diagnose the CSV structure
    print("🔧 Diagnosing CSV structure first...")
    try:
        diagnose_csv_structure(csv_file_path)
    except Exception as e:
        print(f"⚠️ Diagnosis failed: {e}")

    # Then try to process
    try:
        cs = auto_chunk_size()
        print(f"💻 Using chunk_size={cs:,}")
        result = find_min_time_trips_fast(csv_file_path, chunk_size=cs)
        print_summary(result)
        save_results(result, output_file)
        print(f"\n🎉 All done! Check '{output_file}' for complete results.")
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except ValueError as e:
        # Most likely missing required columns; show details
        print(f"❌ Schema error: {e}")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        print("Tip: Ensure columns are exactly the uppercase names shown in REQ_COLS, "
              "and do not force integer dtypes on columns that contain blanks.")