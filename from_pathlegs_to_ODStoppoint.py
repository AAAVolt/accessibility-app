import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import gc
import os

# ========== CONFIG: exact, uppercase column names from your file ==========
REQ_COLS = [
    "ORIGZONENO",
    "DESTZONENO",
    "PATHINDEX",
    "PATHLEGINDEX",  # Must include to identify aggregate rows (NaN)
    "TIME",  # On aggregate row: total path time; on leg rows: ignored
    "FROMSTOPPOINTNO",  # origin stop point
    "TOSTOPPOINTNO"  # destination stop point
]


def detect_delimiter(csv_file_path: str) -> str:
    """
    Detect the delimiter used in the CSV file by reading the first line.
    """
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    if ';' in first_line and first_line.count(';') > first_line.count(','):
        return ';'
    elif ',' in first_line:
        return ','
    elif '\t' in first_line:
        return '\t'
    else:
        return ','


def find_min_time_trips_fast(csv_file_path: str, chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Find the minimum-time path per (ORIGZONENO, DESTZONENO) on very large CSVs.

    Key insight:
      - Aggregate rows (PATHLEGINDEX = NaN) contain the TOTAL TIME for the whole path
      - We ONLY read aggregate rows and extract their TIME directly (no summing needed)
      - Leg rows are ignored - they're just detailed breakdowns
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV not found: {csv_file_path}")

    t0 = time.time()
    print("🚀 Starting fast min-time extraction…")

    delimiter = detect_delimiter(csv_file_path)
    print(f"📋 Detected delimiter: '{delimiter}'")

    # Validate required columns
    header_df = pd.read_csv(csv_file_path, nrows=0, delimiter=delimiter)
    print(f"📁 Raw columns found: {list(header_df.columns)}")

    col_map = {col.upper(): col for col in header_df.columns}
    missing = [c for c in REQ_COLS if c not in col_map]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {sorted(col_map.keys())}"
        )

    actual_cols = [col_map[c] for c in REQ_COLS]
    print(f"📁 Will read columns: {actual_cols}")

    read_kwargs = dict(
        usecols=actual_cols,
        chunksize=chunk_size,
        delimiter=delimiter,
        na_values=["", " "],
        keep_default_na=True,
        low_memory=False
    )
    try:
        read_kwargs["dtype_backend"] = "pyarrow"
    except Exception:
        pass

    min_times = {}  # (orig, dest) -> dict with fields

    with pd.read_csv(csv_file_path, **read_kwargs) as it:
        pbar = tqdm(desc="Processing", unit="rows")
        for i, chunk in enumerate(it):
            # Standardize column names to uppercase
            chunk.columns = chunk.columns.str.upper()

            # Ensure TIME is numeric
            chunk["TIME"] = pd.to_numeric(chunk["TIME"], errors="coerce")

            # ONLY keep aggregate rows (PATHLEGINDEX is NA)
            # These rows have the TOTAL path time already calculated
            agg = chunk[chunk["PATHLEGINDEX"].isna()].copy()

            if not agg.empty:
                agg = agg.dropna(subset=["ORIGZONENO", "DESTZONENO", "PATHINDEX", "TIME"])

                for idx, row in agg.iterrows():
                    try:
                        orig = int(float(row["ORIGZONENO"]))
                        dest = int(float(row["DESTZONENO"]))
                        path_idx = int(float(row["PATHINDEX"]))
                        time_days = float(row["TIME"])  # TIME is in fractional days
                    except Exception:
                        continue

                    if time_days <= 0:  # Skip invalid times
                        continue

                    key = (orig, dest)
                    time_sec = time_days * 86400  # Convert days to seconds
                    time_min = time_sec / 60.0

                    # Extract stop points
                    try:
                        from_stop = int(float(row["FROMSTOPPOINTNO"])) if pd.notna(row["FROMSTOPPOINTNO"]) else None
                    except Exception:
                        from_stop = None

                    try:
                        to_stop = int(float(row["TOSTOPPOINTNO"])) if pd.notna(row["TOSTOPPOINTNO"]) else None
                    except Exception:
                        to_stop = None

                    # Update if this is a new OD pair or faster than previous best
                    prev = min_times.get(key)
                    if prev is None or time_min < prev["Time_Minutes"]:
                        min_times[key] = {
                            "OrigZoneNo": orig,
                            "DestZoneNo": dest,
                            "Time": time_sec,
                            "Time_Minutes": time_min,
                            "FromStopPointNo": from_stop,
                            "ToStopPointNo": to_stop,
                            "PathIndex": path_idx
                        }

            pbar.update(len(chunk))

            if i % 8 == 0:
                gc.collect()

        pbar.close()

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(min_times, orient="index").reset_index(drop=True)
    if not df.empty:
        df = df.sort_values(["OrigZoneNo", "DestZoneNo"]).reset_index(drop=True)

    dt = time.time() - t0
    print(f"✅ Done in {dt:.2f}s – OD pairs: {len(df):,}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 56)
    print("📋 MINIMUM TIME TRIPS PER (ORIGZONENO, DESTZONENO) – RESULTS")
    print("=" * 56)
    print(f"🎯 Total unique OD pairs: {len(df):,}")

    if len(df) == 0:
        return

    print(f"⚡ Average minimum time: {df['Time_Minutes'].mean():.2f} min")
    print(f"🏃 Fastest trip: {df['Time_Minutes'].min():.2f} min")
    print(f"🢢 Slowest trip: {df['Time_Minutes'].max():.2f} min")

    sample_cols = ["OrigZoneNo", "DestZoneNo", "Time_Minutes", "Time", "FromStopPointNo", "ToStopPointNo", "PathIndex"]
    print("\n📊 Sample (first 10):")
    print(df.head(10)[sample_cols].to_string(index=False))


def save_results(df: pd.DataFrame, output_file_path: str) -> None:
    print(f"💾 Saving results to {output_file_path}…")
    df.to_csv(output_file_path, index=False)
    print("✅ Results saved.")


def auto_chunk_size(fallback: int = 200_000) -> int:
    """
    Pick a chunk size based on available RAM, if psutil is available.
    """
    try:
        import psutil
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
    csv_file_path = r"C:\Users\avoltan\Documents\bilbao_PUTHPATHLEGS.csv"
    output_file = "results/min_time_trips_per_od_optimized_bilbao.csv"

    print("🔧 Diagnosing CSV structure first...")
    try:
        diagnose_csv_structure(csv_file_path)
    except Exception as e:
        print(f"⚠️ Diagnosis failed: {e}")

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
        print(f"❌ Schema error: {e}")
    except Exception as e:
        print(f"❌ Error processing file: {e}")