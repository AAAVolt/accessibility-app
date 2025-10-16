import pandas as pd
import sys


def merge_csv_files(table1_path, table2_path, output_path):
    """
    Merge two CSV files based on zone and stop point matching.

    Args:
        table1_path: Path to CSV with OriginZoneNo, DestZoneNo, FromStopPointNo
        table2_path: Path to CSV with origin_id, destination_id, total_cost
        output_path: Path for the output merged CSV
    """

    try:
        # Read the CSV files
        print("Reading CSV files...")

        # Try to detect the separator for table1
        # First, try with comma separator
        try:
            table1 = pd.read_csv(table1_path)
            # If we get only one column, it might be semicolon-separated
            if len(table1.columns) == 1 and ';' in table1.columns[0]:
                print("Detected semicolon separator in table1, re-reading...")
                table1 = pd.read_csv(table1_path, sep=';')
        except:
            # If comma fails, try semicolon
            print("Trying semicolon separator for table1...")
            table1 = pd.read_csv(table1_path, sep=';')

        table2 = pd.read_csv(table2_path)

        # Display basic info about the datasets
        print(f"Table 1 shape: {table1.shape}")
        print(f"Table 1 columns: {list(table1.columns)}")
        print(f"Table 2 shape: {table2.shape}")
        print(f"Table 2 columns: {list(table2.columns)}")

        # Clean column names (remove any whitespace)
        table1.columns = table1.columns.str.strip()
        table2.columns = table2.columns.str.strip()

        # Handle potential column name variations
        origin_col = None
        dest_col = None
        stop_col = None

        # Look for origin zone column (could be OriginZoneNo or OrigZoneNo)
        for col in table1.columns:
            if 'OriginZone' in col or 'OrigZone' in col:
                origin_col = col
                break

        # Look for destination zone column
        for col in table1.columns:
            if 'DestZone' in col:
                dest_col = col
                break

        # Look for stop point column
        for col in table1.columns:
            if 'StopPoint' in col:
                stop_col = col
                break

        if not all([origin_col, dest_col, stop_col]):
            raise KeyError(f"Could not find required columns. Found: {list(table1.columns)}")

        print(f"Using columns: {origin_col}, {dest_col}, {stop_col}")

        # Perform the merge
        # Left join to keep all records from table1
        print("\nPerforming merge...")
        merged_table = pd.merge(
            table1,
            table2[['origin_id', 'destination_id', 'total_cost']],  # Only select needed columns
            left_on=[origin_col, stop_col],
            right_on=['origin_id', 'destination_id'],
            how='left'
        )

        # Drop the redundant columns from table2
        merged_table = merged_table.drop(['origin_id', 'destination_id'], axis=1)

        # Reorder columns to put total_cost at the end
        cols = [col for col in merged_table.columns if col != 'total_cost'] + ['total_cost']
        merged_table = merged_table[cols]

        # Display merge results
        print(f"Merged table shape: {merged_table.shape}")
        print(f"Records with matching costs: {merged_table['total_cost'].notna().sum()}")
        print(f"Records without matching costs: {merged_table['total_cost'].isna().sum()}")

        # Show sample of merged data
        print("\nSample of merged data:")
        print(merged_table.head())

        # Save the merged table
        merged_table.to_csv(output_path, index=False)
        print(f"\nMerged table saved to: {output_path}")

        # Show statistics about unmatched records if any
        if merged_table['total_cost'].isna().sum() > 0:
            print("\nWarning: Some records couldn't be matched.")
            print("Sample unmatched records:")
            unmatched = merged_table[merged_table['total_cost'].isna()]
            print(unmatched[[origin_col, dest_col, stop_col]].head())

        return merged_table

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None
    except KeyError as e:
        print(f"Error: Missing expected column - {e}")
        print("Please check that your CSV files have the correct column names.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    # File paths - modify these to match your file locations
    table1_path = r"C:\Users\avoltan\Documents\App\OD_HOSP_STOP.csv"  # CSV with OriginZoneNo, DestZoneNo, FromStopPointNo
    table2_path = r"C:\Users\avoltan\Documents\OD_Centroids_Stops.csv"  # CSV with origin_id, destination_id, total_cost
    output_path = r"C:\Users\avoltan\Documents\mergedTable.csv"

    # You can also pass file paths as command line arguments
    if len(sys.argv) == 4:
        table1_path = sys.argv[1]
        table2_path = sys.argv[2]
        output_path = sys.argv[3]
    elif len(sys.argv) > 1:
        print("Usage: python script.py [table1.csv] [table2.csv] [output.csv]")
        return

    print("CSV Merger Script")
    print("=================")
    print(f"Table 1 (zones): {table1_path}")
    print(f"Table 2 (costs): {table2_path}")
    print(f"Output file: {output_path}")
    print()

    # Perform the merge
    result = merge_csv_files(table1_path, table2_path, output_path)

    if result is not None:
        print("Merge completed successfully!")
    else:
        print("Merge failed. Please check the error messages above.")


if __name__ == "__main__":
    main()