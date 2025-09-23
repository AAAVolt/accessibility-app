import pandas as pd
import numpy as np


def calculate_accessibility_with_times(walking_distance_file, od_data_file, accessibility_file):
    """
    Combine walking distances, OD data, and accessibility analysis to calculate
    access and egress times for origin-destination pairs.

    Parameters:
    walking_distance_file: CSV with zone-to-stop walking distances
    od_data_file: CSV with origin-destination routing information
    accessibility_file: CSV with accessibility analysis data
    """

    # Read the walking distance data
    print("Reading walking distance data...")
    walking_df = pd.read_csv(walking_distance_file)

    # Read OD routing data
    print("Reading OD routing data...")
    od_df = pd.read_csv(od_data_file)

    # Read accessibility analysis data
    print("Reading accessibility analysis data...")
    accessibility_df = pd.read_csv(accessibility_file)

    # Create a lookup dictionary for walking times (zone_id, stop_id) -> total_cost
    print("Creating walking time lookup...")
    walking_lookup = {}
    for _, row in walking_df.iterrows():
        key = (int(row['zone_id']), int(row['stop_id']))
        walking_lookup[key] = row['total_cost']

    # Process accessibility data to create origin-destination pairs
    print("Processing accessibility analysis...")
    results = []

    # Get all destination columns (those starting with 'Mapeada_')
    destination_cols = [col for col in accessibility_df.columns if col.startswith('Mapeada_')]

    for _, acc_row in accessibility_df.iterrows():
        origin_zone = acc_row['OrigZoneNo']

        for dest_col in destination_cols:
            destination_zone = acc_row[dest_col]

            # Skip if destination is NaN or 0
            if pd.isna(destination_zone) or destination_zone == 0:
                continue

            destination_zone = int(destination_zone)

            # Find the corresponding OD routing information
            od_match = od_df[(od_df['OrigZoneNo'] == origin_zone) &
                             (od_df['DestZoneNo'] == destination_zone)]

            if not od_match.empty:
                od_row = od_match.iloc[0]  # Take first match if multiple

                from_stop = int(od_row['FromStopPointNo'])
                to_stop = int(od_row['ToStopPointNo'])
                travel_time = od_row['Time_Minutes']

                # Get access time (origin zone to from_stop)
                access_key = (int(origin_zone), from_stop)
                access_time = walking_lookup.get(access_key, np.nan)

                # Get egress time (to_stop to destination zone)
                egress_key = (destination_zone, to_stop)
                egress_time = walking_lookup.get(egress_key, np.nan)

                # Convert walking distances to time (assuming walking speed)
                # You may need to adjust this conversion factor based on your units
                # If total_cost is in meters and you want minutes, use walking speed
                walking_speed_m_per_min = 80  # approximately 4.8 km/h

                access_time_minutes = access_time / walking_speed_m_per_min if not pd.isna(access_time) else np.nan
                egress_time_minutes = egress_time / walking_speed_m_per_min if not pd.isna(egress_time) else np.nan

                # Calculate total time
                total_time = np.nan
                if not pd.isna(access_time_minutes) and not pd.isna(egress_time_minutes):
                    total_time = access_time_minutes + travel_time + egress_time_minutes

                results.append({
                    'Origin': int(origin_zone),
                    'Destination': destination_zone,
                    'Destination_Type': dest_col.replace('Mapeada_', ''),
                    'FromStopPointNo': from_stop,
                    'ToStopPointNo': to_stop,
                    'Access_Time_Minutes': access_time_minutes,
                    'Travel_Time_Minutes': travel_time,
                    'Egress_Time_Minutes': egress_time_minutes,
                    'Total_Time_Minutes': total_time,
                    'Access_Distance_Meters': access_time,
                    'Egress_Distance_Meters': egress_time
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    print(f"Processed {len(results_df)} origin-destination pairs")

    return results_df


def save_results(results_df, output_file='accessibility_with_times.csv'):
    """Save results to CSV file"""
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    walking_distance_file = r"C:\Users\avoltan\Desktop\OD_walk_data\OD_v3.csv"
    od_data_file = r"C:\Users\avoltan\Desktop\OD_walk_data\min_time_trips_per_od_optimized_v3.csv"
    accessibility_file = r"C:\Users\avoltan\Desktop\OD_walk_data\Hospital_Accessibility.csv"

    # Calculate accessibility with times
    results = calculate_accessibility_with_times(
        walking_distance_file,
        od_data_file,
        accessibility_file
    )

    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Total origin-destination pairs: {len(results)}")
    print(f"Average access time: {results['Access_Time_Minutes'].mean():.2f} minutes")
    print(f"Average egress time: {results['Egress_Time_Minutes'].mean():.2f} minutes")
    print(f"Average total time: {results['Total_Time_Minutes'].mean():.2f} minutes")

    # Show sample of results
    print("\nSample results:")
    print(results.head(10).to_string())

    # Save results
    save_results(results, 'accessibility_results_with_times.csv')

    # Optional: Create summary by destination type
    summary_by_dest = results.groupby('Destination_Type').agg({
        'Access_Time_Minutes': ['mean', 'std'],
        'Travel_Time_Minutes': ['mean', 'std'],
        'Egress_Time_Minutes': ['mean', 'std'],
        'Total_Time_Minutes': ['mean', 'std']
    }).round(2)

    print("\nSummary by Destination Type:")
    print(summary_by_dest.to_string())