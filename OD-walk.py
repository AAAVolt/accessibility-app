import pandas as pd
import numpy as np


def calculate_accessibility_with_times(walking_distance_file, od_data_file, accessibility_file, walking_speed_kmh=4.8):
    """
    Combine walking distances, OD data, and accessibility analysis to calculate
    access and egress times for origin-destination pairs.

    Parameters:
    walking_distance_file: CSV with zone-to-stop walking distances
    od_data_file: CSV with origin-destination routing information
    accessibility_file: CSV with accessibility analysis data
    walking_speed_kmh: Walking speed in km/hour (default 4.8 km/h)

    Walking Time Calculation:
    - Input: distance in meters (from total_cost column)
    - Walking speed: configurable km/h (default 4.8 km/h = 80 m/min)
    - Output: time in minutes

    Travel Time Source:
    - Taken directly from Time_Minutes column in OD data
    - This is the public transport travel time between stops

    Total Journey Time:
    - Public Transport: Access walk + PT travel + Egress walk
    - Walking Only: Direct walking time (from OD data or calculated from distance)
    """

    # Read the walking distance data
    print("Reading walking distance data...")
    walking_df = pd.read_csv(walking_distance_file)
    print(f"Walking data shape: {walking_df.shape}")

    # Read OD routing data
    print("\nReading OD routing data...")
    od_df = pd.read_csv(od_data_file)
    print(f"OD data shape: {od_df.shape}")

    # Read accessibility analysis data
    print("\nReading accessibility analysis data...")
    accessibility_df = pd.read_csv(accessibility_file)
    print(f"Accessibility data shape: {accessibility_df.shape}")

    # Convert walking speed to meters per minute for calculations
    walking_speed_m_per_min = (walking_speed_kmh * 1000) / 60
    print(f"Using walking speed: {walking_speed_kmh} km/h ({walking_speed_m_per_min:.1f} m/min)")

    # Create a lookup dictionary for walking times (zone_id, stop_id) -> total_cost
    print("Creating walking time lookup...")
    walking_lookup = {}
    zone_stop_combinations = set()

    for _, row in walking_df.iterrows():
        zone_id = int(row['origin_id'])
        stop_id = int(row['destination_id'])
        key = (zone_id, stop_id)
        walking_lookup[key] = row['total_cost']
        zone_stop_combinations.add(key)

    print(f"Loaded {len(walking_lookup)} walking distance combinations")

    # Check what zones and stops we have
    available_zones = set(k[0] for k in walking_lookup.keys())
    available_stops = set(k[1] for k in walking_lookup.keys())

    print(
        f"Available zones in walking data: {len(available_zones)} (range: {min(available_zones)}-{max(available_zones)})")
    print(
        f"Available stops in walking data: {len(available_stops)} (range: {min(available_stops)}-{max(available_stops)})")

    # Check what destination values we have in accessibility file
    print(f"Checking Zona_Mapeada values...")
    zona_mapeada_values = accessibility_df['Zona_Mapeada'].dropna()
    unique_destinations = sorted(zona_mapeada_values.unique())
    print(f"Unique Zona_Mapeada values: {unique_destinations[:20]}...")  # Show first 20
    print(f"Total unique destinations in accessibility: {len(unique_destinations)}")

    # Check overlap with OD destinations
    od_destinations = set(od_df['DestZoneNo'].unique())
    acc_destinations = set(int(x) for x in zona_mapeada_values if not pd.isna(x))
    overlap = acc_destinations.intersection(od_destinations)
    print(f"Destinations overlap with OD data: {len(overlap)} out of {len(acc_destinations)}")

    if len(overlap) == 0:
        print("WARNING: No destination overlap found!")
        print(f"Sample OD destinations: {sorted(list(od_destinations))[:20]}")
        print(f"Sample accessibility destinations: {sorted(list(acc_destinations))[:20]}")

    print(f"Starting processing...")

    # Show a few example lookups for debugging
    if len(accessibility_df) > 0:
        sample_row = accessibility_df.iloc[0]
        sample_origin = sample_row['OrigZoneNo']
        sample_dest = sample_row['Zona_Mapeada']
        print(f"\nExample lookup for Origin {sample_origin} -> Destination {sample_dest}:")

        # Find OD match for this example
        sample_od = od_df[(od_df['OrigZoneNo'] == sample_origin) &
                          (od_df['DestZoneNo'] == sample_dest)]
        if not sample_od.empty:
            od_row = sample_od.iloc[0]
            from_stop = int(od_row['FromStopPointNo']) if not pd.isna(od_row['FromStopPointNo']) else None
            to_stop = int(od_row['ToStopPointNo']) if not pd.isna(od_row['ToStopPointNo']) else None

            print(f"  Uses stops: {from_stop} -> {to_stop}")

            if from_stop and to_stop:
                access_key = (int(sample_origin), from_stop)
                egress_key = (int(sample_dest), to_stop)

                print(f"  Access lookup ({sample_origin}, {from_stop}): {walking_lookup.get(access_key, 'NOT FOUND')}")
                print(f"  Egress lookup ({sample_dest}, {to_stop}): {walking_lookup.get(egress_key, 'NOT FOUND')}")
        else:
            print(f"  No OD match found")

    # Process accessibility analysis data - each row represents an origin-destination pair
    print(f"Processing accessibility analysis with {len(accessibility_df)} origin-destination pairs...")

    results = []
    matches_found = 0

    for _, acc_row in accessibility_df.iterrows():
        origin_zone = acc_row['OrigZoneNo']
        destination_zone = acc_row['Zona_Mapeada']

        # Skip if destination is NaN or 0
        if pd.isna(destination_zone) or destination_zone == 0:
            continue

        try:
            destination_zone = int(destination_zone)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert destination {destination_zone} to int")
            continue

        # Find the corresponding OD routing information
        od_match = od_df[(od_df['OrigZoneNo'] == origin_zone) &
                         (od_df['DestZoneNo'] == destination_zone)]

        if not od_match.empty:
            matches_found += 1
            od_row = od_match.iloc[0]  # Take first match if multiple

            # Handle cases where no public transport is needed (NaN stop points)
            from_stop_raw = od_row['FromStopPointNo']
            to_stop_raw = od_row['ToStopPointNo']
            travel_time = od_row['Time_Minutes']

            # Check if this is a direct journey (no public transport needed)
            if pd.isna(from_stop_raw) or pd.isna(to_stop_raw):
                # Direct journey - no public transport, only walking
                from_stop = None
                to_stop = None
                uses_public_transport = False
                print(f"Direct journey found: Origin {origin_zone} -> Destination {destination_zone}")
            else:
                from_stop = int(from_stop_raw)
                to_stop = int(to_stop_raw)
                uses_public_transport = True

            # Calculate access and egress times based on journey type
            if uses_public_transport:
                # Access time: walk from origin zone to departure stop
                # Lookup: (origin_zone, from_stop) - walking FROM zone TO stop
                access_key = (int(origin_zone), from_stop)
                access_distance = walking_lookup.get(access_key, np.nan)

                # Egress time: walk from destination zone to arrival stop
                # Lookup: (destination_zone, to_stop) - walking FROM destination zone TO arrival stop
                # Note: This represents walking distance between the zone and stop (bidirectional)
                egress_key = (destination_zone, to_stop)
                egress_distance = walking_lookup.get(egress_key, np.nan)

                if pd.isna(egress_distance):
                    # Debug: print when egress lookup fails
                    print(f"Egress lookup failed for zone {destination_zone} to stop {to_stop}")
                    # Try to find any walking distances for this destination zone
                    dest_walks = [(k, v) for k, v in walking_lookup.items() if k[0] == destination_zone]
                    if len(dest_walks) > 0:
                        print(f"  Available walks from zone {destination_zone}: {dest_walks[:3]}...")  # Show first 3
                    else:
                        print(f"  No walking distances found for zone {destination_zone}")

                # Convert walking distances to time
                # - distance: meters (from total_cost column in walking data)
                # - speed: meters per minute (converted from km/h input)
                # - result: minutes

                access_time_minutes = access_distance / walking_speed_m_per_min if not pd.isna(
                    access_distance) else np.nan
                egress_time_minutes = egress_distance / walking_speed_m_per_min if not pd.isna(
                    egress_distance) else np.nan

                # Calculate total time
                # Total = Access walking + Public transport (from OD data) + Egress walking
                total_time = np.nan
                if not pd.isna(access_time_minutes) and not pd.isna(egress_time_minutes):
                    total_time = access_time_minutes + travel_time + egress_time_minutes

                journey_type = "Public Transport"

            else:
                # Direct journey - only walking distance between origin and destination
                # Try to find direct walking distance
                direct_key = (int(origin_zone), destination_zone)
                direct_distance = walking_lookup.get(direct_key, np.nan)

                if pd.isna(direct_distance):
                    # If no direct walking distance available, use travel time from OD data
                    # (assuming it's already calculated as walking time in minutes)
                    access_time_minutes = travel_time  # All time is access walking
                    egress_time_minutes = 0
                    total_time = travel_time
                    access_distance = np.nan
                    egress_distance = 0
                else:
                    # Use direct walking distance, convert to time
                    total_time = direct_distance / walking_speed_m_per_min
                    access_time_minutes = total_time  # All time is walking
                    egress_time_minutes = 0
                    access_distance = direct_distance
                    egress_distance = 0

                journey_type = "Walking Only"

            # Add accessibility scores for this origin-destination pair
            accessibility_scores = {}
            for col in accessibility_df.columns:
                if col.startswith('Mapeada_'):
                    accessibility_scores[col] = acc_row[col]

            result_row = {
                'Origin': int(origin_zone),
                'Origin_ZoneName': acc_row.get('ZoneName', ''),
                'Origin_Population': acc_row.get('Population', np.nan),
                'Destination': destination_zone,
                'Uses_Public_Transport': uses_public_transport,
                'FromStopPointNo': from_stop,
                'ToStopPointNo': to_stop,
                'Access_Time_Minutes': access_time_minutes,
                'Egress_Time_Minutes': egress_time_minutes,
                'Access_Distance_Meters': access_distance,
                'Egress_Distance_Meters': egress_distance
            }

            # Add accessibility scores to the result
            result_row.update(accessibility_scores)

            results.append(result_row)

            if len(results) % 100 == 0:  # Progress indicator
                print(f"Processed {len(results)} origin-destination pairs...")

    print(f"Found {matches_found} matching origin-destination pairs in OD routing data")

    if results:
        # Count transport types
        transport_types = {'Public Transport': 0, 'Walking Only': 0}
        for result in results:
            if result['Uses_Public_Transport']:
                transport_types['Public Transport'] += 1
            else:
                transport_types['Walking Only'] += 1

        print(f"Transport types found:")
        for ttype, count in transport_types.items():
            print(f"  {ttype}: {count} journeys")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    print(f"Processed {len(results_df)} origin-destination pairs")

    return results_df


def analyze_accessibility_coverage(results_df, destination_type='all'):
    """
    Analyze accessibility coverage and create summary statistics
    """
    if results_df.empty:
        print("No data to analyze")
        return

    print("\n=== ACCESSIBILITY ANALYSIS SUMMARY ===")

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Total origin zones: {results_df['Origin'].nunique()}")
    print(f"Total destination zones: {results_df['Destination'].nunique()}")
    print(f"Total origin-destination pairs: {len(results_df)}")

    # Time statistics (only for valid journeys)
    valid_journeys = results_df.dropna(subset=['Access_Time_Minutes', 'Egress_Time_Minutes'])
    if not valid_journeys.empty:
        print(f"\nJourney Time Statistics (valid journeys: {len(valid_journeys)}):")
        print(f"Average access time: {valid_journeys['Access_Time_Minutes'].mean():.2f} minutes")
        print(f"Average egress time: {valid_journeys['Egress_Time_Minutes'].mean():.2f} minutes")

    # Missing data analysis
    missing_access = results_df['Access_Time_Minutes'].isna().sum()
    missing_egress = results_df['Egress_Time_Minutes'].isna().sum()

    print(f"\nData Coverage:")
    print(f"Missing access times: {missing_access} ({missing_access / len(results_df) * 100:.1f}%)")
    print(f"Missing egress times: {missing_egress} ({missing_egress / len(results_df) * 100:.1f}%)")

    return valid_journeys


def save_results(results_df, output_file='accessibility_with_times.csv'):
    """Save results to CSV file"""
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    walking_distance_file = r"C:\Users\avoltan\Desktop\OD_walk_data\OD_v4.csv"
    od_data_file = r"C:\Users\avoltan\Desktop\OD_walk_data\min_time_trips_per_od_optimized_v3.csv"
    accessibility_file = r"C:\Users\avoltan\Desktop\OD_walk_data\Hospital_Accessibility.csv"

    # Calculate accessibility with times
    results = calculate_accessibility_with_times(
        walking_distance_file,
        od_data_file,
        accessibility_file
    )

    if not results.empty:
        # Analyze results
        valid_journeys = analyze_accessibility_coverage(results)

        # Show sample of results - MODIFIED to exclude specified columns
        print(f"\nSample results (first 5 rows):")
        sample_cols = ['Origin', 'Destination', 'Access_Time_Minutes', 'Egress_Time_Minutes']
        print(results[sample_cols].head().to_string(index=False))

        # Save results
        save_results(results, 'accessibility_results_with_times.csv')

        # Create summary by origin zones with accessibility scores
        if 'Mapeada_JRT' in results.columns:
            print(f"\nAccessibility correlation analysis:")
            corr_data = results.groupby('Origin').agg({
                'Access_Time_Minutes': 'mean',
                'Mapeada_JRT': 'first',  # Accessibility scores are same for all destinations from same origin
                'Mapeada_NTR': 'first',
                'Origin_Population': 'first'
            }).dropna()

            if len(corr_data) > 1:
                correlation = corr_data['Access_Time_Minutes'].corr(corr_data['Mapeada_JRT'])
                print(f"Correlation between average access time and JRT accessibility: {correlation:.3f}")

        # Summary by destination zones (most/least accessible)
        dest_summary = results.groupby('Destination').agg({
            'Access_Time_Minutes': ['count', 'mean', 'std'],
            'Origin': 'nunique'
        }).round(2)
        dest_summary.columns = ['Journey_Count', 'Avg_Access_Time', 'Std_Access_Time', 'Connected_Origins']

        print(f"\nTop 10 most connected destinations:")
        top_destinations = dest_summary.nlargest(10, 'Connected_Origins')
        print(top_destinations.to_string())

    else:
        print("\nNo results generated. Please check that:")
        print("1. Zone IDs match between accessibility and OD files")
        print("2. Walking distance file covers the stops used in OD routing")
        print("3. File paths are correct")