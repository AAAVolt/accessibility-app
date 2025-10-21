import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import io


def convert_excel_time_to_time(excel_time):
    """Convert Excel time format to Python time object"""
    if pd.isna(excel_time):
        return None

    try:
        # Handle pandas Timedelta objects (like "0 days 06:23:06")
        if isinstance(excel_time, pd.Timedelta):
            total_seconds = int(excel_time.total_seconds())
            hours = (total_seconds // 3600) % 24
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return time(hours, minutes, seconds)

        # If it's already a pandas datetime, extract time
        if hasattr(excel_time, 'time'):
            return excel_time.time()

        # If it's a string, try to parse it
        if isinstance(excel_time, str):
            dt = pd.to_datetime(excel_time)
            return dt.time()

        # Excel stores dates as numbers (days since 1900-01-01)
        if isinstance(excel_time, (int, float)):
            # Convert Excel date number to datetime
            dt = pd.to_datetime('1899-12-30') + pd.Timedelta(days=excel_time)
            return dt.time()

        # Try direct conversion to datetime
        dt = pd.to_datetime(excel_time)
        return dt.time()

    except Exception as e:
        return None


def load_and_process_data(uploaded_file):
    """Load Excel file and process the data"""
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, sheet_name=0)

        # Clean up the dataframe
        df = df.dropna(how='all').reset_index(drop=True)

        # Handle the case where first column might be unnamed index
        if df.columns[0] in [None, 'Unnamed: 0'] or str(df.columns[0]).startswith('Unnamed'):
            df = df.drop(columns=[df.columns[0]])

        # Clean column names
        df.columns = [str(col).strip() if col is not None else f'Col_{i}' for i, col in enumerate(df.columns)]

        # Check if we have the required columns
        required_cols = ['OrigZoneNo', 'DestZoneNo', 'DepTime', 'ArrTime']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return None, f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}"

        # Convert time columns
        df['DepTime_processed'] = df['DepTime'].apply(convert_excel_time_to_time)
        df['ArrTime_processed'] = df['ArrTime'].apply(convert_excel_time_to_time)

        # Filter out rows where times are invalid or ArrTime <= DepTime
        valid_times_mask = (df['DepTime_processed'].notna() & df['ArrTime_processed'].notna())
        df = df[valid_times_mask].reset_index(drop=True)

        if len(df) == 0:
            return None, "No rows with valid departure and arrival times found"

        # Filter out night trips (where ArrTime <= DepTime)
        valid_order_mask = (df['ArrTime_processed'] > df['DepTime_processed'])
        df = df[valid_order_mask].reset_index(drop=True)

        if len(df) == 0:
            return None, "No valid journeys found (all arrival times were before or equal to departure times)"

        return df, None

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error loading file: {str(e)}\n\nDetails:\n{error_details}"


def apply_zone_filters(df, orig_filter_type, orig_values, dest_filter_type, dest_values):
    """Apply zone filters based on filter type and values"""
    filtered_df = df.copy()

    # Apply origin zone filter
    if orig_filter_type == "Specific Zones" and orig_values:
        filtered_df = filtered_df[filtered_df['OrigZoneNo'].isin(orig_values)]
    elif orig_filter_type == "Greater Than" and orig_values:
        filtered_df = filtered_df[filtered_df['OrigZoneNo'] > orig_values[0]]
    elif orig_filter_type == "Less Than" and orig_values:
        filtered_df = filtered_df[filtered_df['OrigZoneNo'] < orig_values[0]]
    elif orig_filter_type == "Range" and len(orig_values) == 2:
        filtered_df = filtered_df[
            (filtered_df['OrigZoneNo'] >= orig_values[0]) &
            (filtered_df['OrigZoneNo'] <= orig_values[1])
            ]

    # Apply destination zone filter
    if dest_filter_type == "Specific Zones" and dest_values:
        filtered_df = filtered_df[filtered_df['DestZoneNo'].isin(dest_values)]
    elif dest_filter_type == "Greater Than" and dest_values:
        filtered_df = filtered_df[filtered_df['DestZoneNo'] > dest_values[0]]
    elif dest_filter_type == "Less Than" and dest_values:
        filtered_df = filtered_df[filtered_df['DestZoneNo'] < dest_values[0]]
    elif dest_filter_type == "Range" and len(dest_values) == 2:
        filtered_df = filtered_df[
            (filtered_df['DestZoneNo'] >= dest_values[0]) &
            (filtered_df['DestZoneNo'] <= dest_values[1])
            ]

    return filtered_df


def apply_time_filters(df, time_type, time_filter_type, time_values):
    """Apply time filters"""
    filtered_df = df.copy()

    # ALWAYS filter for departure times after 04:00:00
    min_dep_time = time(4, 0, 0)  # 04:00:00
    filtered_df = filtered_df[filtered_df['DepTime_processed'] > min_dep_time]

    # Apply additional user-selected time filters
    if time_values and time_filter_type != "All Times":
        time_col = f'{time_type}_processed'

        if time_filter_type == "After Time":
            filtered_df = filtered_df[filtered_df[time_col] >= time_values[0]]
        elif time_filter_type == "Before Time":
            filtered_df = filtered_df[filtered_df[time_col] <= time_values[0]]
        elif time_filter_type == "Time Range" and len(time_values) == 2:
            filtered_df = filtered_df[
                (filtered_df[time_col] >= time_values[0]) &
                (filtered_df[time_col] <= time_values[1])
                ]

    return filtered_df


def find_earliest_or_latest_per_zone_pair(df, time_type='DepTime', find_type='Earliest'):
    """Find the earliest or latest time for each origin-destination zone pair"""
    if df.empty:
        return pd.DataFrame()

    time_col = f'{time_type}_processed'

    # Group by origin and destination zones and find the earliest/latest time for each pair
    if find_type == 'Earliest':
        result_per_pair = df.loc[df.groupby(['OrigZoneNo', 'DestZoneNo'])[time_col].idxmin()]
        # Sort by time for better presentation
        result_per_pair = result_per_pair.sort_values(time_col).reset_index(drop=True)
    else:  # Latest
        result_per_pair = df.loc[df.groupby(['OrigZoneNo', 'DestZoneNo'])[time_col].idxmax()]
        # Sort by time for better presentation (latest first)
        result_per_pair = result_per_pair.sort_values(time_col, ascending=False).reset_index(drop=True)

    return result_per_pair


# Streamlit App
def main():
    st.set_page_config(
        page_title="Advanced Journey Data Analyzer",
        page_icon="🚌",
        layout="wide"
    )

    st.title("🚌 Advanced Journey Data Analyzer")
    st.markdown("Upload your Excel file and analyze journey paths with advanced filtering capabilities")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with journey data including OrigZoneNo, DestZoneNo, DepTime, and ArrTime columns"
    )

    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            df, error = load_and_process_data(uploaded_file)

        if error:
            st.error(f"Error loading file: {error}")
            return

        if df is None or df.empty:
            st.error("No valid data found in the file")
            return

        # Display basic info
        st.success(f"✅ Data loaded successfully! Found {len(df)} valid rows")

        # Show data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Origin Zones", df['OrigZoneNo'].nunique())
        with col3:
            st.metric("Destination Zones", df['DestZoneNo'].nunique())
        with col4:
            st.metric("Unique Pairs", len(df.groupby(['OrigZoneNo', 'DestZoneNo'])))

        st.divider()

        # Advanced Filters
        st.subheader("🔍 Advanced Filters")

        # Create two columns for zone filters
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Origin Zone Filters**")
            orig_filter_type = st.selectbox(
                "Origin Filter Type",
                ["All Zones", "Specific Zones", "Greater Than", "Less Than", "Range"],
                key="orig_filter_type"
            )

            orig_values = None
            if orig_filter_type == "Specific Zones":
                orig_values = st.multiselect(
                    "Select Origin Zones",
                    options=sorted(df['OrigZoneNo'].unique()),
                    key="orig_specific"
                )
            elif orig_filter_type in ["Greater Than", "Less Than"]:
                orig_values = [st.number_input(
                    f"Origin Zone {orig_filter_type.lower()}:",
                    min_value=int(df['OrigZoneNo'].min()),
                    max_value=int(df['OrigZoneNo'].max()),
                    value=int(df['OrigZoneNo'].min()),
                    key="orig_single"
                )]
            elif orig_filter_type == "Range":
                col_a, col_b = st.columns(2)
                with col_a:
                    min_val = st.number_input(
                        "Min Origin Zone:",
                        min_value=int(df['OrigZoneNo'].min()),
                        max_value=int(df['OrigZoneNo'].max()),
                        value=int(df['OrigZoneNo'].min()),
                        key="orig_min"
                    )
                with col_b:
                    max_val = st.number_input(
                        "Max Origin Zone:",
                        min_value=int(df['OrigZoneNo'].min()),
                        max_value=int(df['OrigZoneNo'].max()),
                        value=int(df['OrigZoneNo'].max()),
                        key="orig_max"
                    )
                orig_values = [min_val, max_val]

        with col2:
            st.markdown("**Destination Zone Filters**")
            dest_filter_type = st.selectbox(
                "Destination Filter Type",
                ["All Zones", "Specific Zones", "Greater Than", "Less Than", "Range"],
                key="dest_filter_type"
            )

            dest_values = None
            if dest_filter_type == "Specific Zones":
                dest_values = st.multiselect(
                    "Select Destination Zones",
                    options=sorted(df['DestZoneNo'].unique()),
                    key="dest_specific"
                )
            elif dest_filter_type in ["Greater Than", "Less Than"]:
                dest_values = [st.number_input(
                    f"Destination Zone {dest_filter_type.lower()}:",
                    min_value=int(df['DestZoneNo'].min()),
                    max_value=int(df['DestZoneNo'].max()),
                    value=int(df['DestZoneNo'].min()),
                    key="dest_single"
                )]
            elif dest_filter_type == "Range":
                col_a, col_b = st.columns(2)
                with col_a:
                    min_val = st.number_input(
                        "Min Dest Zone:",
                        min_value=int(df['DestZoneNo'].min()),
                        max_value=int(df['DestZoneNo'].max()),
                        value=int(df['DestZoneNo'].min()),
                        key="dest_min"
                    )
                with col_b:
                    max_val = st.number_input(
                        "Max Dest Zone:",
                        min_value=int(df['DestZoneNo'].min()),
                        max_value=int(df['DestZoneNo'].max()),
                        value=int(df['DestZoneNo'].max()),
                        key="dest_max"
                    )
                dest_values = [min_val, max_val]

        # Time analysis and filtering
        st.markdown("**Time Analysis & Filters**")
        st.info("ℹ️ Note: All results automatically filtered to show only departure times after 04:00:00")

        col1, col2, col3 = st.columns(3)

        with col1:
            time_type = st.selectbox(
                "Analyze by Time Type",
                options=['DepTime', 'ArrTime'],
                help="Choose whether to analyze departure times or arrival times"
            )

        with col2:
            find_type = st.selectbox(
                "Find Time Type",
                options=['Earliest', 'Latest'],
                help="Choose whether to find earliest or latest times for each zone pair"
            )

        with col3:
            time_filter_type = st.selectbox(
                "Time Filter",
                ["All Times", "After Time", "Before Time", "Time Range"]
            )

        # Time filter inputs
        time_values = None
        if time_filter_type in ["After Time", "Before Time"]:
            time_values = [st.time_input(
                f"Select {time_filter_type.lower()}:",
                value=time(6, 0),  # Default 6:00 AM
                key="time_single"
            )]
        elif time_filter_type == "Time Range":
            col_a, col_b = st.columns(2)
            with col_a:
                start_time = st.time_input(
                    "Start Time:",
                    value=time(6, 0),
                    key="time_start"
                )
            with col_b:
                end_time = st.time_input(
                    "End Time:",
                    value=time(22, 0),
                    key="time_end"
                )
            time_values = [start_time, end_time]

        st.divider()

        # Apply all filters
        filtered_df = apply_zone_filters(df, orig_filter_type, orig_values, dest_filter_type, dest_values)
        filtered_df = apply_time_filters(filtered_df, time_type, time_filter_type, time_values)

        # Results
        st.subheader("📊 Results")

        if filtered_df.empty:
            st.warning("No data matches the selected filters")
        else:
            # Show filtered data count
            unique_pairs = len(filtered_df.groupby(['OrigZoneNo', 'DestZoneNo']))
            st.info(f"Found {len(filtered_df)} journeys across {unique_pairs} unique origin-destination pairs")

            # Find earliest or latest time for each zone pair
            result_per_pair = find_earliest_or_latest_per_zone_pair(filtered_df, time_type, find_type)

            if not result_per_pair.empty:
                st.subheader(f"🕐 {find_type} {time_type} for Each Zone Pair")

                # Prepare display dataframe
                display_df = result_per_pair.copy()

                # Format time columns for display
                for time_col in ['DepTime', 'ArrTime']:
                    if f'{time_col}_processed' in display_df.columns:
                        display_df[f'{time_col}_Display'] = display_df[f'{time_col}_processed'].apply(
                            lambda x: x.strftime("%H:%M:%S") if x is not None else "N/A"
                        )

                # Format journey time (convert from days to hours:minutes)
                if 'JourneyTime' in display_df.columns:
                    display_df['JourneyTime_Display'] = display_df['JourneyTime'].apply(
                        lambda x: f"{int(x * 24)}h {int((x * 24 * 60) % 60)}m" if pd.notna(x) else "N/A"
                    )

                # Select columns for display
                display_columns = ['OrigZoneNo', 'DestZoneNo', 'DepTime_Display', 'ArrTime_Display']
                if 'Index' in display_df.columns:
                    display_columns.insert(2, 'Index')
                if 'JourneyTime' in display_df.columns:
                    display_columns.append('JourneyTime_Display')
                if 'NumTransfers' in display_df.columns:
                    display_columns.append('NumTransfers')

                # Filter to only existing columns
                display_columns = [col for col in display_columns if col in display_df.columns]

                st.dataframe(
                    display_df[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_journey_time = result_per_pair[
                        'JourneyTime'].mean() if 'JourneyTime' in result_per_pair.columns else 0
                    if avg_journey_time > 0:
                        avg_hours = int(avg_journey_time * 24)
                        avg_minutes = int((avg_journey_time * 24 * 60) % 60)
                        st.metric("Avg Journey Time", f"{avg_hours}h {avg_minutes}m")
                    else:
                        st.metric("Avg Journey Time", "N/A")

                with col2:
                    avg_transfers = result_per_pair[
                        'NumTransfers'].mean() if 'NumTransfers' in result_per_pair.columns else 0
                    st.metric("Avg Transfers", f"{avg_transfers:.1f}")

                with col3:
                    extreme_time = result_per_pair[f'{time_type}_processed'].min() if find_type == 'Earliest' else \
                    result_per_pair[f'{time_type}_processed'].max()
                    st.metric(f"{find_type} {time_type}", extreme_time.strftime("%H:%M:%S"))

                with col4:
                    other_extreme_time = result_per_pair[f'{time_type}_processed'].max() if find_type == 'Earliest' else \
                    result_per_pair[f'{time_type}_processed'].min()
                    other_label = "Latest" if find_type == 'Earliest' else "Earliest"
                    st.metric(f"{other_label} {time_type}", other_extreme_time.strftime("%H:%M:%S"))

            else:
                st.error(f"Could not find {find_type.lower()} times in the filtered data")

            # Option to show all filtered data
            with st.expander("View All Filtered Data"):
                # Prepare display dataframe
                all_display_df = filtered_df.copy()

                # Format time columns for display
                for time_col in ['DepTime', 'ArrTime']:
                    if f'{time_col}_processed' in all_display_df.columns:
                        all_display_df[time_col] = all_display_df[f'{time_col}_processed'].apply(
                            lambda x: x.strftime("%H:%M:%S") if x is not None else "N/A"
                        )

                # Format journey time for display
                if 'JourneyTime' in all_display_df.columns:
                    all_display_df['JourneyTime'] = all_display_df['JourneyTime'].apply(
                        lambda x: f"{int(x * 24)}h {int((x * 24 * 60) % 60)}m" if pd.notna(x) else "N/A"
                    )

                # Remove processed columns
                display_df_clean = all_display_df[
                    [col for col in all_display_df.columns if not col.endswith('_processed')]]

                st.dataframe(display_df_clean, use_container_width=True)

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv_all = display_df_clean.to_csv(index=False)
                    st.download_button(
                        label="Download All Filtered Data as CSV",
                        data=csv_all,
                        file_name="all_filtered_journey_data.csv",
                        mime="text/csv"
                    )

                with col2:
                    if not result_per_pair.empty:
                        result_display = result_per_pair[
                            [col for col in result_per_pair.columns if not col.endswith('_processed')]]
                        csv_result = result_display.to_csv(index=False)
                        st.download_button(
                            label=f"Download {find_type} Times CSV",
                            data=csv_result,
                            file_name=f"{find_type.lower()}_times_per_zone.csv",
                            mime="text/csv"
                        )


if __name__ == "__main__":
    main()