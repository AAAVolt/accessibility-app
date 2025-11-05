import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import io


def convert_time_to_display(time_str):
    """Convert time string to display format"""
    if pd.isna(time_str) or time_str == '' or str(time_str).lower() == 'nan':
        return "N/A"

    try:
        if isinstance(time_str, str) and ':' in time_str:
            return time_str

        if hasattr(time_str, 'total_seconds'):
            total_seconds = int(time_str.total_seconds())
            hours = (total_seconds // 3600) % 24
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        dt = pd.to_datetime(time_str)
        return dt.strftime("%H:%M:%S")

    except:
        return str(time_str)


def convert_journey_time_to_display(journey_time):
    """Convert journey time to hours and minutes format"""
    if pd.isna(journey_time) or journey_time == '' or str(journey_time).lower() == 'nan':
        return "N/A"

    try:
        if isinstance(journey_time, str) and ('h' in journey_time or 'm' in journey_time):
            return journey_time

        if isinstance(journey_time, (int, float)):
            if journey_time < 1:
                hours = int(journey_time * 24)
                minutes = int((journey_time * 24 * 60) % 60)
                return f"{hours}h {minutes}m"
            else:
                hours = int(journey_time)
                minutes = int((journey_time * 60) % 60)
                return f"{hours}h {minutes}m"

        return str(journey_time)

    except:
        return str(journey_time)


def convert_time_for_comparison(time_str):
    """Convert time string to comparable format"""
    if pd.isna(time_str) or time_str == '' or str(time_str).lower() == 'nan':
        return None

    try:
        if isinstance(time_str, str) and ':' in time_str:
            return pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
        return pd.to_datetime(time_str, errors='coerce')
    except:
        return None


def load_and_process_municipality_data(uploaded_file):
    """Load Excel file and process municipality data"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)

        st.write("**Debug Info:**")
        st.write(f"Original shape: {df.shape}")
        st.write(f"Column names: {list(df.columns)}")
        st.write("First few rows:")
        st.dataframe(df.head())

        df = df.dropna(how='all').reset_index(drop=True)
        df.columns = [str(col).strip() if col is not None else f'Col_{i}' for i, col in enumerate(df.columns)]

        required_cols = [
            'Municipio',
            'Latest_DepTime', 'Latest_ArrTime', 'Latest_JourneyTime', 'Latest_NumTransfers',
            'Earliest_DepTime', 'Earliest_ArrTime', 'Earliest_JourneyTime', 'Earliest_NumTransfers'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.warning(f"Some columns are missing: {missing_cols}")

        return df, None

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error loading file: {str(e)}\n\nDetails:\n{error_details}"


def aggregate_by_municipality(df):
    """Aggregate data by municipality - find extreme times for Latest and Earliest"""
    if df.empty:
        return pd.DataFrame()

    if 'Municipio' not in df.columns:
        st.error("Municipio column not found")
        return pd.DataFrame()

    st.write("**Sample time data from source:**")
    time_cols = ['Latest_DepTime', 'Latest_ArrTime', 'Earliest_DepTime', 'Earliest_ArrTime']
    available_time_cols = [col for col in time_cols if col in df.columns]
    if available_time_cols:
        st.dataframe(df[['Municipio'] + available_time_cols].head())

    grouped = df.groupby('Municipio')
    result_data = []

    for municipio, group in grouped:
        try:
            row_data = {'Municipio': municipio}

            # Latest trip: Find the row with EARLIEST Latest_DepTime
            if 'Latest_DepTime' in group.columns:
                group_copy = group.copy()

                valid_times = group_copy[
                    group_copy['Latest_DepTime'].notna() &
                    (group_copy['Latest_DepTime'] != '') &
                    (group_copy['Latest_DepTime'] != 'NaT')
                    ]

                if len(valid_times) > 0:
                    try:
                        valid_times['Latest_DepTime_converted'] = valid_times['Latest_DepTime'].apply(
                            convert_time_for_comparison
                        )

                        if valid_times['Latest_DepTime_converted'].isna().all():
                            latest_idx = valid_times['Latest_DepTime'].astype(str).idxmin()
                        else:
                            latest_idx = valid_times['Latest_DepTime_converted'].idxmin()

                        latest_row = group.loc[latest_idx]
                        row_data['latest_dep_time'] = latest_row.get('Latest_DepTime', 'N/A')
                        row_data['latest_arr_time'] = latest_row.get('Latest_ArrTime', 'N/A')
                        row_data['latest_journey_time'] = latest_row.get('Latest_JourneyTime', 'N/A')
                        row_data['latest_num_transfers'] = latest_row.get('Latest_NumTransfers', 'N/A')

                    except Exception as e:
                        st.warning(f"Error processing Latest times for {municipio}: {e}")
                        row_data['latest_dep_time'] = 'N/A'
                        row_data['latest_arr_time'] = 'N/A'
                        row_data['latest_journey_time'] = 'N/A'
                        row_data['latest_num_transfers'] = 'N/A'
                else:
                    row_data['latest_dep_time'] = 'N/A'
                    row_data['latest_arr_time'] = 'N/A'
                    row_data['latest_journey_time'] = 'N/A'
                    row_data['latest_num_transfers'] = 'N/A'

            # Earliest trip: Find the row with LATEST Earliest_ArrTime
            if 'Earliest_ArrTime' in group.columns:
                group_copy = group.copy()

                valid_times = group_copy[
                    group_copy['Earliest_ArrTime'].notna() &
                    (group_copy['Earliest_ArrTime'] != '') &
                    (group_copy['Earliest_ArrTime'] != 'NaT')
                    ]

                if len(valid_times) > 0:
                    try:
                        valid_times['Earliest_ArrTime_converted'] = valid_times['Earliest_ArrTime'].apply(
                            convert_time_for_comparison
                        )

                        if valid_times['Earliest_ArrTime_converted'].isna().all():
                            earliest_idx = valid_times['Earliest_ArrTime'].astype(str).idxmax()
                        else:
                            earliest_idx = valid_times['Earliest_ArrTime_converted'].idxmax()

                        earliest_row = group.loc[earliest_idx]
                        row_data['earliest_dep_time'] = earliest_row.get('Earliest_DepTime', 'N/A')
                        row_data['earliest_arr_time'] = earliest_row.get('Earliest_ArrTime', 'N/A')
                        row_data['earliest_journey_time'] = earliest_row.get('Earliest_JourneyTime', 'N/A')
                        row_data['earliest_num_transfers'] = earliest_row.get('Earliest_NumTransfers', 'N/A')

                    except Exception as e:
                        st.warning(f"Error processing Earliest times for {municipio}: {e}")
                        row_data['earliest_dep_time'] = 'N/A'
                        row_data['earliest_arr_time'] = 'N/A'
                        row_data['earliest_journey_time'] = 'N/A'
                        row_data['earliest_num_transfers'] = 'N/A'
                else:
                    row_data['earliest_dep_time'] = 'N/A'
                    row_data['earliest_arr_time'] = 'N/A'
                    row_data['earliest_journey_time'] = 'N/A'
                    row_data['earliest_num_transfers'] = 'N/A'

            result_data.append(row_data)

        except Exception as e:
            st.warning(f"Error processing municipality {municipio}: {e}")
            continue

    if not result_data:
        st.warning("No valid data could be aggregated. Check the time column formats.")
        return pd.DataFrame()

    result_df = pd.DataFrame(result_data)

    st.write(f"**Aggregated {len(result_df)} municipalities successfully**")

    # Format display columns
    time_cols_to_format = ['latest_dep_time', 'latest_arr_time', 'earliest_dep_time', 'earliest_arr_time']
    for col in time_cols_to_format:
        if col in result_df.columns:
            result_df[f'{col}_display'] = result_df[col].apply(convert_time_to_display)

    journey_cols_to_format = ['latest_journey_time', 'earliest_journey_time']
    for col in journey_cols_to_format:
        if col in result_df.columns:
            result_df[f'{col}_display'] = result_df[col].apply(convert_journey_time_to_display)

    return result_df


def filter_municipalities(df, municipio_filter_type, municipio_values):
    """Filter data by municipality"""
    filtered_df = df.copy()

    if municipio_filter_type == "Specific Municipalities" and municipio_values:
        filtered_df = filtered_df[filtered_df['Municipio'].isin(municipio_values)]
    elif municipio_filter_type == "Contains Text" and municipio_values:
        pattern = '|'.join(municipio_values)
        filtered_df = filtered_df[filtered_df['Municipio'].str.contains(pattern, case=False, na=False)]

    return filtered_df


def main():
    st.set_page_config(
        page_title="Municipality Journey Analyzer",
        page_icon="🏘️",
        layout="wide"
    )

    st.title("🏘️ Municipality Journey Analyzer")
    st.markdown("Analyze the earliest and latest trips per municipality")

    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with municipality journey data"
    )

    if uploaded_file is not None:
        with st.spinner("Loading and processing data..."):
            df, error = load_and_process_municipality_data(uploaded_file)

        if error:
            st.error(f"Error loading file: {error}")
            return

        if df is None or df.empty:
            st.error("No valid data found in the file")
            return

        st.success(f"✅ Data loaded successfully! Found {len(df)} rows")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Unique Municipalities", df['Municipio'].nunique() if 'Municipio' in df.columns else 0)

        st.divider()

        # Process and display results
        st.subheader("📊 Municipality Analysis Results")

        with st.spinner("Aggregating data by municipality..."):
            # Process both analysis types in one pass
            aggregated_df = aggregate_by_municipality(df)

        if aggregated_df.empty:
            st.warning("No data available for aggregation")
        else:
            filtered_df = aggregated_df

            if filtered_df.empty:
                st.warning("No municipalities match the selected filters")
            else:
                st.success(f"Found {len(filtered_df)} municipalities matching your criteria")

                display_df = filtered_df.copy()

                display_final = display_df[[
                    'Municipio',
                    'latest_dep_time_display', 'latest_arr_time_display',
                    'latest_journey_time_display', 'latest_num_transfers',
                    'earliest_dep_time_display', 'earliest_arr_time_display',
                    'earliest_journey_time_display', 'earliest_num_transfers'
                ]].copy()

                display_final = display_final.rename(columns={
                    'latest_dep_time_display': 'Latest Dep Time',
                    'latest_arr_time_display': 'Latest Arr Time',
                    'latest_journey_time_display': 'Latest Journey Time',
                    'latest_num_transfers': 'Latest Transfers',
                    'earliest_dep_time_display': 'Earliest Dep Time',
                    'earliest_arr_time_display': 'Earliest Arr Time',
                    'earliest_journey_time_display': 'Earliest Journey Time',
                    'earliest_num_transfers': 'Earliest Transfers'
                })

                st.dataframe(display_final, use_container_width=True, hide_index=True)

                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    total_municipalities = len(filtered_df)
                    st.metric("Municipalities", total_municipalities)

                with col2:
                    avg_transfers = filtered_df['NumTransfers'].mean() if 'NumTransfers' in filtered_df.columns else 0
                    st.metric("Avg Transfers", f"{avg_transfers:.1f}")

                with col3:
                    latest_transfers = filtered_df[
                        'latest_num_transfers'].mean() if 'latest_num_transfers' in filtered_df.columns else 0
                    st.metric("Avg Latest Transfers", f"{latest_transfers:.1f}")

                # Download option
                st.subheader("💾 Download Results")
                csv = display_final.to_csv(index=False)
                st.download_button(
                    label="Download Municipality Analysis as CSV",
                    data=csv,
                    file_name=f"municipality_{analysis_type.lower()}_analysis.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()