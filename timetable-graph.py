import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import re

# Set page configuration
st.set_page_config(
    page_title="VISUM Journey Timetable Visualizer",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)


def parse_time_string(time_str):
    """Convert time string to minutes since midnight for plotting"""
    if pd.isna(time_str) or time_str == '' or time_str is None:
        return None
    try:
        # Handle HH:MM:SS format
        time_str = str(time_str).strip()
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
        return time_obj.hour * 60 + time_obj.minute + time_obj.second / 60
    except (ValueError, AttributeError):
        try:
            # Try HH:MM format as fallback
            time_obj = datetime.strptime(time_str, '%H:%M')
            return time_obj.hour * 60 + time_obj.minute
        except (ValueError, AttributeError):
            return None


def minutes_to_time_str(minutes):
    """Convert minutes since midnight back to HH:MM format"""
    if pd.isna(minutes):
        return ""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def load_stop_grouping(grouping_file):
    """Load stop grouping configuration from CSV"""
    try:
        group_df = pd.read_csv(grouping_file, sep=';', encoding='utf-8-sig')
        group_df.columns = group_df.columns.str.strip()

        # Expected columns: StopPointNo, GroupName, GroupOrder (optional)
        if 'StopPointNo' in group_df.columns and 'GroupName' in group_df.columns:
            # Convert to numeric to handle different data types
            group_df['StopPointNo'] = pd.to_numeric(group_df['StopPointNo'], errors='coerce')

            # Create mapping from stop number to group
            stop_to_group = dict(zip(group_df['StopPointNo'], group_df['GroupName']))
            # Remove any NaN keys
            stop_to_group = {k: v for k, v in stop_to_group.items() if pd.notna(k)}

            # Create group ordering if available
            group_order = {}
            if 'GroupOrder' in group_df.columns:
                for _, row in group_df.iterrows():
                    if pd.notna(row['StopPointNo']) and pd.notna(row['GroupOrder']):
                        group_name = row['GroupName']
                        if group_name not in group_order:
                            group_order[group_name] = row['GroupOrder']

            return stop_to_group, group_order
        else:
            st.error("Stop grouping file should contain 'StopPointNo' and 'GroupName' columns")
            return None, None
    except Exception as e:
        st.error(f"Error loading stop grouping: {str(e)}")
        return None, None


def load_stoplist(stoplist_file):
    """Load stop list mapping from CSV"""
    try:
        stop_df = pd.read_csv(stoplist_file, sep=';', encoding='utf-8-sig')
        stop_df.columns = stop_df.columns.str.strip()

        # Create mapping from stop number to stop name
        if 'No' in stop_df.columns and 'Name' in stop_df.columns:
            # Convert to numeric to handle different data types
            stop_df['No'] = pd.to_numeric(stop_df['No'], errors='coerce')
            stop_mapping = dict(zip(stop_df['No'], stop_df['Name']))
            # Remove any NaN keys
            stop_mapping = {k: v for k, v in stop_mapping.items() if pd.notna(k)}
            return stop_mapping
        else:
            st.error("Stop list file should contain 'No' and 'Name' columns")
            return None
    except Exception as e:
        st.error(f"Error loading stop list: {str(e)}")
        return None


def load_and_process_data(uploaded_file, stop_mapping=None, stop_grouping=None, group_order=None):
    """Load and process VISUM export data"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')

        # Clean column names (remove any BOM characters)
        df.columns = df.columns.str.strip()

        # Parse time columns
        time_columns = ['Arr', 'Dep', 'ExtArrival', 'ExtDeparture']
        for col in time_columns:
            if col in df.columns:
                df[f'{col}_minutes'] = df[col].apply(parse_time_string)

        # Debug: Check for missing times
        df['has_arrival'] = df['Arr_minutes'].notna()
        df['has_departure'] = df['Dep_minutes'].notna()
        df['has_any_time'] = df['has_arrival'] | df['has_departure']

        missing_times = df[~df['has_any_time']]
        if not missing_times.empty:
            st.warning(f"Found {len(missing_times)} records without valid arrival or departure times")

        # Get stop information - use StopPointNo as stop identifier
        if 'StopPointNo' in df.columns:
            df['stop_id'] = df['StopPointNo']

            # Add stop grouping if available
            if stop_grouping:
                # Convert stop_id to numeric for matching
                df['stop_id_numeric'] = pd.to_numeric(df['stop_id'], errors='coerce')
                df['stop_group'] = df['stop_id_numeric'].map(stop_grouping)

                # Count successful groupings
                grouped_count = df['stop_group'].notna().sum()
                total_records = len(df)
                unique_stops = df['stop_id'].nunique()
                grouped_unique_stops = df[df['stop_group'].notna()]['stop_id'].nunique()
                unique_groups = df['stop_group'].nunique()

                if grouped_count > 0:
                    st.info(
                        f"🏗️ Stop grouping: {grouped_unique_stops}/{unique_stops} stops grouped into {unique_groups} logical stations")
                else:
                    st.warning(f"⚠️ No stop IDs matched the grouping configuration")
            else:
                df['stop_group'] = None

            # Add stop names if mapping is available
            if stop_mapping:
                # Convert stop_id to numeric for matching
                if 'stop_id_numeric' not in df.columns:
                    df['stop_id_numeric'] = pd.to_numeric(df['stop_id'], errors='coerce')
                df['stop_name'] = df['stop_id_numeric'].map(stop_mapping)

                # Count successful mappings
                mapped_count = df['stop_name'].notna().sum()
                total_records = len(df)
                unique_stops = df['stop_id'].nunique()
                mapped_unique_stops = df[df['stop_name'].notna()]['stop_id'].nunique()

                # Create display names
                if stop_grouping:
                    # Prioritize group names, then stop names, then stop IDs
                    df['stop_display'] = df.apply(lambda row:
                                                  row['stop_group'] if pd.notna(row['stop_group'])
                                                  else (f"{row['stop_name']} ({row['stop_id']})" if pd.notna(
                                                      row['stop_name'])
                                                        else str(row['stop_id'])), axis=1)
                else:
                    df['stop_display'] = df.apply(lambda row:
                                                  f"{row['stop_name']} ({row['stop_id']})" if pd.notna(row['stop_name'])
                                                  else str(row['stop_id']), axis=1)

                # Show mapping statistics
                if mapped_count > 0:
                    st.info(
                        f"📍 Stop mapping: {mapped_unique_stops}/{unique_stops} unique stops mapped ({mapped_count}/{total_records} records)")
                else:
                    st.warning(
                        f"⚠️ No stop IDs matched between journey data and stop list. Journey stops: {sorted(df['stop_id'].unique())[:10]}{'...' if unique_stops > 10 else ''}")
            else:
                if stop_grouping:
                    # Use group names when available, otherwise stop IDs
                    df['stop_display'] = df.apply(lambda row:
                                                  row['stop_group'] if pd.notna(row['stop_group'])
                                                  else str(row['stop_id']), axis=1)
                else:
                    df['stop_display'] = df['stop_id'].astype(str)
        else:
            st.error("Required column 'StopPointNo' not found in the data")
            return None

        # Determine direction based on journey patterns
        # We'll analyze if journeys follow the same stop sequence or reverse
        df = determine_journey_direction(df, group_order)

        return df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def determine_journey_direction(df, group_order=None):
    """Determine if journeys are going in different directions"""
    # Group by journey and get stop sequences
    journey_sequences = {}

    for journey_no in df['VehJourneyNo'].unique():
        journey_data = df[df['VehJourneyNo'] == journey_no].sort_values('Index')
        # Use grouped stops if available, otherwise individual stops
        if 'stop_group' in df.columns and df['stop_group'].notna().any():
            # Use groups for direction analysis, but only for grouped stops
            stop_sequence = []
            for _, row in journey_data.iterrows():
                if pd.notna(row['stop_group']):
                    stop_sequence.append(row['stop_group'])
                else:
                    stop_sequence.append(row['stop_id'])
        else:
            stop_sequence = journey_data['stop_id'].tolist()
        journey_sequences[journey_no] = stop_sequence

    # Find the most common sequence length and pattern
    sequences = list(journey_sequences.values())
    if not sequences:
        df['direction'] = 'Direction 1'
        return df

    # Use the first journey as reference
    reference_sequence = sequences[0]

    # Compare other sequences to determine direction
    df['direction'] = 'Direction 1'  # Default

    for journey_no, sequence in journey_sequences.items():
        # Simple heuristic: if sequence is reverse of reference, it's direction 2
        if len(sequence) == len(reference_sequence):
            if sequence == reference_sequence[::-1]:
                df.loc[df['VehJourneyNo'] == journey_no, 'direction'] = 'Direction 2'
            elif sequence != reference_sequence:
                # Different pattern - could be direction 2
                df.loc[df['VehJourneyNo'] == journey_no, 'direction'] = 'Direction 2'

    return df


def create_timetable_plot(df, selected_journeys=None, time_range=None, direction_filter=None):
    """Create the main timetable visualization"""

    # Filter data based on selections
    filtered_df = df.copy()

    if selected_journeys:
        filtered_df = filtered_df[filtered_df['VehJourneyNo'].isin(selected_journeys)]

    if direction_filter and direction_filter != 'All':
        filtered_df = filtered_df[filtered_df['direction'] == direction_filter]

    # Apply time range filter at the data level (exclude journeys outside time range)
    if time_range:
        if len(time_range) == 3:
            start_minutes, end_minutes, is_cross_midnight = time_range
        else:
            start_minutes, end_minutes = time_range
            is_cross_midnight = False

        if is_cross_midnight:
            # Cross-midnight case: include times >= start_minutes OR <= end_minutes
            time_mask = (
                    (filtered_df['Dep_minutes'] >= start_minutes) |
                    (filtered_df['Dep_minutes'] <= end_minutes) |
                    (filtered_df['Arr_minutes'] >= start_minutes) |
                    (filtered_df['Arr_minutes'] <= end_minutes) |
                    (filtered_df['Dep_minutes'].isna() &
                     ((filtered_df['Arr_minutes'] >= start_minutes) | (filtered_df['Arr_minutes'] <= end_minutes))) |
                    (filtered_df['Arr_minutes'].isna() &
                     ((filtered_df['Dep_minutes'] >= start_minutes) | (filtered_df['Dep_minutes'] <= end_minutes)))
            )
        else:
            # Same day case: include times between start and end
            time_mask = (
                    (filtered_df['Dep_minutes'].between(start_minutes, end_minutes, inclusive='both')) |
                    (filtered_df['Arr_minutes'].between(start_minutes, end_minutes, inclusive='both')) |
                    (filtered_df['Dep_minutes'].isna() & filtered_df['Arr_minutes'].between(start_minutes, end_minutes,
                                                                                            inclusive='both')) |
                    (filtered_df['Arr_minutes'].isna() & filtered_df['Dep_minutes'].between(start_minutes, end_minutes,
                                                                                            inclusive='both'))
            )

        filtered_df = filtered_df[time_mask]

        if filtered_df.empty:
            # Return empty plot with message
            fig = go.Figure()
            if is_cross_midnight:
                time_desc = f"({start_minutes // 60:02d}:{start_minutes % 60:02d} - {end_minutes // 60:02d}:{end_minutes % 60:02d} next day)"
            else:
                time_desc = f"({start_minutes // 60:02d}:{start_minutes % 60:02d} - {end_minutes // 60:02d}:{end_minutes % 60:02d})"

            fig.add_annotation(
                text=f"No journeys found in the selected time range<br>{time_desc}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Vehicle Journey Timetable - No Data in Time Range",
                xaxis_title="Time of Day",
                yaxis_title="Stops",
                height=400
            )
            return fig

    # Create the plot
    fig = go.Figure()

    # Get unique stops and sort them by first appearance or group order
    stops_order = []
    stop_display_mapping = {}

    # If we have group ordering, use it to sort the stops
    if 'stop_group' in filtered_df.columns and filtered_df['stop_group'].notna().any():
        # Get unique display names (which are now group names or individual stops)
        unique_displays = []
        display_to_order = {}

        for journey_no in filtered_df['VehJourneyNo'].unique():
            journey_data = filtered_df[filtered_df['VehJourneyNo'] == journey_no].sort_values('Index')
            for _, row in journey_data.iterrows():
                display_name = row.get('stop_display', str(row['stop_id']))
                if display_name not in unique_displays:
                    unique_displays.append(display_name)
                    # If this is a grouped stop, try to get its order
                    if pd.notna(row.get('stop_group')) and hasattr(create_timetable_plot, 'group_order'):
                        group_order = getattr(create_timetable_plot, 'group_order', {})
                        if row['stop_group'] in group_order:
                            display_to_order[display_name] = group_order[row['stop_group']]

        # Sort by group order if available, otherwise keep discovery order
        if display_to_order:
            unique_displays.sort(key=lambda x: display_to_order.get(x, float('inf')))

        stops_order = unique_displays
        stop_display_mapping = {display: display for display in unique_displays}
    else:
        # Original logic for non-grouped stops
        for journey_no in filtered_df['VehJourneyNo'].unique():
            journey_data = filtered_df[filtered_df['VehJourneyNo'] == journey_no].sort_values('Index')
            for _, row in journey_data.iterrows():
                stop_id = row['stop_id']
                if stop_id not in stops_order:
                    stops_order.append(stop_id)
                    stop_display_mapping[stop_id] = row.get('stop_display', str(stop_id))

    # Create a mapping of stops to y-axis positions
    if 'stop_group' in filtered_df.columns and filtered_df['stop_group'].notna().any():
        # For grouped stops, map display names to positions
        stop_positions = {display: i for i, display in enumerate(stops_order)}
    else:
        # For individual stops, map stop IDs to positions
        stop_positions = {stop: i for i, stop in enumerate(stops_order)}

    # Color mapping for directions
    direction_colors = {
        'Direction 1': '#1f77b4',  # Blue
        'Direction 2': '#ff7f0e'  # Orange
    }

    # Plot each journey
    for journey_no in filtered_df['VehJourneyNo'].unique():
        journey_data = filtered_df[filtered_df['VehJourneyNo'] == journey_no].sort_values('Index')

        if journey_data.empty:
            continue

        direction = journey_data['direction'].iloc[0]
        color = direction_colors.get(direction, '#1f77b4')

        # Extract times and stops for this journey
        times = []
        stops = []
        hover_texts = []

        for _, row in journey_data.iterrows():
            # Use departure time if available, otherwise arrival time
            # This ensures we capture terminal stops that only have arrival times
            dep_time_val = row.get('Dep_minutes')
            arr_time_val = row.get('Arr_minutes')

            # Prefer departure time, but use arrival time if departure is not available
            time_val = dep_time_val if pd.notna(dep_time_val) else arr_time_val

            if time_val is not None and not pd.isna(time_val):
                times.append(time_val)

                # Use grouped stop display if available, otherwise individual stop
                if 'stop_group' in filtered_df.columns and pd.notna(row.get('stop_group')):
                    display_key = row.get('stop_display', str(row['stop_id']))
                    stops.append(stop_positions[display_key])
                else:
                    stops.append(stop_positions[row['stop_id']])

                # Create hover text
                arr_time = minutes_to_time_str(row.get('Arr_minutes'))
                dep_time = minutes_to_time_str(row.get('Dep_minutes'))
                stop_display = row.get('stop_display', str(row['stop_id']))

                hover_text = f"Journey: {journey_no}<br>"
                hover_text += f"Stop: {stop_display}<br>"
                if pd.notna(row.get('stop_group')):
                    hover_text += f"Individual Stop: {row['stop_id']}<br>"
                if arr_time:
                    hover_text += f"Arrival: {arr_time}<br>"
                if dep_time:
                    hover_text += f"Departure: {dep_time}<br>"
                else:
                    hover_text += f"Terminal stop<br>"
                hover_text += f"Direction: {direction}"
                hover_texts.append(hover_text)

        if times and stops:
            # Add the journey line
            fig.add_trace(go.Scatter(
                x=times,
                y=stops,
                mode='lines+markers',
                name=f'Journey {journey_no} ({direction})',
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False
            ))

    # Apply time range for x-axis display (this is for visual range, data is already filtered)
    if time_range:
        if len(time_range) == 3:
            start_minutes, end_minutes, is_cross_midnight = time_range
        else:
            start_minutes, end_minutes = time_range
            is_cross_midnight = False

        if is_cross_midnight:
            # For cross-midnight, we need to show two ranges: start to 24:00 and 00:00 to end
            # We'll extend the x-axis to show 24+ hours
            extended_end = end_minutes + 24 * 60  # Add 24 hours to end time
            fig.update_xaxes(range=[start_minutes, extended_end])

            # Create custom ticks that show both day 1 and day 2 times
            tick_interval = 60  # 1 hour intervals
            if (extended_end - start_minutes) <= 180:  # Less than 3 hours total
                tick_interval = 30
            elif (extended_end - start_minutes) <= 360:  # Less than 6 hours total
                tick_interval = 60
            else:
                tick_interval = 120

            time_tickvals = []
            time_ticktext = []

            # First part: from start_minutes to end of day
            current_time = start_minutes
            while current_time < 24 * 60:
                time_tickvals.append(current_time)
                time_ticktext.append(f"{int(current_time // 60):02d}:{int(current_time % 60):02d}")
                current_time += tick_interval

            # Second part: from start of next day to end_minutes (shown as 24+ hours)
            current_time = 24 * 60  # Start of next day
            while current_time <= extended_end:
                time_tickvals.append(current_time)
                actual_hour = int((current_time - 24 * 60) // 60)
                actual_minute = int((current_time - 24 * 60) % 60)
                time_ticktext.append(f"{actual_hour:02d}:{actual_minute:02d}+1")  # +1 indicates next day
                current_time += tick_interval
        else:
            # Normal same-day range
            fig.update_xaxes(range=[start_minutes, end_minutes])

            # Create custom time ticks for the selected range
            tick_interval = 60  # 1 hour intervals
            if end_minutes - start_minutes <= 180:  # Less than 3 hours
                tick_interval = 30  # 30 minute intervals
            elif end_minutes - start_minutes <= 360:  # Less than 6 hours
                tick_interval = 60  # 1 hour intervals
            else:
                tick_interval = 120  # 2 hour intervals

            time_tickvals = list(range(int(start_minutes), int(end_minutes + 1), tick_interval))
            time_ticktext = [f"{int(t // 60):02d}:{int(t % 60):02d}" for t in time_tickvals]
    else:
        # Default full day view
        time_tickvals = list(range(0, 24 * 60, 60))  # Every hour
        time_ticktext = [f"{h:02d}:00" for h in range(24)]

    fig.update_xaxes(
        tickvals=time_tickvals,
        ticktext=time_ticktext,
        tickangle=45
    )

    # Update layout
    fig.update_layout(
        title="Vehicle Journey Timetable",
        xaxis_title="Time of Day",
        yaxis_title="Stops",
        height=max(600, len(stops_order) * 30),  # Dynamic height based on number of stops
        hovermode='closest',
        showlegend=True
    )

    # Format y-axis to show stop names
    if 'stop_group' in filtered_df.columns and filtered_df['stop_group'].notna().any():
        stop_labels = stops_order  # stops_order already contains display names for grouped stops
    else:
        stop_labels = [stop_display_mapping.get(stop, str(stop)) for stop in stops_order]

    fig.update_yaxes(
        tickvals=list(range(len(stops_order))),
        ticktext=stop_labels,
        tickmode='array'
    )

    return fig


def main():
    st.title("🚌 VISUM Vehicle Journey Timetable Visualizer")
    st.markdown("Upload a VISUM vehicle journey export CSV file to visualize public transport timetables")

    # Sidebar for controls
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a VISUM Journey CSV file",
            type=['csv'],
            help="Upload a CSV file exported from VISUM containing vehicle journey data",
            key="journey_file"
        )

        stoplist_file = st.file_uploader(
            "Choose a Stop List CSV file (optional)",
            type=['csv'],
            help="Upload a CSV file with stop numbers and names for better labeling",
            key="stoplist_file"
        )

        stop_grouping_file = st.file_uploader(
            "Choose a Stop Grouping CSV file (optional)",
            type=['csv'],
            help="Upload a CSV file to group multiple stop points into single lines on the graph",
            key="stop_grouping_file"
        )

        if uploaded_file is not None:
            # Load stop grouping if provided
            stop_grouping = None
            group_order = None
            if stop_grouping_file is not None:
                with st.spinner("Loading stop grouping..."):
                    stop_grouping, group_order = load_stop_grouping(stop_grouping_file)
                if stop_grouping:
                    st.success(
                        f"✅ Loaded grouping for {len(stop_grouping)} stops into {len(set(stop_grouping.values()))} groups")
                    # Show sample groupings
                    sample_groups = list(set(stop_grouping.values()))[:3]
                    st.caption(f"Sample groups: {', '.join(sample_groups)}")
                else:
                    st.error("Failed to load stop grouping")

            # Load stop mapping if provided
            stop_mapping = None
            if stoplist_file is not None:
                with st.spinner("Loading stop list..."):
                    stop_mapping = load_stoplist(stoplist_file)
                if stop_mapping:
                    st.success(f"✅ Loaded {len(stop_mapping)} stop names from stop list")
                    # Show sample mappings
                    sample_stops = list(stop_mapping.items())[:3]
                    sample_text = ", ".join([f"{k}: {v[:30]}{'...' if len(v) > 30 else ''}" for k, v in sample_stops])
                    st.caption(f"Sample mappings: {sample_text}")
                else:
                    st.error("Failed to load stop list")

            # Load and process data
            with st.spinner("Processing journey data..."):
                df = load_and_process_data(uploaded_file, stop_mapping, stop_grouping, group_order)

            # Store group_order for use in plotting function
            if group_order:
                create_timetable_plot.group_order = group_order

            if df is not None:
                st.success(f"✅ Loaded {len(df)} records from {df['VehJourneyNo'].nunique()} journeys")

                st.header("🎛️ Filters")

                # Journey selection
                all_journeys = sorted(df['VehJourneyNo'].unique())
                selected_journeys = st.multiselect(
                    "Select Journeys",
                    options=all_journeys,
                    default=all_journeys,  # Default to all journeys
                    help="Select specific journeys to display (all selected by default)"
                )

                # Direction filter
                directions = ['All'] + sorted(df['direction'].unique())
                direction_filter = st.selectbox(
                    "Direction",
                    options=directions,
                    help="Filter by journey direction"
                )

                # Time range filter
                st.subheader("Time Range")

                # Get actual time range from data for better defaults
                min_hour = 0
                max_hour = 24
                min_minute = 0
                max_minute = 0

                if 'Dep_minutes' in df.columns:
                    valid_times = df['Dep_minutes'].dropna()
                    if not valid_times.empty:
                        min_time_minutes = valid_times.min()
                        max_time_minutes = valid_times.max()
                        min_hour = max(0, int(min_time_minutes // 60))
                        max_hour = min(24, int(max_time_minutes // 60) + 1)
                        min_minute = int(min_time_minutes % 60)
                        max_minute = int(max_time_minutes % 60)

                # Time filtering mode selection
                time_filter_mode = st.radio(
                    "Time Filter Mode",
                    options=["Full Day", "Custom Range"],
                    help="Choose how to filter the time display"
                )

                if time_filter_mode == "Custom Range":
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Start Time**")
                        start_hour = st.selectbox(
                            "Hour",
                            options=list(range(24)),
                            index=min_hour,
                            key="start_hour"
                        )
                        start_minute = st.selectbox(
                            "Minute",
                            options=[0, 15, 30, 45],
                            index=0,
                            key="start_minute"
                        )

                    with col2:
                        st.write("**End Time**")
                        end_hour = st.selectbox(
                            "Hour",
                            options=list(range(24)),
                            index=max_hour if max_hour < 24 else 23,
                            key="end_hour"
                        )
                        end_minute = st.selectbox(
                            "Minute",
                            options=[0, 15, 30, 45],
                            index=3,  # Default to 45 minutes
                            key="end_minute"
                        )

                    # Convert to minutes for internal use
                    start_time_minutes = start_hour * 60 + start_minute
                    end_time_minutes = end_hour * 60 + end_minute

                    # Handle cross-midnight ranges
                    if start_time_minutes > end_time_minutes:
                        # Cross-midnight case (e.g., 22:00 to 01:00)
                        st.info(
                            f"📅 Cross-midnight range: {start_hour:02d}:{start_minute:02d} to {end_hour:02d}:{end_minute:02d} (next day)")
                        time_range = (start_time_minutes, end_time_minutes,
                                      True)  # Third parameter indicates cross-midnight
                    elif start_time_minutes == end_time_minutes:
                        st.warning("⚠️ Start and end times cannot be the same")
                        time_range = None
                    else:
                        time_range = (start_time_minutes, end_time_minutes, False)  # Same day
                        st.info(
                            f"📅 Showing journeys from {start_hour:02d}:{start_minute:02d} to {end_hour:02d}:{end_minute:02d}")
                else:
                    time_range = None

                # Data summary
                st.header("📊 Data Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Journeys", df['VehJourneyNo'].nunique())
                    st.metric("Total Stops", df['stop_id'].nunique())
                with col2:
                    st.metric("Direction 1 Journeys",
                              len(df[df['direction'] == 'Direction 1']['VehJourneyNo'].unique()))
                    st.metric("Direction 2 Journeys",
                              len(df[df['direction'] == 'Direction 2']['VehJourneyNo'].unique()))

    # Main content area
    if uploaded_file is not None and 'df' in locals() and df is not None:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🗓️ Timetable Chart", "📋 Raw Data", "ℹ️ Instructions"])

        with tab1:
            if selected_journeys:
                fig = create_timetable_plot(df, selected_journeys, time_range, direction_filter)
                st.plotly_chart(fig, use_container_width=True)

                # Additional statistics
                st.subheader("Selected Data Statistics")
                display_df = df[df['VehJourneyNo'].isin(selected_journeys)]
                if direction_filter and direction_filter != 'All':
                    display_df = display_df[display_df['direction'] == direction_filter]

                # Apply same time filtering for statistics
                if time_range:
                    if len(time_range) == 3:
                        start_minutes, end_minutes, is_cross_midnight = time_range
                    else:
                        start_minutes, end_minutes = time_range
                        is_cross_midnight = False

                    if is_cross_midnight:
                        # Cross-midnight case
                        time_mask = (
                                (display_df['Dep_minutes'] >= start_minutes) |
                                (display_df['Dep_minutes'] <= end_minutes) |
                                (display_df['Arr_minutes'] >= start_minutes) |
                                (display_df['Arr_minutes'] <= end_minutes) |
                                (display_df['Dep_minutes'].isna() &
                                 ((display_df['Arr_minutes'] >= start_minutes) | (
                                             display_df['Arr_minutes'] <= end_minutes))) |
                                (display_df['Arr_minutes'].isna() &
                                 ((display_df['Dep_minutes'] >= start_minutes) | (
                                             display_df['Dep_minutes'] <= end_minutes)))
                        )
                    else:
                        # Same day case
                        time_mask = (
                                (display_df['Dep_minutes'].between(start_minutes, end_minutes, inclusive='both')) |
                                (display_df['Arr_minutes'].between(start_minutes, end_minutes, inclusive='both')) |
                                (display_df['Dep_minutes'].isna() & display_df['Arr_minutes'].between(start_minutes,
                                                                                                      end_minutes,
                                                                                                      inclusive='both')) |
                                (display_df['Arr_minutes'].isna() & display_df['Dep_minutes'].between(start_minutes,
                                                                                                      end_minutes,
                                                                                                      inclusive='both'))
                        )

                    display_df = display_df[time_mask]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Filtered Journeys", display_df['VehJourneyNo'].nunique())
                with col2:
                    st.metric("Filtered Stops", display_df['stop_id'].nunique())
                with col3:
                    valid_times = display_df['Dep_minutes'].dropna()
                    if not valid_times.empty:
                        earliest = minutes_to_time_str(valid_times.min())
                        st.metric("Earliest Departure", earliest)
                    else:
                        st.metric("Earliest Departure", "N/A")
                with col4:
                    if not valid_times.empty:
                        latest = minutes_to_time_str(valid_times.max())
                        st.metric("Latest Departure", latest)
                    else:
                        st.metric("Latest Departure", "N/A")
            else:
                st.warning("Please select at least one journey to display the timetable.")

        with tab2:
            st.subheader("Raw Data Preview")
            st.dataframe(df, use_container_width=True)

            # Download processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Processed Data as CSV",
                data=csv,
                file_name="processed_visum_data.csv",
                mime="text/csv"
            )

        with tab3:
            st.markdown("""
            ## How to Use This App

            ### 1. Data Upload
            - Upload a CSV file exported from VISUM containing vehicle journey data
            - Optionally upload a stop list CSV with stop numbers and names for better labeling
            - Optionally upload a stop grouping CSV to combine multiple stop points into logical stations
            - The journey file should contain columns like: VehJourneyNo, Index, Arr, Dep, StopPointNo
            - The stop list file should contain columns: No, Name
            - The stop grouping file should contain columns: StopPointNo, GroupName, GroupOrder (optional)

            ### 2. Stop Grouping
            - **Purpose**: Combine multiple physical stop points into single logical stations
            - **Example**: Platform A (Stop 123) + Platform B (Stop 124) → "Central Station"
            - **Benefits**: Cleaner visualization, logical station representation
            - **Order**: Use GroupOrder column to control vertical arrangement of groups

            ### 3. Visualization Features
            - **X-axis**: Time of day (24-hour format)
            - **Y-axis**: Stops or grouped stations (with names if provided)
            - **Lines**: Each line represents one vehicle journey
            - **Colors**: Different colors represent different directions
            - **Grouping**: Multiple stop points appear as single lines when grouped

            ### 4. Interactive Controls
            - **Journey Selection**: Choose specific journeys to display
            - **Direction Filter**: Filter by journey direction (useful for round-trips)
            - **Time Range**: Choose between full day or custom time range
            - **Custom Time Range**: Select specific start and end times (excludes journeys outside this range)
            - **Hover**: Hover over points to see detailed information including individual stop IDs

            ### 5. Data Processing
            - The app automatically detects journey directions based on stop sequences
            - Times are parsed from HH:MM:SS format
            - Stop names are mapped from the stop list if provided
            - Stop grouping combines multiple physical stops into logical stations
            - Missing data is handled gracefully

            ### 6. Export Options
            - Download the processed data as CSV
            - Use browser's print function to save charts as PDF

            ### Supported File Formats
            **Journey CSV** should be semicolon-separated with these key columns:
            - `VehJourneyNo`: Journey identifier
            - `Index`: Stop sequence within journey
            - `Arr`: Arrival time (HH:MM:SS)
            - `Dep`: Departure time (HH:MM:SS)
            - `StopPointNo`: Stop identifier

            **Stop List CSV** should be semicolon-separated with:
            - `No`: Stop number (matches StopPointNo in journey data)
            - `Name`: Stop name for display

            **Stop Grouping CSV** should be semicolon-separated with:
            - `StopPointNo`: Individual stop number (matches journey data)
            - `GroupName`: Name of the logical station/group
            - `GroupOrder`: (Optional) Number to control vertical order of groups in chart
            """)

    else:
        # Show example and instructions when no file is uploaded
        st.markdown("""
        ## Welcome to the VISUM Timetable Visualizer! 🚌

        This app helps you create beautiful, interactive timetable charts from VISUM vehicle journey exports.

        ### Features:
        - 📊 **Interactive Timetables**: Visualize vehicle journeys with time on X-axis and stops on Y-axis
        - 🏷️ **Stop Name Mapping**: Upload a stop list to show readable stop names instead of numbers
        - 🔄 **Direction Detection**: Automatically identifies outbound and return journeys
        - 🎛️ **Flexible Filtering**: Filter by journey, direction, and time range
        - 📱 **Responsive Design**: Works on desktop and mobile devices
        - 💾 **Data Export**: Download processed data for further analysis

        ### Getting Started:
        1. Upload your VISUM journey CSV export using the first file uploader in the sidebar
        2. Optionally upload a stop list CSV for better stop names using the second uploader
        3. Use the filters to focus on specific journeys or time periods
        4. Explore the interactive timetable chart
        5. Export your results if needed

        **👈 Start by uploading files in the sidebar!**
        """)


if __name__ == "__main__":
    main()