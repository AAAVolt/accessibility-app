import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io


def load_data(uploaded_file):
    """Load and validate skim matrix data from CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        # Validate required columns
        required_cols = ['OrigZoneNo', 'DestZoneNo', 'ACD', 'ACT', 'EGD', 'EGT',
                         'JRD', 'JRT', 'NTR', 'RID', 'RIT', 'SFQ', 'TWT']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def load_zone_metadata(uploaded_file):
    """Load zone metadata (No, Name, Population) from CSV file"""
    try:
        df = pd.read_csv(uploaded_file, sep=';')  # Using semicolon separator as specified

        # Validate required columns (flexible column names)
        df.columns = df.columns.str.strip()  # Remove whitespace

        # Check for required columns with flexible naming
        required_info = {'zone_no': None, 'name': None, 'population': None}

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['no', 'zone_no', 'zoneno', 'zone']:
                required_info['zone_no'] = col
            elif col_lower in ['name', 'zone_name', 'zonename']:
                required_info['name'] = col
            elif col_lower in ['population', 'pop', 'inhabitants']:
                required_info['population'] = col

        missing = [k for k, v in required_info.items() if v is None]
        if missing:
            st.error(f"Missing columns in zone metadata: {missing}. Expected columns like 'No', 'Name', 'Population'")
            return None

        # Standardize column names
        zone_df = df[[required_info['zone_no'], required_info['name'], required_info['population']]].copy()
        zone_df.columns = ['ZoneNo', 'ZoneName', 'Population']

        # Ensure ZoneNo is integer and Population is numeric
        zone_df['ZoneNo'] = pd.to_numeric(zone_df['ZoneNo'], errors='coerce')
        zone_df['Population'] = pd.to_numeric(zone_df['Population'], errors='coerce')

        # Remove rows with invalid data
        zone_df = zone_df.dropna()

        return zone_df

    except Exception as e:
        st.error(f"Error loading zone metadata: {str(e)}")
        return None


def compare_scenarios(df1, df2, dest_zone, zone_metadata=None):
    """Compare two scenarios for a specific destination zone"""
    # Filter data for the selected destination zone
    df1_filtered = df1[df1['DestZoneNo'] == dest_zone].copy()
    df2_filtered = df2[df2['DestZoneNo'] == dest_zone].copy()

    # Merge on origin zone
    merged = pd.merge(df1_filtered, df2_filtered, on='OrigZoneNo', suffixes=('_scenario1', '_scenario2'))

    if merged.empty:
        return None, None

    # Add zone metadata if available
    if zone_metadata is not None:
        # Add origin zone names and population
        merged = merged.merge(
            zone_metadata[['ZoneNo', 'ZoneName', 'Population']],
            left_on='OrigZoneNo',
            right_on='ZoneNo',
            how='left'
        ).drop('ZoneNo', axis=1)
        merged = merged.rename(columns={'ZoneName': 'OriginZoneName', 'Population': 'OriginPopulation'})

        # Add destination zone name
        dest_zone_info = zone_metadata[zone_metadata['ZoneNo'] == dest_zone]
        if not dest_zone_info.empty:
            merged['DestinationZoneName'] = dest_zone_info.iloc[0]['ZoneName']
        else:
            merged['DestinationZoneName'] = f"Zone {dest_zone}"
    else:
        merged['OriginZoneName'] = merged['OrigZoneNo'].apply(lambda x: f"Zone {x}")
        merged['OriginPopulation'] = None
        merged['DestinationZoneName'] = f"Zone {dest_zone}"

    # Calculate differences for each metric
    metrics = ['ACD', 'ACT', 'EGD', 'EGT', 'JRD', 'JRT', 'NTR', 'RID', 'RIT', 'SFQ', 'TWT']

    for metric in metrics:
        merged[f'{metric}_diff'] = merged[f'{metric}_scenario2'] - merged[f'{metric}_scenario1']
        merged[f'{metric}_pct_change'] = ((merged[f'{metric}_scenario2'] - merged[f'{metric}_scenario1']) /
                                          merged[f'{metric}_scenario1'].replace(0, np.nan) * 100)

    # Identify all zones with any changes (not just "significant" ones)
    changes = merged[
        (abs(merged['JRT_diff']) > 0.01) |  # Any journey time change > 0.01 min
        (abs(merged['JRD_diff']) > 0.01) |  # Any journey distance change > 0.01 km
        (abs(merged['NTR_diff']) > 0.01) |  # Any transfer change > 0.01
        (abs(merged['SFQ_diff']) > 0.01) |  # Any service frequency change > 0.01
        (abs(merged['TWT_diff']) > 0.01)  # Any transfer wait time change > 0.01
        ].copy()

    # Sort by absolute journey time change (descending)
    changes = changes.sort_values(by='JRT_diff', key=abs, ascending=False)

    return merged, changes


def create_comparison_charts(comparison_df, dest_zone, zone_metadata=None):
    """Create visualization charts for comparison"""

    if comparison_df.empty:
        return None

    # Prepare display names for x-axis
    if 'OriginZoneName' in comparison_df.columns:
        x_labels = comparison_df['OriginZoneName']
        x_title = "Origin Zone"
        hover_template = '<b>%{text}</b><br>Value: %{y}<extra></extra>'
    else:
        x_labels = comparison_df['OrigZoneNo']
        x_title = "Origin Zone Number"
        hover_template = '<b>Zone %{text}</b><br>Value: %{y}<extra></extra>'

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Journey Time Comparison', 'Journey Distance Comparison',
                        'Number of Transfers Comparison', 'Service Frequency Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Journey Time
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['JRT_scenario1'],
                   mode='markers',
                   name='Scenario 1 - Journey Time',
                   marker=dict(color='blue', size=8),
                   text=x_labels,
                   hovertemplate=hover_template),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['JRT_scenario2'],
                   mode='markers',
                   name='Scenario 2 - Journey Time',
                   marker=dict(color='red', size=8),
                   text=x_labels,
                   hovertemplate=hover_template),
        row=1, col=1
    )

    # Journey Distance
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['JRD_scenario1'],
                   mode='markers',
                   name='Scenario 1 - Journey Distance',
                   marker=dict(color='blue', size=8),
                   showlegend=False,
                   text=x_labels,
                   hovertemplate=hover_template),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['JRD_scenario2'],
                   mode='markers',
                   name='Scenario 2 - Journey Distance',
                   marker=dict(color='red', size=8),
                   showlegend=False,
                   text=x_labels,
                   hovertemplate=hover_template),
        row=1, col=2
    )

    # Number of Transfers
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['NTR_scenario1'],
                   mode='markers',
                   name='Scenario 1 - Transfers',
                   marker=dict(color='blue', size=8),
                   showlegend=False,
                   text=x_labels,
                   hovertemplate=hover_template),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['NTR_scenario2'],
                   mode='markers',
                   name='Scenario 2 - Transfers',
                   marker=dict(color='red', size=8),
                   showlegend=False,
                   text=x_labels,
                   hovertemplate=hover_template),
        row=2, col=1
    )

    # Service Frequency
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['SFQ_scenario1'],
                   mode='markers',
                   name='Scenario 1 - Service Frequency',
                   marker=dict(color='blue', size=8),
                   showlegend=False,
                   text=x_labels,
                   hovertemplate=hover_template),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_labels,
                   y=comparison_df['SFQ_scenario2'],
                   mode='markers',
                   name='Scenario 2 - Service Frequency',
                   marker=dict(color='red', size=8),
                   showlegend=False,
                   text=x_labels,
                   hovertemplate=hover_template),
        row=2, col=2
    )

    dest_name = comparison_df.iloc[0][
        'DestinationZoneName'] if 'DestinationZoneName' in comparison_df.columns else f'Zone {dest_zone}'

    fig.update_layout(
        title=f'Comparison of Key Metrics - Destination: {dest_name}',
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text=x_title, row=1, col=1)
    fig.update_xaxes(title_text=x_title, row=1, col=2)
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    fig.update_xaxes(title_text=x_title, row=2, col=2)

    fig.update_yaxes(title_text="Journey Time (min)", row=1, col=1)
    fig.update_yaxes(title_text="Journey Distance (km)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Transfers", row=2, col=1)
    fig.update_yaxes(title_text="Service Frequency", row=2, col=2)

    return fig


def create_difference_chart(comparison_df, dest_zone):
    """Create chart showing differences between scenarios"""

    if comparison_df.empty:
        return None

    # Prepare display names for x-axis
    if 'OriginZoneName' in comparison_df.columns:
        x_labels = comparison_df['OriginZoneName']
        x_title = "Origin Zone"
        hover_template = '<b>%{text}</b><br>Change: %{y:.1f}%<extra></extra>'
    else:
        x_labels = comparison_df['OrigZoneNo']
        x_title = "Origin Zone Number"
        hover_template = '<b>Zone %{text}</b><br>Change: %{y:.1f}%<extra></extra>'

    # Create bar chart for journey time differences
    fig = go.Figure()

    # Add bars for journey time percentage change
    fig.add_trace(go.Bar(
        x=x_labels,
        y=comparison_df['JRT_pct_change'],
        name='Journey Time % Change',
        marker_color=['red' if x > 0 else 'green' for x in comparison_df['JRT_pct_change']],
        text=x_labels,
        hovertemplate=hover_template
    ))

    dest_name = comparison_df.iloc[0][
        'DestinationZoneName'] if 'DestinationZoneName' in comparison_df.columns else f'Zone {dest_zone}'

    fig.update_layout(
        title=f'Journey Time Percentage Change - Destination: {dest_name}',
        xaxis_title=x_title,
        yaxis_title='Percentage Change (%)',
        height=400
    )

    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    return fig


def create_population_analysis_chart(changes_df):
    """Create chart showing relationship between population and changes"""

    if changes_df.empty or 'OriginPopulation' not in changes_df.columns:
        return None

    # Filter out zones without population data
    pop_data = changes_df.dropna(subset=['OriginPopulation'])

    if pop_data.empty:
        return None

    # Create scatter plot
    fig = go.Figure()

    # Add scatter plot with population vs journey time change
    fig.add_trace(go.Scatter(
        x=pop_data['OriginPopulation'],
        y=pop_data['JRT_pct_change'],
        mode='markers',
        marker=dict(
            size=10,
            color=pop_data['JRT_pct_change'],
            colorscale='RdYlGn_r',
            colorbar=dict(title="Journey Time<br>Change (%)"),
            showscale=True
        ),
        text=pop_data['OriginZoneName'] if 'OriginZoneName' in pop_data.columns else pop_data['OrigZoneNo'],
        hovertemplate='<b>%{text}</b><br>Population: %{x:,.0f}<br>Journey Time Change: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Journey Time Change vs Zone Population',
        xaxis_title='Zone Population',
        yaxis_title='Journey Time Change (%)',
        height=500
    )

    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    return fig


def main():
    st.set_page_config(page_title="PTV Visum Skim Matrix Analyzer", layout="wide")

    st.title("🚌 PTV Visum Skim Matrix Analyzer")
    st.markdown("Compare two skim matrix scenarios with zone names and population analysis")

    # Sidebar for file uploads
    st.sidebar.header("📁 Data Upload")

    scenario1_file = st.sidebar.file_uploader(
        "Upload Scenario 1 CSV",
        type=['csv'],
        help="Upload the first skim matrix CSV file"
    )

    scenario2_file = st.sidebar.file_uploader(
        "Upload Scenario 2 CSV",
        type=['csv'],
        help="Upload the second skim matrix CSV file"
    )

    zone_metadata_file = st.sidebar.file_uploader(
        "Upload Zone Metadata (Optional)",
        type=['csv'],
        help="CSV file with zone No;Name;Population (semicolon separated)"
    )

    # Load zone metadata if provided
    zone_metadata = None
    if zone_metadata_file is not None:
        with st.spinner("Loading zone metadata..."):
            zone_metadata = load_zone_metadata(zone_metadata_file)

        if zone_metadata is not None:
            st.sidebar.success(f"✅ Zone metadata loaded: {len(zone_metadata)} zones")

            # Show preview of zone metadata
            with st.sidebar.expander("📋 Zone Metadata Preview"):
                st.dataframe(zone_metadata.head(), use_container_width=True)

    if scenario1_file is not None and scenario2_file is not None:
        # Load data
        with st.spinner("Loading scenario data..."):
            df1 = load_data(scenario1_file)
            df2 = load_data(scenario2_file)

        if df1 is not None and df2 is not None:
            st.success("✅ Scenario data loaded successfully!")

            # Show basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Scenario 1 - Total O-D Pairs", len(df1))
                st.metric("Scenario 1 - Unique Zones", df1['OrigZoneNo'].nunique())

            with col2:
                st.metric("Scenario 2 - Total O-D Pairs", len(df2))
                st.metric("Scenario 2 - Unique Zones", df2['OrigZoneNo'].nunique())

            with col3:
                if zone_metadata is not None:
                    st.metric("Zone Metadata", len(zone_metadata))
                    total_population = zone_metadata['Population'].sum()
                    st.metric("Total Population", f"{total_population:,.0f}")
                else:
                    st.info("No zone metadata loaded")

            # Destination zone selection
            st.sidebar.header("🎯 Analysis Settings")

            # Get common destination zones
            dest_zones_1 = set(df1['DestZoneNo'].unique())
            dest_zones_2 = set(df2['DestZoneNo'].unique())
            common_dest_zones = sorted(list(dest_zones_1.intersection(dest_zones_2)))

            if not common_dest_zones:
                st.error("No common destination zones found between scenarios!")
                return

            # Create destination zone options with names if available
            if zone_metadata is not None:
                dest_zone_options = {}
                for zone_no in common_dest_zones:
                    zone_info = zone_metadata[zone_metadata['ZoneNo'] == zone_no]
                    if not zone_info.empty:
                        zone_name = zone_info.iloc[0]['ZoneName']
                        dest_zone_options[f"{zone_name} (Zone {zone_no})"] = zone_no
                    else:
                        dest_zone_options[f"Zone {zone_no}"] = zone_no

                selected_dest_display = st.sidebar.selectbox(
                    "Select Destination Zone",
                    list(dest_zone_options.keys()),
                    help="Choose the destination zone to analyze"
                )
                selected_dest_zone = dest_zone_options[selected_dest_display]
            else:
                selected_dest_zone = st.sidebar.selectbox(
                    "Select Destination Zone",
                    common_dest_zones,
                    help="Choose the destination zone to analyze"
                )

            # Analysis threshold settings
            st.sidebar.subheader("🔧 Analysis Settings")
            show_all_changes = st.sidebar.checkbox("Show all zones with changes", value=True)
            min_change_threshold = st.sidebar.slider(
                "Minimum change to display (minutes)",
                min_value=0.0,
                max_value=10.0,
                value=0.1,
                step=0.1,
                help="Only show zones with journey time changes above this threshold"
            )

            # Comparison analysis
            if st.sidebar.button("🔍 Analyze Scenarios"):
                with st.spinner("Analyzing scenarios..."):
                    comparison_df, changes = compare_scenarios(df1, df2, selected_dest_zone, zone_metadata)

                if comparison_df is not None:
                    # Filter changes based on threshold
                    if not show_all_changes:
                        changes = changes[abs(changes['JRT_diff']) >= min_change_threshold]

                    dest_name = comparison_df.iloc[0][
                        'DestinationZoneName'] if 'DestinationZoneName' in comparison_df.columns else f'Zone {selected_dest_zone}'

                    st.header(f"📊 Analysis Results for Destination: {dest_name}")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        avg_time_change = comparison_df['JRT_pct_change'].mean()
                        st.metric(
                            "Avg Journey Time Change",
                            f"{avg_time_change:.1f}%",
                            delta=f"{avg_time_change:.1f}%"
                        )

                    with col2:
                        avg_dist_change = comparison_df['JRD_pct_change'].mean()
                        st.metric(
                            "Avg Journey Distance Change",
                            f"{avg_dist_change:.1f}%",
                            delta=f"{avg_dist_change:.1f}%"
                        )

                    with col3:
                        zones_with_changes = len(changes)
                        st.metric(
                            "Zones with Changes",
                            zones_with_changes,
                            delta=f"out of {len(comparison_df)}"
                        )

                    with col4:
                        avg_transfer_change = comparison_df['NTR_diff'].mean()
                        st.metric(
                            "Avg Transfer Change",
                            f"{avg_transfer_change:.2f}",
                            delta=f"{avg_transfer_change:.2f}"
                        )

                    # Charts
                    st.subheader("📈 Scenario Comparison Charts")

                    comparison_chart = create_comparison_charts(comparison_df, selected_dest_zone, zone_metadata)
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)

                    # Difference chart
                    st.subheader("📊 Journey Time Changes")
                    difference_chart = create_difference_chart(comparison_df, selected_dest_zone)
                    if difference_chart:
                        st.plotly_chart(difference_chart, use_container_width=True)

                    # Population analysis if available
                    if zone_metadata is not None and 'OriginPopulation' in changes.columns:
                        st.subheader("👥 Population vs Journey Time Changes")
                        pop_chart = create_population_analysis_chart(changes)
                        if pop_chart:
                            st.plotly_chart(pop_chart, use_container_width=True)
                        else:
                            st.info("No population data available for analysis")

                    # Changes table
                    if not changes.empty:
                        st.subheader("📋 Zones with Changes")

                        # Select relevant columns for display
                        if zone_metadata is not None:
                            display_cols = ['OriginZoneName', 'OrigZoneNo', 'OriginPopulation',
                                            'JRT_scenario1', 'JRT_scenario2', 'JRT_diff', 'JRT_pct_change',
                                            'JRD_scenario1', 'JRD_scenario2', 'JRD_diff', 'JRD_pct_change',
                                            'NTR_scenario1', 'NTR_scenario2', 'NTR_diff']
                            column_names = ['Zone Name', 'Zone No', 'Population',
                                            'JT S1', 'JT S2', 'JT Diff', 'JT Change %',
                                            'JD S1', 'JD S2', 'JD Diff', 'JD Change %',
                                            'Transfers S1', 'Transfers S2', 'Transfer Diff']
                        else:
                            display_cols = ['OrigZoneNo', 'JRT_scenario1', 'JRT_scenario2', 'JRT_diff',
                                            'JRT_pct_change',
                                            'JRD_scenario1', 'JRD_scenario2', 'JRD_diff', 'JRD_pct_change',
                                            'NTR_scenario1', 'NTR_scenario2', 'NTR_diff']
                            column_names = ['Zone No', 'JT S1', 'JT S2', 'JT Diff', 'JT Change %',
                                            'JD S1', 'JD S2', 'JD Diff', 'JD Change %',
                                            'Transfers S1', 'Transfers S2', 'Transfer Diff']

                        display_df = changes[display_cols].round(2)
                        display_df.columns = column_names

                        st.dataframe(display_df, use_container_width=True)

                        # Download button for changes
                        csv_buffer = io.StringIO()
                        display_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Changes Data",
                            data=csv_buffer.getvalue(),
                            file_name=f"changes_dest_{selected_dest_zone}.csv",
                            mime="text/csv"
                        )

                    else:
                        st.info("No zones with changes above the specified threshold.")

                    # Full comparison data download
                    st.subheader("💾 Export Results")
                    csv_buffer_full = io.StringIO()
                    comparison_df.to_csv(csv_buffer_full, index=False)
                    st.download_button(
                        label="📥 Download Full Comparison Data",
                        data=csv_buffer_full.getvalue(),
                        file_name=f"full_comparison_dest_{selected_dest_zone}.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning(f"No data found for destination zone {selected_dest_zone}")

    else:
        st.info("👆 Please upload both scenario CSV files to begin analysis")

        # Show example data formats
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Skim Matrix CSV Format")
            st.markdown("Your skim matrix CSV files should have this structure:")

            example_data = """OrigZoneNo,DestZoneNo,ACD,ACT,EGD,EGT,JRD,JRT,NTR,RID,RIT,SFQ,TWT
1,1,0.000,0.00,0.000,0.00,0.000,0.00,0.000,0.000,0.00,0.000,0.00
1,2,0.037,0.37,0.761,7.62,53.281,94.08,1.000,52.483,86.10,25.000,3.68
1,3,0.037,0.37,0.470,4.70,19.717,36.75,0.000,19.210,31.68,22.000,0.00"""

            st.code(example_data, language="csv")

        with col2:
            st.subheader("📋 Zone Metadata CSV Format")
            st.markdown("Optional zone metadata file should be semicolon-separated:")

            example_zones = """No;Name;Population
1;Downtown;25000
2;Residential Area;18500
3;Industrial Zone;12000
4;Shopping District;8500"""

            st.code(example_zones, language="csv")

        st.markdown("""
        **Column Descriptions:**
        - **ACD**: Access distance
        - **ACT**: Access time
        - **EGD**: Egress distance  
        - **EGT**: Egress time
        - **JRD**: Journey distance
        - **JRT**: Journey time
        - **NTR**: Number of transfers
        - **RID**: Ride distance
        - **RIT**: Ride time
        - **SFQ**: Service frequency
        - **TWT**: Transfer wait time
        """)


if __name__ == "__main__":
    main()