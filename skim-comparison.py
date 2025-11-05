"""
Journey Time Comparison Streamlit App

A web-based tool for comparing journey times between two transport scenarios.
Focuses on clear tabular output without graphs, following Python best practices.

Author: Claude
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import io


class JourneyTimeComparator:
    """
    A class for comparing journey times between two transport scenarios.

    This tool analyzes skim matrix data and provides clear tabular comparisons
    focusing specifically on journey time metrics.
    """

    def __init__(self):
        """Initialize the comparator."""
        self.required_columns = [
            'OrigZoneNo', 'DestZoneNo', 'ACD', 'ACT', 'EGD', 'EGT',
            'JRD', 'JRT', 'NTR', 'RID', 'RIT', 'SFQ', 'TWT'
        ]

    def load_skim_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load and validate skim matrix data from uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            DataFrame with skim data, or None if loading fails
        """
        try:
            df = pd.read_csv(uploaded_file)

            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return None

            # Validate data types
            numeric_cols = [col for col in self.required_columns if col not in ['OrigZoneNo', 'DestZoneNo']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ensure zone numbers are integers
            df['OrigZoneNo'] = pd.to_numeric(df['OrigZoneNo'], errors='coerce').astype('Int64')
            df['DestZoneNo'] = pd.to_numeric(df['DestZoneNo'], errors='coerce').astype('Int64')

            # Remove rows with invalid data
            initial_rows = len(df)
            df = df.dropna(subset=['OrigZoneNo', 'DestZoneNo', 'JRT'])
            if len(df) < initial_rows:
                st.warning(f"⚠️ Removed {initial_rows - len(df)} rows with invalid data")

            st.success(f"✅ Loaded skim data with {len(df)} valid rows")
            return df

        except Exception as e:
            st.error(f"❌ Error loading skim data: {str(e)}")
            return None

    def load_population_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load population data from uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            DataFrame with population data, or None if loading fails
        """
        try:
            # Try different separators
            content = uploaded_file.getvalue().decode('utf-8')

            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) >= 3:
                        break
                except:
                    continue
            else:
                st.error("❌ Could not parse population file with any common separator")
                return None

            # Clean column names
            df.columns = df.columns.str.strip()

            # Map column names flexibly
            zone_col = None
            name_col = None
            pop_col = None

            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['no', 'zone_no', 'zoneno', 'zone']:
                    zone_col = col
                elif col_lower in ['name', 'zone_name', 'zonename']:
                    name_col = col
                elif col_lower in ['population', 'pop', 'inhabitants']:
                    pop_col = col

            if not all([zone_col, name_col, pop_col]):
                st.error("❌ Could not identify required columns (Zone No, Name, Population) in population file")
                return None

            # Create standardized dataframe
            result_df = df[[zone_col, name_col, pop_col]].copy()
            result_df.columns = ['ZoneNo', 'ZoneName', 'Population']

            # Convert to appropriate types
            result_df['ZoneNo'] = pd.to_numeric(result_df['ZoneNo'], errors='coerce').astype('Int64')
            result_df['Population'] = pd.to_numeric(result_df['Population'], errors='coerce')

            # Remove invalid rows
            result_df = result_df.dropna()

            st.success(f"✅ Loaded population data for {len(result_df)} zones")
            return result_df

        except Exception as e:
            st.error(f"❌ Error loading population data: {str(e)}")
            return None

    def compare_journey_times(
            self,
            scenario1_df: pd.DataFrame,
            scenario2_df: pd.DataFrame,
            destination_zone: int,
            population_df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        Compare journey times between two scenarios for a specific destination.

        Args:
            scenario1_df: Skim data for scenario 1 (baseline)
            scenario2_df: Skim data for scenario 2 (alternative)
            destination_zone: Zone number for the destination
            population_df: Optional population data for weighting

        Returns:
            DataFrame with journey time comparison results
        """
        try:
            # Filter data for the destination zone
            df1_dest = scenario1_df[scenario1_df['DestZoneNo'] == destination_zone].copy()
            df2_dest = scenario2_df[scenario2_df['DestZoneNo'] == destination_zone].copy()

            if df1_dest.empty or df2_dest.empty:
                st.warning(f"⚠️ No data found for destination zone {destination_zone}")
                return None

            # Merge scenarios on origin zone
            comparison = pd.merge(
                df1_dest,
                df2_dest,
                on='OrigZoneNo',
                suffixes=('_baseline', '_alternative')
            )

            if comparison.empty:
                st.warning("⚠️ No matching origin zones between scenarios")
                return None

            # Calculate journey time metrics
            comparison['JRT_difference'] = (
                    comparison['JRT_alternative'] - comparison['JRT_baseline']
            )
            comparison['JRT_percent_change'] = (
                    (comparison['JRT_alternative'] - comparison['JRT_baseline']) /
                    comparison['JRT_baseline'].replace(0, np.nan) * 100
            )

            # Add population data if available
            if population_df is not None:
                comparison = comparison.merge(
                    population_df[['ZoneNo', 'ZoneName', 'Population']],
                    left_on='OrigZoneNo',
                    right_on='ZoneNo',
                    how='left'
                )
                comparison = comparison.drop('ZoneNo', axis=1)
                comparison = comparison.rename(columns={
                    'ZoneName': 'OriginZoneName',
                    'Population': 'OriginPopulation'
                })

                # Calculate population-weighted impact
                comparison['OriginPopulation'] = comparison['OriginPopulation'].fillna(0)
                comparison['Population_impact'] = (
                        comparison['JRT_difference'] * comparison['OriginPopulation']
                )

                # Get destination zone name
                dest_info = population_df[population_df['ZoneNo'] == destination_zone]
                dest_name = dest_info.iloc[0]['ZoneName'] if not dest_info.empty else f"Zone {destination_zone}"
                comparison['DestinationZoneName'] = dest_name
            else:
                comparison['OriginZoneName'] = comparison['OrigZoneNo'].apply(lambda x: f"Zone {x}")
                comparison['OriginPopulation'] = 0
                comparison['Population_impact'] = 0
                comparison['DestinationZoneName'] = f"Zone {destination_zone}"

            # Sort by impact (largest absolute changes first)
            comparison = comparison.reindex(
                comparison['JRT_difference'].abs().sort_values(ascending=False).index
            )

            return comparison

        except Exception as e:
            st.error(f"❌ Error comparing journey times: {str(e)}")
            return None

    def format_comparison_table(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the comparison results into a clean table for display.

        Args:
            comparison_df: Raw comparison results

        Returns:
            Formatted DataFrame ready for display
        """
        # Select and order columns for display
        display_columns = [
            'OrigZoneNo',
            'OriginZoneName',
            'OriginPopulation',
            'DestinationZoneName',
            'JRT_baseline',
            'JRT_alternative',
            'JRT_difference',
            'JRT_percent_change',
            'Population_impact'
        ]

        # Only include columns that exist
        available_columns = [col for col in display_columns if col in comparison_df.columns]
        result_df = comparison_df[available_columns].copy()

        # Round numeric values appropriately
        numeric_formatting = {
            'JRT_baseline': 2,
            'JRT_alternative': 2,
            'JRT_difference': 2,
            'JRT_percent_change': 2,
            'Population_impact': 0,
            'OriginPopulation': 0
        }

        for col, decimals in numeric_formatting.items():
            if col in result_df.columns:
                result_df[col] = result_df[col].round(decimals)

        # Rename columns for better readability
        column_names = {
            'OrigZoneNo': 'Origin Zone',
            'OriginZoneName': 'Origin Name',
            'OriginPopulation': 'Population',
            'DestinationZoneName': 'Destination',
            'JRT_baseline': 'Baseline Time (min)',
            'JRT_alternative': 'Alternative Time (min)',
            'JRT_difference': 'Time Difference (min)',
            'JRT_percent_change': 'Percent Change (%)',
            'Population_impact': 'Population Impact (person-min)'
        }

        result_df = result_df.rename(columns=column_names)

        return result_df

    def generate_summary_statistics(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the comparison.

        Args:
            comparison_df: Comparison results DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        try:
            # Basic statistics
            summary['total_zones'] = len(comparison_df)
            summary['zones_improved'] = len(comparison_df[comparison_df['JRT_difference'] < 0])
            summary['zones_worsened'] = len(comparison_df[comparison_df['JRT_difference'] > 0])
            summary['zones_unchanged'] = len(comparison_df[comparison_df['JRT_difference'] == 0])

            # Journey time statistics
            summary['avg_time_change'] = comparison_df['JRT_difference'].mean()
            summary['max_improvement'] = comparison_df['JRT_difference'].min()
            summary['max_degradation'] = comparison_df['JRT_difference'].max()
            summary['median_change'] = comparison_df['JRT_difference'].median()

            # Population-weighted statistics (if available)
            if 'Population_impact' in comparison_df.columns:
                total_pop = comparison_df['OriginPopulation'].sum()
                if total_pop > 0:
                    summary['total_population'] = int(total_pop)
                    summary['weighted_avg_change'] = (
                            comparison_df['Population_impact'].sum() / total_pop
                    )
                    summary['total_person_minutes_impact'] = comparison_df['Population_impact'].sum()

            return summary

        except Exception as e:
            st.error(f"❌ Error generating summary statistics: {str(e)}")
            return {}


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Journey Time Comparison Tool",
        page_icon="🚌",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚌 Journey Time Comparison Tool")
    st.markdown("Compare journey times between two transport scenarios with clear tabular analysis")

    # Initialize the comparator
    if 'comparator' not in st.session_state:
        st.session_state.comparator = JourneyTimeComparator()

    # Sidebar for file uploads
    st.sidebar.header("📁 Data Upload")

    # Scenario file uploads
    scenario1_file = st.sidebar.file_uploader(
        "Scenario 1 (Baseline) CSV",
        type=['csv'],
        help="Upload the skim matrix CSV for your baseline scenario"
    )

    scenario2_file = st.sidebar.file_uploader(
        "Scenario 2 (Alternative) CSV",
        type=['csv'],
        help="Upload the skim matrix CSV for your alternative scenario"
    )

    # Population file upload (optional)
    population_file = st.sidebar.file_uploader(
        "Population Data CSV (Optional)",
        type=['csv'],
        help="Upload population data for zones (semicolon-separated)"
    )

    # Main content area
    if scenario1_file is not None and scenario2_file is not None:

        # Load data
        with st.spinner("Loading scenario data..."):
            scenario1_data = st.session_state.comparator.load_skim_data(scenario1_file)
            scenario2_data = st.session_state.comparator.load_skim_data(scenario2_file)

        population_data = None
        if population_file is not None:
            with st.spinner("Loading population data..."):
                population_data = st.session_state.comparator.load_population_data(population_file)

        if scenario1_data is not None and scenario2_data is not None:

            # Get available destination zones
            dest_zones_s1 = set(scenario1_data['DestZoneNo'].unique())
            dest_zones_s2 = set(scenario2_data['DestZoneNo'].unique())
            common_dest_zones = sorted(dest_zones_s1.intersection(dest_zones_s2))

            if not common_dest_zones:
                st.error("❌ No common destination zones found between scenarios")
                return

            # Zone selection
            st.sidebar.header("🎯 Analysis Settings")
            selected_dest_zone = st.sidebar.selectbox(
                "Select Destination Zone",
                options=common_dest_zones,
                help="Choose the destination zone for journey time analysis"
            )

            # Filter options
            min_change_threshold = st.sidebar.slider(
                "Minimum Time Change (minutes)",
                min_value=0.00,
                max_value=30.0,
                value=0.00,
                step=0.01,
                help="Only show zones with changes above this threshold"
            )

            max_rows_display = st.sidebar.selectbox(
                "Maximum Rows to Display",
                options=[10, 20, 50, 100, "All"],
                index=2,
                help="Limit the number of rows shown in the table"
            )

            # Perform comparison
            if st.sidebar.button("🔍 Run Analysis", type="primary"):

                with st.spinner(f"Comparing journey times to zone {selected_dest_zone}..."):
                    comparison_results = st.session_state.comparator.compare_journey_times(
                        scenario1_data,
                        scenario2_data,
                        selected_dest_zone,
                        population_data
                    )

                if comparison_results is not None:

                    # Generate summary statistics
                    summary_stats = st.session_state.comparator.generate_summary_statistics(comparison_results)

                    # Format table
                    formatted_table = st.session_state.comparator.format_comparison_table(comparison_results)

                    # Apply filters
                    if min_change_threshold > 0:
                        mask = abs(comparison_results['JRT_difference']) >= min_change_threshold
                        formatted_table = formatted_table[mask]
                        comparison_results = comparison_results[mask]

                    # Limit rows if specified
                    if max_rows_display != "All":
                        formatted_table = formatted_table.head(max_rows_display)

                    # Display results
                    st.header("📊 Analysis Results")

                    # Summary statistics in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Zones", summary_stats.get('total_zones', 0))
                        st.metric("Zones Improved", summary_stats.get('zones_improved', 0))

                    with col2:
                        st.metric("Zones Worsened", summary_stats.get('zones_worsened', 0))
                        st.metric("Zones Unchanged", summary_stats.get('zones_unchanged', 0))

                    with col3:
                        avg_change = summary_stats.get('avg_time_change', 0)
                        st.metric("Average Change (min)", f"{avg_change:.2f}")
                        max_improvement = summary_stats.get('max_improvement', 0)
                        st.metric("Best Improvement (min)", f"{max_improvement:.2f}")

                    with col4:
                        max_degradation = summary_stats.get('max_degradation', 0)
                        st.metric("Worst Degradation (min)", f"{max_degradation:.2f}")
                        median_change = summary_stats.get('median_change', 0)
                        st.metric("Median Change (min)", f"{median_change:.2f}")

                    # Population statistics if available
                    if 'total_population' in summary_stats:
                        st.subheader("🏘️ Population-Weighted Impact")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Population", f"{summary_stats['total_population']:,}")

                        with col2:
                            weighted_avg = summary_stats.get('weighted_avg_change', 0)
                            st.metric("Weighted Avg Change (min)", f"{weighted_avg:.2f}")

                        with col3:
                            total_impact = summary_stats.get('total_person_minutes_impact', 0)
                            st.metric("Total Impact (person-min)", f"{total_impact:,.0f}")

                    # Data table
                    st.subheader("📋 Detailed Journey Time Comparison")

                    if len(formatted_table) > 0:
                        st.dataframe(
                            formatted_table,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Download button
                        csv_buffer = io.StringIO()
                        formatted_table.to_csv(csv_buffer, index=False)

                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"journey_time_comparison_dest_{selected_dest_zone}.csv",
                            mime="text/csv"
                        )

                    else:
                        st.info("ℹ️ No zones meet the specified filter criteria.")

            # Show data preview
            with st.expander("👀 Preview Loaded Data"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Scenario 1 (Sample)")
                    st.dataframe(scenario1_data.head(), use_container_width=True)

                with col2:
                    st.subheader("Scenario 2 (Sample)")
                    st.dataframe(scenario2_data.head(), use_container_width=True)

                if population_data is not None:
                    st.subheader("Population Data (Sample)")
                    st.dataframe(population_data.head(), use_container_width=True)

    else:
        # Show instructions when no files uploaded
        st.info("👆 Please upload both scenario CSV files in the sidebar to begin analysis")

        # Example data formats
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Skim Matrix CSV Format")
            st.markdown("Your skim matrix CSV files should have this structure:")

            example_data = """OrigZoneNo,DestZoneNo,ACD,ACT,EGD,EGT,JRD,JRT,NTR,RID,RIT,SFQ,TWT
1,1,0.000,0.00,0.000,0.00,0.000,0.00,0.000,0.000,0.00,0.000,0.00
1,2,0.037,0.37,0.761,7.62,53.281,94.08,1.000,52.483,86.10,32.000,3.68
1,3,0.037,0.37,0.470,4.70,19.717,36.75,0.000,19.210,31.68,22.000,0.00"""

            st.code(example_data, language="csv")

        with col2:
            st.subheader("📋 Population CSV Format")
            st.markdown("Optional population file should be semicolon-separated:")

            example_zones = """No;Name;Population
1;Abadiño-Zelaieta;1559
2;Abanto;25
3;Agarre;42
4;Agirre;22"""

            st.code(example_zones, language="csv")

        st.markdown("""
        ### 📖 Column Descriptions:

        **Skim Matrix Columns:**
        - **ACD**: Access distance
        - **ACT**: Access time
        - **EGD**: Egress distance  
        - **EGT**: Egress time
        - **JRD**: Journey distance
        - **JRT**: Journey time ⭐ (main focus)
        - **NTR**: Number of transfers
        - **RID**: Ride distance
        - **RIT**: Ride time
        - **SFQ**: Service frequency
        - **TWT**: Transfer wait time

        **Population File Columns:**
        - **No**: Zone number
        - **Name**: Zone name
        - **Population**: Population count
        """)


if __name__ == "__main__":
    main()