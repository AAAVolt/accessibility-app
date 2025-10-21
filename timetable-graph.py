import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
from io import BytesIO
import openpyxl


class TimetableGraphGenerator:
    def __init__(self):
        self.df = None
        self.stations = {}

    def load_excel_file(self, file_path_or_buffer):
        """Load Excel file and process the data"""
        try:
            # Read the Excel file
            self.df = pd.read_excel(file_path_or_buffer, sheet_name='Vehicle journeys')

            # Clean column names (remove potential extra characters)
            self.df.columns = [col.split(':')[-1].strip() if ':' in str(col) else str(col).strip()
                               for col in self.df.columns]

            # Handle datetime conversion more carefully
            # The Excel dates might be in different formats
            for col in ['Dep', 'Arr']:
                if col in self.df.columns:
                    # Try different approaches to convert datetime
                    if self.df[col].dtype == 'object':
                        # If it's already a string, try to parse it
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif 'timedelta' in str(self.df[col].dtype):
                        # If it's a timedelta, convert to datetime by adding to a base date
                        base_date = pd.Timestamp('1899-12-30')  # Excel's base date
                        self.df[col] = base_date + self.df[col]
                    else:
                        # Try direct conversion
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

            # Extract time components for plotting
            self.df['dep_time'] = self.df['Dep'].dt.time
            self.df['arr_time'] = self.df['Arr'].dt.time

            # Convert times to minutes from midnight for easier plotting
            self.df['dep_minutes'] = (self.df['Dep'].dt.hour * 60 +
                                      self.df['Dep'].dt.minute +
                                      self.df['Dep'].dt.second / 60)
            self.df['arr_minutes'] = (self.df['Arr'].dt.hour * 60 +
                                      self.df['Arr'].dt.minute +
                                      self.df['Arr'].dt.second / 60)

            # Create station mapping from identifiers
            self._create_station_mapping()

            # Debug: Print column names and first few rows
            st.write("Debug - Column names:", list(self.df.columns))
            st.write("Debug - First row:", self.df.iloc[0].to_dict())

            return True

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _create_station_mapping(self):
        """Create a mapping of station identifiers to positions"""
        try:
            # Extract unique station identifiers with safer column access
            from_col = 'FromTProfItemIdentifier'
            to_col = 'ToTProfItemIdentifier'

            # Check if columns exist
            if from_col not in self.df.columns or to_col not in self.df.columns:
                st.warning(f"Station identifier columns not found. Available columns: {list(self.df.columns)}")
                # Try alternative column names
                possible_from_cols = [col for col in self.df.columns if
                                      'from' in col.lower() or 'origin' in col.lower()]
                possible_to_cols = [col for col in self.df.columns if 'to' in col.lower() or 'dest' in col.lower()]

                if possible_from_cols and possible_to_cols:
                    from_col = possible_from_cols[0]
                    to_col = possible_to_cols[0]
                    st.info(f"Using alternative columns: {from_col}, {to_col}")
                else:
                    # Create simple numeric station positions based on route
                    self.df['from_station_pos'] = 0
                    self.df['to_station_pos'] = 1
                    self.stations = {"Origin": 0, "Destination": 1}
                    return

            from_stations = self.df[from_col].dropna().unique()
            to_stations = self.df[to_col].dropna().unique()

            all_stations = list(set(list(from_stations) + list(to_stations)))
            all_stations.sort()

            # Create position mapping
            self.stations = {station: i for i, station in enumerate(all_stations)}

            # Add station positions to dataframe
            self.df['from_station_pos'] = self.df[from_col].map(self.stations)
            self.df['to_station_pos'] = self.df[to_col].map(self.stations)

            # Fill NaN values with 0 (fallback)
            self.df['from_station_pos'] = self.df['from_station_pos'].fillna(0)
            self.df['to_station_pos'] = self.df['to_station_pos'].fillna(1)

        except Exception as e:
            st.warning(f"Error creating station mapping: {str(e)}")
            # Fallback: create simple positions
            self.df['from_station_pos'] = 0
            self.df['to_station_pos'] = 1
            self.stations = {"Origin": 0, "Destination": 1}

    def get_available_lines(self):
        """Get list of available line names"""
        if self.df is None:
            return []

        # Try to find line name column
        line_col = None
        for col in self.df.columns:
            if 'line' in col.lower() and 'name' in col.lower():
                line_col = col
                break

        if line_col is None:
            # Try just 'line'
            for col in self.df.columns:
                if 'line' in col.lower():
                    line_col = col
                    break

        if line_col is None:
            return []

        return sorted(self.df[line_col].dropna().unique())

    def generate_timetable_graph(self, selected_lines=None, direction_filter=None):
        """Generate the timetable graph (string diagram)"""
        if self.df is None:
            st.error("No data loaded. Please upload an Excel file first.")
            return None

        # Filter data
        filtered_df = self.df.copy()

        if selected_lines:
            line_col = 'LineName'
            if line_col not in filtered_df.columns:
                # Try to find alternative line column
                possible_line_cols = [col for col in filtered_df.columns if 'line' in col.lower()]
                if possible_line_cols:
                    line_col = possible_line_cols[0]
                    st.info(f"Using line column: {line_col}")
                else:
                    st.warning("No line name column found")
                    return None

            filtered_df = filtered_df[filtered_df[line_col].isin(selected_lines)]

        if direction_filter is not None:
            dir_col = 'DirectionCode'
            if dir_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[dir_col] == direction_filter]

        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
            return None

        # Ensure we have the required columns
        required_cols = ['from_station_pos', 'to_station_pos', 'dep_minutes', 'arr_minutes']
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Get line column name
        line_col = 'LineName' if 'LineName' in filtered_df.columns else \
        [col for col in filtered_df.columns if 'line' in col.lower()][0]

        # Color mapping for different lines
        unique_lines = filtered_df[line_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_lines)))
        color_map = dict(zip(unique_lines, colors))

        # Plot each journey as a line
        for idx, row in filtered_df.iterrows():
            line_name = row[line_col]
            color = color_map[line_name]

            # Plot the journey line
            if pd.notna(row['from_station_pos']) and pd.notna(row['to_station_pos']):
                x_coords = [row['dep_minutes'], row['arr_minutes']]
                y_coords = [row['from_station_pos'], row['to_station_pos']]

                ax.plot(x_coords, y_coords, color=color, alpha=0.7, linewidth=2,
                        label=line_name if line_name not in ax.get_legend_handles_labels()[1] else "")

        # Customize the plot
        ax.set_xlabel('Time of Day', fontsize=12)
        ax.set_ylabel('Station Position', fontsize=12)
        ax.set_title('Timetable Graph (String Diagram)', fontsize=14, fontweight='bold')

        # Format x-axis to show time
        time_ticks = range(0, 24 * 60, 60)  # Every hour
        time_labels = [f"{h:02d}:00" for h in range(24)]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, rotation=45)

        # Set y-axis to show station names (simplified)
        if self.stations:
            station_names = list(self.stations.keys())
            station_positions = list(self.stations.values())

            # Limit to reasonable number of station labels
            if len(station_names) > 20:
                step = max(1, len(station_names) // 10)
                ax.set_yticks(station_positions[::step])
                ax.set_yticklabels([name.split()[-1][:15] for name in station_names[::step]], fontsize=8)
            else:
                ax.set_yticks(station_positions)
                ax.set_yticklabels([name.split()[-1][:15] for name in station_names], fontsize=8)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add legend
        if len(unique_lines) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def get_data_summary(self):
        """Get summary statistics of the loaded data"""
        if self.df is None:
            return None

        # Find line column dynamically
        line_col = None
        for col in self.df.columns:
            if 'line' in col.lower():
                line_col = col
                break

        # Find direction column dynamically
        dir_col = None
        for col in self.df.columns:
            if 'direction' in col.lower():
                dir_col = col
                break

        summary = {
            'total_journeys': len(self.df),
            'unique_lines': len(self.df[line_col].unique()) if line_col else 0,
            'unique_stations': len(self.stations),
            'time_range': f"{self.df['dep_time'].min()} - {self.df['arr_time'].max()}" if 'dep_time' in self.df.columns else "N/A",
            'directions': sorted(self.df[dir_col].unique()) if dir_col else []
        }
        return summary


def main():
    st.set_page_config(page_title="Timetable Graph Generator", layout="wide")

    st.title("🚇 Vehicle Journey Timetable Graph Generator")
    st.markdown("Generate string diagrams (space-time diagrams) from vehicle journey data")

    # Initialize the generator
    if 'generator' not in st.session_state:
        st.session_state.generator = TimetableGraphGenerator()

    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel file with vehicle journeys",
            type=['xlsx', 'xls']
        )

        if uploaded_file is not None:
            if st.session_state.generator.load_excel_file(uploaded_file):
                st.success("✅ File loaded successfully!")

                # Show data summary
                summary = st.session_state.generator.get_data_summary()
                if summary:
                    st.subheader("📊 Data Summary")
                    st.write(f"**Total Journeys:** {summary['total_journeys']}")
                    st.write(f"**Unique Lines:** {summary['unique_lines']}")
                    st.write(f"**Stations:** {summary['unique_stations']}")
                    st.write(f"**Time Range:** {summary['time_range']}")
                    st.write(f"**Directions:** {summary['directions']}")

        # Controls section
        if st.session_state.generator.df is not None:
            st.header("🎛️ Graph Controls")

            # Line selector
            available_lines = st.session_state.generator.get_available_lines()
            selected_lines = st.multiselect(
                "Select Lines to Display",
                options=available_lines,
                default=available_lines[:3] if len(available_lines) > 3 else available_lines
            )

            # Direction filter
            dir_col = None
            for col in st.session_state.generator.df.columns:
                if 'direction' in col.lower():
                    dir_col = col
                    break

            if dir_col:
                directions = sorted(st.session_state.generator.df[dir_col].unique())
                direction_filter = st.selectbox(
                    "Filter by Direction",
                    options=[None] + list(directions),
                    format_func=lambda x: "All Directions" if x is None else f"Direction {x}"
                )
            else:
                direction_filter = None

            # Generate graph button
            if st.button("🔄 Generate Graph", type="primary"):
                st.session_state.generate_graph = True

    # Main content area
    if st.session_state.generator.df is not None:
        # Show sample data
        with st.expander("📋 View Sample Data"):
            st.dataframe(st.session_state.generator.df.head(10))

        # Generate and display graph
        if 'generate_graph' in st.session_state and st.session_state.generate_graph:
            with st.spinner("Generating timetable graph..."):
                fig = st.session_state.generator.generate_timetable_graph(
                    selected_lines=selected_lines if 'selected_lines' in locals() and selected_lines else None,
                    direction_filter=direction_filter if 'direction_filter' in locals() else None
                )

                if fig is not None:
                    st.pyplot(fig)

                    # Download button for the plot
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)

                    st.download_button(
                        label="📥 Download Graph as PNG",
                        data=buf.getvalue(),
                        file_name="timetable_graph.png",
                        mime="image/png"
                    )
    else:
        # Welcome message
        st.info("👆 Please upload an Excel file with vehicle journey data to get started.")

        # Instructions
        st.markdown("""
        ### 📖 Instructions

        1. **Upload your Excel file** using the sidebar uploader
        2. **Select the lines** you want to visualize
        3. **Choose direction filter** (optional)
        4. **Generate the graph** to see your timetable diagram

        ### 📋 Required Data Format

        Your Excel file should contain a sheet named "Vehicle journeys" with these columns:
        - `Dep`: Departure time
        - `Arr`: Arrival time  
        - `LineName`: Name/ID of the transit line
        - `DirectionCode`: Direction identifier (0, 1, etc.)
        - `FromTProfItemIdentifier`: Origin station identifier
        - `ToTProfItemIdentifier`: Destination station identifier

        ### 🎯 What is a Timetable Graph?

        A timetable graph (also known as a string diagram or space-time diagram) is a visual representation where:
        - **X-axis**: Time of day
        - **Y-axis**: Station positions along the route
        - **Lines**: Each journey drawn as a line connecting departure and arrival points

        This visualization is commonly used in railway operations to analyze schedules, identify conflicts, and optimize timetables.
        """)


if __name__ == "__main__":
    main()