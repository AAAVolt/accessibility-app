import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict
import re

st.set_page_config(page_title="Bizkaibus Schedule Analyzer", layout="wide")

st.title("Bizkaibus Schedule Analyzer")
st.markdown("Upload your Bizkaibus JSON schedule file to analyze routes and periods")

# File uploader
uploaded_file = st.file_uploader("Choose a JSON file", type="json")


def parse_date(date_str):
    """Parse date from DD/MM/YYYY format"""
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        return None


def extract_times_from_schedule(schedule_text):
    """Extract departure times from schedule text"""
    times = []
    if not schedule_text or schedule_text == "EZ DAGO ZERBITZURIK" or schedule_text == "NO HAY SERVICIO":
        return times

    # Pattern for times like 07:05, 22:35, etc.
    time_pattern = r'\b(\d{1,2}):(\d{2})\b'
    matches = re.findall(time_pattern, schedule_text)

    for hour, minute in matches:
        try:
            hour = int(hour)
            minute = int(minute)
            if 0 <= hour < 24 and 0 <= minute < 60:
                times.append(hour + minute / 60)  # Store as decimal hour
        except:
            pass

    # Pattern for ranges like "07:00etik 22:00era" or "De 7:00 a 22:00"
    range_pattern = r'(\d{1,2}):(\d{2}).*?(\d{1,2}):(\d{2})'
    range_matches = re.findall(range_pattern, schedule_text)

    for start_h, start_m, end_h, end_m in range_matches:
        try:
            start_hour = int(start_h) + int(start_m) / 60
            end_hour = int(end_h) + int(end_m) / 60

            # Check if it mentions "orduro" (every hour) or "cada hora"
            if "orduro" in schedule_text.lower() or "cada hora" in schedule_text.lower():
                current = int(start_h)
                while current <= int(end_h):
                    if current not in [t for t in times if int(t) == current]:
                        times.append(float(current))
                    current += 1
            # Check for "30 minuturik behin" or "cada 30 minutos"
            elif "30 minut" in schedule_text.lower() or "cada 30" in schedule_text.lower():
                current = float(start_h)
                while current <= end_hour:
                    times.append(current)
                    current += 0.5
        except:
            pass

    return sorted(set(times))


def get_day_type(date, schedule):
    """Determine day type (laborable/weekday, sabado/saturday, festivo/holiday)"""
    weekday = date.weekday()  # 0=Monday, 6=Sunday

    # Check schedule text for day type indicators
    schedule_ida = schedule.get("ORDUTEGIA_JOAN_CAS-HORARIO_IDA_CAS", "")

    if weekday == 6:  # Sunday
        return "festivo"
    elif weekday == 5:  # Saturday
        return "sabado"
    else:
        return "laborable"


def extract_schedule_data(json_data):
    """Extract schedule information from JSON"""
    lines_data = {}

    if "LINEAK-LINEAS" not in json_data:
        return lines_data

    lineas = json_data["LINEAK-LINEAS"].get("LINEA-LINEA", [])

    for linea in lineas:
        code = linea.get("KODEA-CODIGO", "")
        description = linea.get("DESKRIPZIOA-DESCRIPCION", "")
        schedules = linea.get("ORDUTEGIA-HORARIO", [])

        periods = []
        for schedule in schedules:
            period_name = schedule.get("DENBORALDI-TEMPORADA", "")
            start_date = parse_date(schedule.get("NOIZTIK-PERIODO_DESDE", ""))
            end_date = parse_date(schedule.get("NOIZ_ARTE-PERIODO_HASTA", ""))

            # Extract times for both directions
            ida_text = schedule.get("ORDUTEGIA_JOAN_CAS-HORARIO_IDA_CAS", "")
            vuelta_text = schedule.get("ORDUTEGIA_ETORRI_CAS-HORARIO_VUELTA_CAS", "")

            if start_date and end_date:
                periods.append({
                    "name": period_name,
                    "start": start_date,
                    "end": end_date,
                    "duration": (end_date - start_date).days + 1,
                    "schedule_ida": ida_text,
                    "schedule_vuelta": vuelta_text
                })

        if periods:
            lines_data[code] = {
                "description": description,
                "periods": periods
            }

    return lines_data


def get_active_lines_for_datetime(lines_data, target_date, target_hour):
    """Get all active lines and their frequencies for a specific date and time"""
    day_type = get_day_type(target_date, {})
    active_lines = []

    for line_code, line_data in lines_data.items():
        for period in line_data["periods"]:
            if period["start"] <= target_date <= period["end"]:
                schedule_text = period["schedule_ida"]

                # Normalize encoding issues
                schedule_text = schedule_text.replace("SÃ¡bados", "Sábados")
                schedule_text = schedule_text.replace("sÃ¡bado", "sábado")

                section = ""

                # Define patterns that include each day type
                if day_type == "sabado":
                    patterns_to_check = [
                        "Sábados y festivos:",
                        "Sábados y Festivos:",
                        "Sábados y domingos:",
                        "Larunbat eta jaiegunetan:",
                        "Laborables y sábados:",
                        "Laborables y sábado:",
                        "De lunes a sábado:",
                        "Lunes a sábado:",
                        "Astelehenetik larunbatera:",
                        "Sábados:",
                        "Larunbatetan:",
                    ]

                elif day_type == "festivo":
                    patterns_to_check = [
                        "Sábados y festivos:",
                        "Sábados y Festivos:",
                        "Domingos y festivos:",
                        "Larunbat eta jaiegunetan:",
                        "Festivos:",
                        "Jaiegunetan:",
                    ]

                elif day_type == "laborable":
                    patterns_to_check = [
                        "De lunes a sábado:",
                        "Lunes a sábado:",
                        "Laborables y sábados:",
                        "Laborables y sábado:",
                        "Astelehenetik larunbatera:",
                        "De lunes a viernes:",
                        "Lunes a viernes:",
                        "Laborables:",
                        "Lanegunetan:",
                        "Astelehenetik ostiralera:",
                    ]

                # Find the first matching pattern
                found_pattern = None
                for pattern in patterns_to_check:
                    if pattern in schedule_text:
                        found_pattern = pattern
                        break

                if found_pattern:
                    # Extract section starting from the found pattern
                    section = schedule_text.split(found_pattern)[1]

                    # Define all possible end markers (other day type patterns)
                    end_markers = [
                        "Laborables:",
                        "Lanegunetan:",
                        "Sábados:",
                        "Larunbatetan:",
                        "Festivos:",
                        "Jaiegunetan:",
                        "Sábados y festivos:",
                        "Sábados y Festivos:",
                        "Sábados y domingos:",
                        "Domingos y festivos:",
                        "Larunbat eta jaiegunetan:",
                        "Laborables y sábados:",
                        "Laborables y sábado:",
                        "De lunes a sábado:",
                        "Lunes a sábado:",
                        "De lunes a viernes:",
                        "Lunes a viernes:",
                        "Astelehenetik larunbatera:",
                        "Astelehenetik ostiralera:",
                    ]

                    # Find the earliest end marker to stop at
                    earliest_pos = len(section)
                    for marker in end_markers:
                        if marker != found_pattern and marker in section:
                            pos = section.find(marker)
                            if pos < earliest_pos:
                                earliest_pos = pos

                    section = section[:earliest_pos]

                times = extract_times_from_schedule(section)
                buses_at_hour = sum(1 for t in times if int(t) == target_hour)

                if buses_at_hour > 0:
                    active_lines.append({
                        "line_code": line_code,
                        "description": line_data["description"],
                        "period": period["name"],
                        "buses_at_hour": buses_at_hour,
                        "all_times": times
                    })
                break

    return active_lines

def create_hourly_analysis(lines_data, target_date):
    """Create hourly analysis for entire day"""
    hourly_counts = defaultdict(lambda: {"total_buses": 0, "active_lines": 0, "lines": []})

    for hour in range(24):
        active = get_active_lines_for_datetime(lines_data, target_date, hour)
        hourly_counts[hour]["total_buses"] = sum(line["buses_at_hour"] for line in active)
        hourly_counts[hour]["active_lines"] = len(active)
        hourly_counts[hour]["lines"] = active

    return hourly_counts


def create_line_calendar(line_code, line_data):
    """Create a calendar visualization for a specific line"""
    periods = line_data["periods"]

    period_types = list(set(p["name"] for p in periods))
    colors = px.colors.qualitative.Set3[:len(period_types)]
    color_map = dict(zip(period_types, colors))

    timeline_data = []
    for period in periods:
        timeline_data.append({
            "Period": period["name"],
            "Start": period["start"],
            "End": period["end"],
            "Duration": period["duration"]
        })

    df = pd.DataFrame(timeline_data)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Period",
        color="Period",
        color_discrete_map=color_map,
        title=f"Schedule Calendar: {line_code} - {line_data['description']}",
        labels={"Period": "Schedule Type"}
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=max(400, len(period_types) * 50),
        xaxis_title="Date",
        showlegend=False
    )

    return fig


def create_overall_analysis(lines_data):
    """Create overall analysis visualizations"""

    period_type_counts = defaultdict(int)
    period_type_days = defaultdict(int)

    for line_code, line_data in lines_data.items():
        for period in line_data["periods"]:
            period_type_counts[period["name"]] += 1
            period_type_days[period["name"]] += period["duration"]

    period_lines = defaultdict(set)
    for line_code, line_data in lines_data.items():
        for period in line_data["periods"]:
            period_lines[period["name"]].add(line_code)

    return period_type_counts, period_type_days, period_lines


if uploaded_file is not None:
    try:
        json_data = json.load(uploaded_file)
        lines_data = extract_schedule_data(json_data)

        if not lines_data:
            st.error("No valid schedule data found in the JSON file")
        else:
            st.success(f"Successfully loaded {len(lines_data)} bus lines")

            tab1, tab2, tab3 = st.tabs(
                ["Individual Line Calendars", "Overall Analysis", "Operational Analysis by Date/Time"])

            with tab1:
                st.header("Individual Line Schedules")

                line_codes = sorted(lines_data.keys())
                selected_line = st.selectbox("Select a bus line:", line_codes)

                if selected_line:
                    line_data = lines_data[selected_line]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Line Code", selected_line)
                    with col2:
                        st.metric("Total Periods", len(line_data["periods"]))

                    st.subheader(f"Route: {line_data['description']}")

                    fig = create_line_calendar(selected_line, line_data)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Period Details")
                    periods_df = pd.DataFrame([
                        {
                            "Period Type": p["name"],
                            "Start Date": p["start"].strftime("%d/%m/%Y"),
                            "End Date": p["end"].strftime("%d/%m/%Y"),
                            "Duration (days)": p["duration"]
                        }
                        for p in line_data["periods"]
                    ])
                    st.dataframe(periods_df, use_container_width=True)

            with tab2:
                st.header("Overall Schedule Analysis")

                period_counts, period_days, period_lines = create_overall_analysis(lines_data)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Lines", len(lines_data))
                with col2:
                    st.metric("Unique Period Types", len(period_counts))
                with col3:
                    total_periods = sum(len(ld["periods"]) for ld in lines_data.values())
                    st.metric("Total Schedule Periods", total_periods)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Period Type Frequency")
                    freq_df = pd.DataFrame([
                        {"Period Type": k, "Count": v}
                        for k, v in sorted(period_counts.items(), key=lambda x: x[1], reverse=True)
                    ])
                    fig1 = px.bar(
                        freq_df,
                        x="Count",
                        y="Period Type",
                        orientation="h",
                        title="Number of Period Occurrences"
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.subheader("Total Days by Period Type")
                    days_df = pd.DataFrame([
                        {"Period Type": k, "Total Days": v}
                        for k, v in sorted(period_days.items(), key=lambda x: x[1], reverse=True)
                    ])
                    fig2 = px.bar(
                        days_df,
                        x="Total Days",
                        y="Period Type",
                        orientation="h",
                        title="Total Days Covered by Each Period Type"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Lines Affected by Each Period Type")
                lines_per_period_df = pd.DataFrame([
                    {
                        "Period Type": k,
                        "Number of Lines": len(v),
                        "Lines": ", ".join(sorted(v))
                    }
                    for k, v in sorted(period_lines.items(), key=lambda x: len(x[1]), reverse=True)
                ])
                st.dataframe(lines_per_period_df, use_container_width=True)

            with tab3:
                st.header("Operational Analysis by Date and Time")
                st.markdown("Analyze how many buses are circulating at any specific moment")

                col1, col2 = st.columns(2)
                with col1:
                    analysis_date = st.date_input(
                        "Select date",
                        value=datetime(2025, 1, 15),
                        min_value=datetime(2025, 1, 1),
                        max_value=datetime(2025, 12, 31)
                    )

                with col2:
                    analysis_hour = st.slider("Select hour", 0, 23, 16, format="%d:00")

                target_datetime = datetime.combine(analysis_date, datetime.min.time())

                # Get active lines at specific time
                st.subheader(f"Active Lines at {analysis_hour}:00 on {analysis_date.strftime('%A, %B %d, %Y')}")

                active_lines = get_active_lines_for_datetime(lines_data, target_datetime, analysis_hour)

                if active_lines:
                    total_buses = sum(line["buses_at_hour"] for line in active_lines)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Buses Circulating", total_buses)
                    with col2:
                        st.metric("Active Lines", len(active_lines))
                    with col3:
                        avg_freq = total_buses / len(active_lines) if active_lines else 0
                        st.metric("Avg Buses per Line", f"{avg_freq:.1f}")

                    # Show active lines table
                    active_df = pd.DataFrame([
                        {
                            "Line": line["line_code"],
                            "Route": line["description"],
                            "Period": line["period"],
                            "Buses at this hour": line["buses_at_hour"]
                        }
                        for line in sorted(active_lines, key=lambda x: x["buses_at_hour"], reverse=True)
                    ])
                    st.dataframe(active_df, use_container_width=True)
                else:
                    st.info("No buses circulating at this time")

                # Hourly analysis for entire day
                st.subheader(f"Hourly Analysis for {analysis_date.strftime('%A, %B %d, %Y')}")

                hourly_data = create_hourly_analysis(lines_data, target_datetime)

                hourly_df = pd.DataFrame([
                    {
                        "Hour": f"{h:02d}:00",
                        "Total Buses": data["total_buses"],
                        "Active Lines": data["active_lines"]
                    }
                    for h, data in sorted(hourly_data.items())
                ])

                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=hourly_df["Hour"],
                    y=hourly_df["Total Buses"],
                    name="Total Buses",
                    marker_color='lightblue'
                ))
                fig_hourly.add_trace(go.Scatter(
                    x=hourly_df["Hour"],
                    y=hourly_df["Active Lines"],
                    name="Active Lines",
                    yaxis="y2",
                    marker_color='red',
                    line=dict(width=3)
                ))

                fig_hourly.update_layout(
                    title="Bus Circulation Throughout the Day",
                    xaxis_title="Hour",
                    yaxis_title="Total Buses Circulating",
                    yaxis2=dict(
                        title="Number of Active Lines",
                        overlaying="y",
                        side="right"
                    ),
                    hovermode="x unified",
                    height=500
                )

                st.plotly_chart(fig_hourly, use_container_width=True)

                # Peak hours analysis
                st.subheader("Peak Hours Analysis")
                max_buses = max(data["total_buses"] for data in hourly_data.values())
                peak_hours = [h for h, data in hourly_data.items() if data["total_buses"] == max_buses]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Peak Bus Circulation", max_buses)
                with col2:
                    peak_hours_str = ", ".join([f"{h:02d}:00" for h in peak_hours])
                    st.metric("Peak Hours", peak_hours_str)

    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

else:
    st.info("Please upload a JSON file to begin analysis")

    st.markdown("""
    ### How to use this app:

    1. **Upload** your Bizkaibus schedule JSON file using the file uploader above
    2. **View Individual Line Calendars** to see the schedule periods for each bus line
    3. **Analyze Overall Patterns** to understand how different period types are distributed
    4. **Operational Analysis** to see how many buses are running at any specific date and time

    ### Features:
    - See which lines are active at any specific moment
    - Understand bus frequency throughout the day
    - Identify peak and off-peak hours
    - Compare operational intensity across different days
    """)