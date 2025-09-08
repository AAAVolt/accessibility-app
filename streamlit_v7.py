# Calculated skims:
# ACD	Access distance
# ACT	Access time
# EGD	Egress distance
# EGT	Egress time
# JRD	Journey distance
# JRT	Journey time
# NTR	Number of transfers
# RID	Ride distance
# RIT	Ride time
# SFQ	Service frequency
# TWT	Transfer wait time

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Enhanced Public Transport Accessibility Analysis", layout="wide")
st.title("🚌 Enhanced Public Transport Accessibility Analysis")

# Upload files
uploaded_skim_file = st.file_uploader("Upload your skim matrix CSV", type=["csv"], key="skim")
uploaded_zones_file = st.file_uploader("Upload zone-destination mapping CSV", type=["csv"], key="zones")
pop_file = st.file_uploader("Upload Population XLSX (optional)", type=["xlsx"], key="pop")

# ====== NEW: metric catalog ======
ALL_METRICS = ["JRT", "NTR", "RIT", "ACT", "EGT", "ACD", "EGD", "SFQ", "JRD", "RID", "TWT"]  # display/order helpers
TIME_METRICS = {"JRT", "RIT", "ACT", "EGT", "TWT"}  # minutes  # minutes
DIST_METRICS = {"JRD", "ACD", "EGD", "RID"}  # distance (km or m)                               # distance (km or m)
COUNT_METRICS = {"NTR"}  # integer                              # integer
FREQ_METRICS = {"SFQ"}  # frequency or headway                               # frequency or headway
DEFAULT_COLS = ["OrigZoneNo", "DestZoneNo", "ACD", "ACT", "EGD", "EGT", "JRD", "JRT", "NTR", "RID", "RIT", "SFQ",
                "TWT"] + ["ACT", "EGT", "", "JRD", "JRT", "NTR", "RIT", "SFQ"]

# Default metric descriptions
metric_descriptions = {
    "JRT": "Journey Time (total door-to-door time, min)",
    "NTR": "Number of Transfers",
    "RIT": "Ride Time (time on vehicle, min)",
    "ACT": "Access Time (walk/approach time, min)",
    "EGT": "Egress Time (walk/alight time, min)",
    "ACD": "Access Distance",
    "EGD": "Egress Distance",
    "RID": "Ride Distance",
    "SFQ": "Service Frequency / Headway",
    "JRD": "Journey Distance (km)",
    "TWT": "Transfer Wait Time (min)"
}

if uploaded_skim_file:
    st.sidebar.header("Skim Matrix Configuration")

    skim_type = st.sidebar.selectbox(
        "What type of destinations does this skim matrix contain?",
        [
            "Hospitales",
            "Ozakidetza",
            "Residencias",
            "Bachilleres",
            "Hacienda",
            "Catastro",
            "Atencion",
            "Servicios Sociales",
            "Agricultura",
            "Comarca (within normal zones)",
            "Single Zone (like Bilbao, Uni Bilbao)",
            "Custom Range"
        ]
    )

    predefined_ranges = {
        "Hospitales": (900, 904),
        "Ozakidetza": (500, 768),
        "Residencias": (600, 756),
        "Bachilleres": (800, 848),
        "Hacienda": (500, 511),
        "Catastro": (512, 523),
        "Atencion": (524, 575),
        "Servicios Sociales": (584, 591),
        "Agricultura": (576, 583),
        "Comarca (within normal zones)": (1, 453),
        "Single Zone (like Bilbao, Uni Bilbao)": (1, 453),
        "Custom Range": (500, 600)
    }
    default_start, default_end = predefined_ranges[skim_type]

    st.sidebar.subheader("Destination Zone Configuration")

    if skim_type == "Single Zone (like Bilbao, Uni Bilbao)":
        st.sidebar.info("💡 For single zones, set start and end to the same zone number")
        dest_start = st.sidebar.number_input(
            "Single Destination Zone ID",
            value=1,
            help="The specific zone ID for destinations like Bilbao, University, etc."
        )
        dest_end = dest_start
        st.sidebar.write(f"Will analyze accessibility to zone: **{dest_start}**")
    elif skim_type == "Comarca (within normal zones)":
        st.sidebar.info("💡 Comarca zones are within the normal zone range (1-453)")
        dest_start = st.sidebar.number_input(
            "Comarca Zone Start", value=default_start, min_value=1, max_value=453,
            help="Starting zone ID for comarca destinations"
        )
        dest_end = st.sidebar.number_input(
            "Comarca Zone End", value=default_end, min_value=dest_start, max_value=453,
            help="Ending zone ID for comarca destinations"
        )
        st.sidebar.write(f"Will analyze accessibility to comarca zones: **{dest_start}-{dest_end}**")
    else:
        dest_start = st.sidebar.number_input(
            "Destination Zone Start", value=default_start,
            help="Starting zone ID for destinations in this skim matrix"
        )
        dest_end = st.sidebar.number_input(
            "Destination Zone End", value=default_end,
            help="Ending zone ID for destinations in this skim matrix"
        )
        st.sidebar.info(f"Will analyze accessibility to destinations in zones **{dest_start}-{dest_end}**")

    # ===== NEW: Interpretation for SFQ =====
    st.sidebar.subheader("Service Frequency Interpretation")
    sfq_mode = st.sidebar.selectbox(
        "How should SFQ be interpreted?",
        ["Frequency (trips/hour) – higher is better", "Headway (minutes) – lower is better"],
        index=0
    )
    # Directions: min (lower is better) / max (higher is better)
    metric_direction = {m: "min" for m in ALL_METRICS}
    if "Frequency" in sfq_mode:
        metric_direction["SFQ"] = "max"
    else:
        metric_direction["SFQ"] = "min"

    # Analysis metric selection
    st.sidebar.subheader("Analysis Metrics")
    primary_metric = st.sidebar.selectbox(
        "Primary metric for accessibility analysis",
        ALL_METRICS,
        index=ALL_METRICS.index("JRT"),
        help="Choose which metric to use for finding the 'best' destination"
    )
    st.sidebar.info(f"Using **{primary_metric}**: {metric_descriptions.get(primary_metric, primary_metric)}")

    # Secondary metrics to include in results
    st.sidebar.subheader("Additional Metrics to Include")
    secondary_metrics = []
    for metric in ALL_METRICS:
        if metric != primary_metric:
            if st.sidebar.checkbox(f"Include {metric} ({metric_descriptions[metric]})", value=True):
                secondary_metrics.append(metric)

if uploaded_zones_file:
    zones_df = pd.read_csv(uploaded_zones_file, sep=';')
    st.sidebar.subheader("Zone Mapping File")
    st.sidebar.write(f"📁 {len(zones_df)} zones loaded")

    id_columns = ['id', 'name']
    destination_columns = [col for col in zones_df.columns if col not in id_columns and not zones_df[col].isna().all()]

    with st.sidebar.expander("Available destination categories in mapping file"):
        for col in destination_columns:
            non_null_count = zones_df[col].notna().sum()
            unique_destinations = zones_df[col].nunique()
            st.write(f"• **{col}**: {non_null_count} zones mapped to {unique_destinations} unique destinations")
            if unique_destinations == 1:
                single_dest = zones_df[col].dropna().iloc[0]
                st.write(f"  ↳ All zones map to destination: **{single_dest}**")

if uploaded_skim_file and uploaded_zones_file:
    # Origin zones configuration (fixed range)
    st.sidebar.subheader("Origin Zone Configuration")
    origin_start = 1
    origin_end = 453
    st.sidebar.info(f"Origin zones: {origin_start}-{origin_end} (fixed)")
    normal_zones = range(origin_start, origin_end + 1)

    # Analysis mode selection
    st.sidebar.subheader("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "How do you want to find destinations?",
        [
            "Nearest within range",
            "Use zone mapping (if available)",
            "Both methods",
            "Zone mapping with fallback to nearest"
        ],
        help="Choose how to determine which destinations to analyze for each origin zone"
    )

    mapping_column = None
    if analysis_mode in ["Use zone mapping (if available)", "Both methods", "Zone mapping with fallback to nearest"]:
        mapping_column = st.sidebar.selectbox(
            "Select destination mapping column",
            [None] + destination_columns,
            help="Choose which column contains the destination assignments"
        )

        if mapping_column:
            mapped_zones = zones_df[mapping_column].notna().sum()
            unique_destinations = zones_df[mapping_column].nunique()
            st.sidebar.success(f"✅ {mapped_zones} zones have mapped destinations in '{mapping_column}'")
            st.sidebar.info(f"📊 Maps to {unique_destinations} unique destinations")

            if analysis_mode == "Zone mapping with fallback to nearest":
                unmapped_zones = len(zones_df) - mapped_zones
                st.sidebar.info(f"🔄 {unmapped_zones} zones will use nearest fallback")

    # Process the analysis
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        st.subheader("Processing Skim Matrix...")


        # Read skim matrix - robust loader
        def load_skim(file):
            try:
                df = pd.read_csv(file)
                if set(["OrigZoneNo", "DestZoneNo"]).issubset(df.columns) and len(df.columns) >= 4:
                    return df
                # fallback old-style parse (lines starting with digits)
                file.seek(0)
                content = file.read().decode("utf-8")
                lines = [line.strip() for line in content.splitlines() if re.match(r'^\d', line)]
                data = "\n".join(lines)
                df = pd.read_csv(
                    io.StringIO(data),
                    names=DEFAULT_COLS,
                    on_bad_lines="skip",
                    thousands=","
                )
                return df
            except:
                file.seek(0)
                content = file.read().decode("utf-8")
                lines = [line.strip() for line in content.splitlines() if re.match(r'^\d', line)]
                data = "\n".join(lines)
                df = pd.read_csv(
                    io.StringIO(data),
                    names=DEFAULT_COLS,
                    on_bad_lines="skip",
                    thousands=","
                )
                return df


        skim_df = load_skim(uploaded_skim_file)

        # Clean and convert data types
        numeric_columns = ["OrigZoneNo", "DestZoneNo"] + list(ALL_METRICS)
        for col in numeric_columns:
            if col in skim_df.columns:
                skim_df[col] = pd.to_numeric(skim_df[col], errors='coerce')

        # Drop rows with missing origin or destination
        skim_df = skim_df.dropna(subset=['OrigZoneNo', 'DestZoneNo'])

        # Replace impossible/sentinel values with NaN for time & distance metrics
        for col in ALL_METRICS:
            if col in skim_df.columns and (col in TIME_METRICS or col in DIST_METRICS):
                skim_df[col] = skim_df[col].replace(999999, np.nan)

        # Filter to normal zones for origins and destination range
        skim_df = skim_df[skim_df['OrigZoneNo'].isin(normal_zones)]
        skim_df = skim_df[skim_df['DestZoneNo'].between(dest_start, dest_end)]

        # Display skim matrix info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(skim_df):,}")
        with col2:
            st.metric("Origin Zones", f"{skim_df['OrigZoneNo'].nunique()}")
        with col3:
            st.metric("Destination Zones", f"{skim_df['DestZoneNo'].nunique()}")
        with col4:
            available_metrics = [col for col in ALL_METRICS if col in skim_df.columns]
            st.metric("Available Metrics", len(available_metrics))

        with st.expander("📊 Skim Matrix Sample Data"):
            sample_cols = ['OrigZoneNo', 'DestZoneNo'] + available_metrics
            st.dataframe(skim_df[sample_cols].head(10), use_container_width=True)

        if skim_df.empty:
            st.error("⚠️ No data found in skim matrix for the specified destination range!")
            st.stop()

        # Initialize results
        results = pd.DataFrame({'OrigZoneNo': sorted(set(skim_df['OrigZoneNo']))})

        # Merge with zone names
        zone_names = zones_df[['id', 'name']].rename(columns={'id': 'OrigZoneNo', 'name': 'ZoneName'})
        results = results.merge(zone_names, on='OrigZoneNo', how='left')


        # Helper: find best row based on primary metric + direction
        def find_best_destination(group_df, primary_col):
            valid_rows = group_df.dropna(subset=[primary_col])
            if valid_rows.empty:
                return None
            direction = metric_direction.get(primary_col, "min")
            if direction == "max":
                return valid_rows.loc[valid_rows[primary_col].idxmax()]
            return valid_rows.loc[valid_rows[primary_col].idxmin()]


        # Method 1: Nearest within range
        if analysis_mode in ["Nearest within range", "Both methods"]:
            st.subheader(
                f"🔍 Nearest {skim_type} Analysis (by {primary_metric} / {'maximize' if metric_direction[primary_metric] == 'max' else 'minimize'})")

            best_destinations = []
            for orig_zone in results['OrigZoneNo']:
                zone_data = skim_df[skim_df['OrigZoneNo'] == orig_zone]
                best_row = find_best_destination(zone_data, primary_metric)

                result_row = {'OrigZoneNo': orig_zone}
                if best_row is not None:
                    result_row[f'Nearest_Zone'] = int(best_row['DestZoneNo'])
                    result_row[f'Nearest_{primary_metric}'] = best_row[primary_metric]
                    for metric in secondary_metrics:
                        if metric in best_row:
                            result_row[f'Nearest_{metric}'] = best_row[metric]
                else:
                    result_row[f'Nearest_Zone'] = np.nan
                    result_row[f'Nearest_{primary_metric}'] = np.nan
                    for metric in secondary_metrics:
                        result_row[f'Nearest_{metric}'] = np.nan
                best_destinations.append(result_row)

            if best_destinations:
                nearest_df = pd.DataFrame(best_destinations)
                results = results.merge(nearest_df, on='OrigZoneNo', how='left')

                valid_vals = results[f'Nearest_{primary_metric}'].dropna()
                if len(valid_vals) > 0:
                    cols = st.columns(len(secondary_metrics) + 4)
                    with cols[0]:
                        st.metric("Zones with Access", f"{len(valid_vals)}")
                    with cols[1]:
                        st.metric(f"Avg {primary_metric}", f"{valid_vals.mean():.2f}")
                    with cols[2]:
                        st.metric(f"Min {primary_metric}", f"{valid_vals.min():.2f}")
                    with cols[3]:
                        st.metric(f"Max {primary_metric}", f"{valid_vals.max():.2f}")
                    for i, metric in enumerate(secondary_metrics, 4):
                        colname = f'Nearest_{metric}'
                        if colname in results.columns:
                            vv = results[colname].dropna()
                            if len(vv) > 0 and i < len(cols):
                                with cols[i]:
                                    st.metric(f"Avg {metric}", f"{vv.mean():.2f}")

        # Method 2: Use zone mapping
        if analysis_mode in ["Use zone mapping (if available)", "Both methods"] and mapping_column:
            st.subheader(f"🗺️ Zone Mapping Analysis ({mapping_column})")

            zone_dest_mapping = zones_df[['id', mapping_column]].dropna()
            zone_dest_mapping.columns = ['OrigZoneNo', 'DestZoneNo']

            mapping_results = []
            destinations_in_range = 0
            destinations_outside_range = 0

            for _, row in zone_dest_mapping.iterrows():
                orig_zone = row['OrigZoneNo']
                dest_zone = row['DestZoneNo']

                if not (dest_start <= dest_zone <= dest_end):
                    destinations_outside_range += 1
                    continue

                destinations_in_range += 1
                travel_row = skim_df[(skim_df['OrigZoneNo'] == orig_zone) & (skim_df['DestZoneNo'] == dest_zone)]

                result_row = {'OrigZoneNo': orig_zone, 'Mapped_Zone': dest_zone}
                if not travel_row.empty:
                    result_row[f'Mapped_{primary_metric}'] = travel_row[primary_metric].iloc[0]
                    for metric in secondary_metrics:
                        if metric in travel_row.columns:
                            result_row[f'Mapped_{metric}'] = travel_row[metric].iloc[0]
                else:
                    result_row[f'Mapped_{primary_metric}'] = np.nan
                    for metric in secondary_metrics:
                        result_row[f'Mapped_{metric}'] = np.nan
                mapping_results.append(result_row)

            if mapping_results:
                mapping_df = pd.DataFrame(mapping_results)
                results = results.merge(mapping_df, on='OrigZoneNo', how='left')

                st.info(f"📊 **Mapping Analysis Results:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Destinations in Range", destinations_in_range)
                with col2:
                    st.metric("Destinations Outside Range", destinations_outside_range)
                with col3:
                    valid_mapped = results[f'Mapped_{primary_metric}'].dropna()
                    st.metric("Successful Mappings", len(valid_mapped))

                if len(valid_mapped) > 0:
                    st.write(
                        f"**Mapped {primary_metric} Stats:** Avg: {valid_mapped.mean():.2f}, Min: {valid_mapped.min():.2f}, Max: {valid_mapped.max():.2f}")
                    for metric in secondary_metrics:
                        coln = f'Mapped_{metric}'
                        if coln in results.columns:
                            vv = results[coln].dropna()
                            if len(vv) > 0:
                                st.write(
                                    f"**{metric} Stats:** Avg: {vv.mean():.2f}, Min: {vv.min():.2f}, Max: {vv.max():.2f}")

        # Method 4: Zone mapping with fallback to nearest
        if analysis_mode == "Zone mapping with fallback to nearest" and mapping_column:
            st.subheader(f"🔄 Zone Mapping with Fallback Analysis ({mapping_column})")

            zone_dest_mapping = zones_df[['id', mapping_column]].dropna()
            zone_dest_mapping.columns = ['OrigZoneNo', 'DestZoneNo']

            # Pre-calc nearest by primary metric
            nearest_fallback = {}
            for orig_zone in results['OrigZoneNo']:
                zone_data = skim_df[skim_df['OrigZoneNo'] == orig_zone]
                best_row = find_best_destination(zone_data, primary_metric)
                if best_row is not None:
                    fallback_data = {'dest_zone': int(best_row['DestZoneNo']), primary_metric: best_row[primary_metric]}
                    for metric in secondary_metrics:
                        if metric in best_row:
                            fallback_data[metric] = best_row[metric]
                    nearest_fallback[orig_zone] = fallback_data

            fallback_results = []
            zones_used_mapping = zones_used_fallback = zones_no_access = destinations_outside_range = 0

            for orig_zone in results['OrigZoneNo']:
                mapping_row = zone_dest_mapping[zone_dest_mapping['OrigZoneNo'] == orig_zone]

                if not mapping_row.empty:
                    dest_zone = mapping_row['DestZoneNo'].iloc[0]

                    if not (dest_start <= dest_zone <= dest_end):
                        destinations_outside_range += 1
                        if orig_zone in nearest_fallback:
                            fb = nearest_fallback[orig_zone]
                            result_row = {
                                'OrigZoneNo': orig_zone,
                                'Fallback_Zone': fb['dest_zone'],
                                f'Fallback_{primary_metric}': fb[primary_metric],
                                'Method_Used': 'Nearest (mapped dest outside range)'
                            }
                            for metric in secondary_metrics:
                                if metric in fb:
                                    result_row[f'Fallback_{metric}'] = fb[metric]
                            fallback_results.append(result_row)
                            zones_used_fallback += 1
                        else:
                            result_row = {'OrigZoneNo': orig_zone, 'Fallback_Zone': np.nan,
                                          f'Fallback_{primary_metric}': np.nan, 'Method_Used': 'No access'}
                            for metric in secondary_metrics:
                                result_row[f'Fallback_{metric}'] = np.nan
                            fallback_results.append(result_row)
                            zones_no_access += 1
                        continue

                    travel_row = skim_df[(skim_df['OrigZoneNo'] == orig_zone) & (skim_df['DestZoneNo'] == dest_zone)]
                    if not travel_row.empty:
                        result_row = {
                            'OrigZoneNo': orig_zone,
                            'Fallback_Zone': dest_zone,
                            f'Fallback_{primary_metric}': travel_row[primary_metric].iloc[0],
                            'Method_Used': 'Zone mapping'
                        }
                        for metric in secondary_metrics:
                            if metric in travel_row.columns:
                                result_row[f'Fallback_{metric}'] = travel_row[metric].iloc[0]
                        fallback_results.append(result_row)
                        zones_used_mapping += 1
                    else:
                        if orig_zone in nearest_fallback:
                            fb = nearest_fallback[orig_zone]
                            result_row = {
                                'OrigZoneNo': orig_zone,
                                'Fallback_Zone': fb['dest_zone'],
                                f'Fallback_{primary_metric}': fb[primary_metric],
                                'Method_Used': 'Nearest (mapping failed)'
                            }
                            for metric in secondary_metrics:
                                if metric in fb:
                                    result_row[f'Fallback_{metric}'] = fb[metric]
                            fallback_results.append(result_row)
                            zones_used_fallback += 1
                        else:
                            result_row = {'OrigZoneNo': orig_zone, 'Fallback_Zone': np.nan,
                                          f'Fallback_{primary_metric}': np.nan, 'Method_Used': 'No access'}
                            for metric in secondary_metrics:
                                result_row[f'Fallback_{metric}'] = np.nan
                            fallback_results.append(result_row)
                            zones_no_access += 1
                else:
                    if orig_zone in nearest_fallback:
                        fb = nearest_fallback[orig_zone]
                        result_row = {
                            'OrigZoneNo': orig_zone,
                            'Fallback_Zone': fb['dest_zone'],
                            f'Fallback_{primary_metric}': fb[primary_metric],
                            'Method_Used': 'Nearest (no mapping)'
                        }
                        for metric in secondary_metrics:
                            if metric in fb:
                                result_row[f'Fallback_{metric}'] = fb[metric]
                        fallback_results.append(result_row)
                        zones_used_fallback += 1
                    else:
                        result_row = {'OrigZoneNo': orig_zone, 'Fallback_Zone': np.nan,
                                      f'Fallback_{primary_metric}': np.nan, 'Method_Used': 'No access'}
                        for metric in secondary_metrics:
                            result_row[f'Fallback_{metric}'] = np.nan
                        fallback_results.append(result_row)
                        zones_no_access += 1

            if fallback_results:
                fallback_df = pd.DataFrame(fallback_results)
                results = results.merge(fallback_df, on='OrigZoneNo', how='left')

                st.info(f"🔄 **Fallback Analysis Results:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Used Zone Mapping", zones_used_mapping)
                with col2:
                    st.metric("Used Nearest Fallback", zones_used_fallback)
                with col3:
                    st.metric("No Access", zones_no_access)
                with col4:
                    if destinations_outside_range > 0:
                        st.metric("Dest. Outside Range", destinations_outside_range)

                valid_fb = results[f'Fallback_{primary_metric}'].dropna()
                if len(valid_fb) > 0:
                    st.write(
                        f"**Combined {primary_metric} Stats:** Avg: {valid_fb.mean():.2f}, Min: {valid_fb.min():.2f}, Max: {valid_fb.max():.2f}")
                    for metric in secondary_metrics:
                        coln = f'Fallback_{metric}'
                        if coln in results.columns:
                            vv = results[coln].dropna()
                            if len(vv) > 0:
                                st.write(
                                    f"**{metric} Stats:** Avg: {vv.mean():.2f}, Min: {vv.min():.2f}, Max: {vv.max():.2f}")

                if 'Method_Used' in results.columns:
                    method_counts = results['Method_Used'].value_counts()
                    if not method_counts.empty:
                        st.write("**Method Usage Breakdown:**")
                        for method, count in method_counts.items():
                            percentage = (count / len(results)) * 100
                            st.write(f"• {method}: {count} zones ({percentage:.1f}%)")

        # Population (optional)
        if pop_file:
            try:
                pop_df = pd.read_excel(pop_file)
                if 'OrigZoneNo' in pop_df.columns and 'Population' in pop_df.columns:
                    pop_df = pop_df[['OrigZoneNo', 'Population']]
                    results = results.merge(pop_df, on='OrigZoneNo', how='left')
                    results['Population'] = results['Population'].fillna(0).astype(int)
                    st.success(f"✅ Population data added for {len(pop_df)} zones")
                else:
                    st.error("⚠️ Population file must have columns: 'OrigZoneNo' and 'Population'")
            except Exception as e:
                st.error(f"⚠️ Error loading population file: {str(e)}")

        # Display results
        st.subheader("🎯 Accessibility Results")

        display_columns = ['OrigZoneNo', 'ZoneName']
        if 'Population' in results.columns:
            display_columns.append('Population')

        metric_columns, zone_columns, method_columns = [], [], []
        for col in results.columns:
            if any(col.endswith(f'_{m}') for m in ALL_METRICS):
                metric_columns.append(col)
            elif col.endswith('_Zone'):
                zone_columns.append(col)
            elif col.endswith('_Used'):
                method_columns.append(col)

        primary_cols = [col for col in metric_columns if col.endswith(f'_{primary_metric}')]
        secondary_cols = [col for col in metric_columns if col not in primary_cols]

        display_columns.extend(primary_cols + secondary_cols + zone_columns + method_columns)
        display_columns = [col for col in display_columns if col in results.columns]

        st.dataframe(results[display_columns], use_container_width=True)

        # Download results
        csv_download = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_download,
            file_name=f"accessibility_{skim_type.lower().replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_results.csv",
            mime="text/csv"
        )

        # ===== Population accessibility summary with thresholds =====
        if 'Population' in results.columns:
            st.subheader("👥 Population Accessibility Summary")


            # Default thresholds by metric type
            def default_thresholds_for(metric):
                if metric in COUNT_METRICS:  # transfers
                    return "0,1,2,3,4,5"
                if metric in DIST_METRICS:  # distance (km)
                    return "1,2,5,10,15,20"
                if metric in FREQ_METRICS:  # frequency/headway
                    return "2,4,6,8,12" if metric_direction["SFQ"] == "max" else "5,10,15,20,30"
                # time metrics (minutes)
                return "15,30,45,60,90,120,180"


            threshold_input = st.text_input(
                f"Thresholds for {primary_metric} (comma-separated)",
                value=default_thresholds_for(primary_metric)
            )

            try:
                # parse thresholds as float then round if transfers
                thresholds = [float(x.strip()) for x in threshold_input.split(',') if x.strip() != ""]
                if primary_metric in COUNT_METRICS:
                    thresholds = [int(round(x)) for x in thresholds]

                summary_rows = []
                total_pop = results['Population'].sum()

                primary_metric_columns = [col for col in metric_columns if col.endswith(f'_{primary_metric}')]

                for coln in primary_metric_columns:
                    method_name = coln.replace(f'_{primary_metric}', '').replace('_', ' ').title()
                    for thr in thresholds:
                        if metric_direction.get(primary_metric, "min") == "max":
                            accessible_mask = results[coln] >= thr
                        else:
                            accessible_mask = results[coln] <= thr
                        accessible_pop = results.loc[accessible_mask, 'Population'].sum()
                        accessible_zones = int(accessible_mask.sum())
                        total_zones = len(results)
                        pop_pct = 100 * accessible_pop / total_pop if total_pop > 0 else 0
                        zone_pct = 100 * accessible_zones / total_zones if total_zones > 0 else 0

                        summary_rows.append({
                            "Method": method_name,
                            f"Threshold ({primary_metric})": thr,
                            "Accessible Population": int(accessible_pop),
                            "Total Population": int(total_pop),
                            "Population (%)": round(pop_pct, 2),
                            "Accessible Zones": accessible_zones,
                            "Total Zones": total_zones,
                            "Zones (%)": round(zone_pct, 2)
                        })

                    # unreachable info
                    unreachable_mask = results[coln].isna()
                    unreachable_pop = int(results.loc[unreachable_mask, 'Population'].sum())
                    unreachable_zones = int(unreachable_mask.sum())
                    if unreachable_zones > 0:
                        st.warning(
                            f"🚫 **{method_name}**: {unreachable_zones} zones "
                            f"({unreachable_pop:,} people) have no reachable destination"
                        )

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(summary_df, use_container_width=True)

                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary as CSV",
                        data=csv_summary,
                        file_name=f"accessibility_summary_{skim_type.lower().replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                        mime="text/csv"
                    )

                    st.subheader("📊 Accessibility Visualization")
                    st.write(f"**Population Accessibility by {primary_metric}**")
                    if len(primary_metric_columns) > 1:
                        pivot_df_pop = summary_df.pivot(index=f'Threshold ({primary_metric})', columns='Method',
                                                        values='Population (%)')
                        st.bar_chart(pivot_df_pop)
                    else:
                        chart_data_pop = summary_df.set_index(f'Threshold ({primary_metric})')['Population (%)']
                        st.bar_chart(chart_data_pop)

                    st.write(f"**Zone Accessibility by {primary_metric}**")
                    if len(primary_metric_columns) > 1:
                        pivot_df_zones = summary_df.pivot(index=f'Threshold ({primary_metric})', columns='Method',
                                                          values='Zones (%)')
                        st.bar_chart(pivot_df_zones)
                    else:
                        chart_data_zones = summary_df.set_index(f'Threshold ({primary_metric})')['Zones (%)']
                        st.bar_chart(chart_data_zones)

            except ValueError:
                st.error("⚠️ Invalid threshold format. Use comma-separated numbers like: 15,30,45,60")

        # ===== Multi-metric comparison =====
        if (secondary_metrics and any(col in results.columns for col in metric_columns)) or True:
            st.subheader("📈 Multi-Metric Analysis")
            st.write("**Correlation between metrics:**")

            available_metric_cols = [col for col in metric_columns if col in results.columns]
            if len(available_metric_cols) >= 2:
                metric_data = results[available_metric_cols].select_dtypes(include=[np.number])
                if not metric_data.empty:
                    correlation_matrix = metric_data.corr()
                    st.write("Correlation matrix between accessibility metrics:")
                    st.dataframe(correlation_matrix.round(3), use_container_width=True)

                    with st.expander("💡 Metric Insights"):
                        for col in available_metric_cols:
                            if col in results.columns:
                                valid_data = results[col].dropna()
                                if len(valid_data) > 0:
                                    col_name = col.replace('_', ' ').title()
                                    st.write(f"**{col_name}:**")
                                    st.write(f"  • Mean: {valid_data.mean():.2f}")
                                    st.write(f"  • Median: {valid_data.median():.2f}")
                                    st.write(f"  • Std Dev: {valid_data.std():.2f}")
                                    st.write(f"  • Range: {valid_data.min():.2f} - {valid_data.max():.2f}")
                                    st.write("")

else:
    st.info("👆 Please upload both files to begin analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 Skim Matrix File
        - **CSV format** with multiple travel metrics
        - **Required columns**: `OrigZoneNo, DestZoneNo, ACD, ACT, EGD, EGT, JRD, JRT, NTR, RID, RIT, SFQ, TWT`
        - **ACD**: Access Distance  
- **ACT**: Access Time (min)  
        - **EGD**: Egress Distance  
- **EGT**: Egress Time (min)  
        - ****: In-Vehicle Time (min)  
        - **JRD**: Journey Distance (km)  
- **RID**: Ride Distance  
        - **JRT**: Journey Time (min)  
        - **NTR**: Number of Transfers  
        - **RIT**: Ride Time (min)  
- **TWT**: Transfer Wait Time (min)  
        - **SFQ**: Service Frequency **(trips/hour)** or **Headway (min)** — choose interpretation in the sidebar

        **Example:**
        ```
OrigZoneNo,DestZoneNo,ACD,ACT,EGD,EGT,JRD,JRT,NTR,RID,RIT,SFQ,TWT
1,501,0.5,3.2,0.6,4.1,7.5,28.5,1,6.9,18.0,6,2.0
1,502,0.8,5.0,0.4,3.0,9.0,34.0,2,8.6,22.5,4,3.0
2,501,0.4,2.8,0.3,2.5,5.0,18.0,0,4.6,12.5,10,1.5
```
        """)

    with col2:
        st.markdown("""
        ### 🗺️ Zone Mapping File  
        - **CSV with semicolon separator**
        - **Format**: `id;name;hospital_id;osakidetza_id;comarca_id;bilbao_id;...`
        - Maps each zone to specific destinations

        **New Features:**
        - ✅ **Expanded metrics**: ACT, EGT, JRD, JRT, NTR, RIT, SFQ
        - ✅ **Primary metric optimization** with min/max logic based on metric type (e.g., SFQ)
        - ✅ **Population-based threshold summaries** supporting ≤ or ≥ depending on the metric
        - ✅ **Visual comparisons & correlations**
        """)

# Footer info
with st.expander("ℹ️ How to use this enhanced tool"):
    st.markdown("""
    ## Metric Optimization
    - Most metrics are minimized (lower is better): **JRT, RIT, ACT, EGT, JRD, NTR**
    - **SFQ** can be interpreted as:
      - **Frequency (trips/hour)** → **maximize**
      - **Headway (minutes)** → **minimize**
      Set this in the sidebar.

    ## Workflow
    1. Upload skim matrix (with new columns)
    2. Choose destination range/type
    3. Set **SFQ interpretation**
    4. Select **primary** and **secondary** metrics
    5. Choose analysis mode (nearest, mapping, fallback)
    6. (Optional) Add population and run summaries/plots
    """)

with st.expander("📖 Understanding the Metrics"):
    st.markdown("""
    **JRT** Total journey time (min) • **NTR** Transfers (count) • **RIT/** Time on vehicle (min)  
    **ACT/EGT** Access/Egress walking time (min) • **JRD** Distance (km)  
    **SFQ** Service offered: frequency (trips/h, higher better) or headway (min, lower better)
    """)
