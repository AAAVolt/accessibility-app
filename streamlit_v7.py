# Enhanced Public Transport Accessibility Analysis — v7 (Auto-metrics)
# Changes in this version:
# - No metric selection UI. We always compute accessibility for ALL metrics.
# - SFQ is fixed as frequency (trips/hour) — higher is better.
# - For each origin and EACH metric, we find the best destination based on that metric.
# - Mapping and Fallback modes compute values for ALL metrics.
# - Population summaries and thresholds are generated for ALL metrics automatically.
# - Unified results view with per-metric best zones and values for each method.

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Enhanced Public Transport Accessibility (Auto-Metrics)", layout="wide")
st.title("🚌 Enhanced Public Transport Accessibility — Auto Metrics")

# =====================
# Upload files
# =====================
uploaded_skim_file = st.file_uploader("Upload your skim matrix CSV", type=["csv"], key="skim")
uploaded_zones_file = st.file_uploader("Upload zone-destination mapping CSV", type=["csv"], key="zones")
pop_file = st.file_uploader("Upload Population XLSX (optional)", type=["xlsx"], key="pop")

# ====== Metric catalog (fixed) ======
ALL_METRICS = [
    "JRT", "NTR", "RIT", "ACT", "EGT", "ACD", "EGD", "SFQ", "JRD", "RID", "TWT"
]
TIME_METRICS = {"JRT", "RIT", "ACT", "EGT", "TWT"}  # minutes
DIST_METRICS = {"JRD", "ACD", "EGD", "RID"}          # distance (km or m)
COUNT_METRICS = {"NTR"}                                   # integer
FREQ_METRICS = {"SFQ"}                                   # frequency (trips/hour)

# Directions: min (lower is better) / max (higher is better)
# SFQ is FIXED as frequency (trips/hour) — higher is better
metric_direction = {m: "min" for m in ALL_METRICS}
metric_direction["SFQ"] = "max"

metric_descriptions = {
    "JRT": "Journey Time (total door-to-door time, min)",
    "NTR": "Number of Transfers",
    "RIT": "Ride Time (time on vehicle, min)",
    "ACT": "Access Time (walk/approach time, min)",
    "EGT": "Egress Time (walk/alight time, min)",
    "ACD": "Access Distance",
    "EGD": "Egress Distance",
    "RID": "Ride Distance",
    "SFQ": "Service Frequency (trips/hour)",
    "JRD": "Journey Distance (km)",
    "TWT": "Transfer Wait Time (min)",
}

DEFAULT_COLS = [
    "OrigZoneNo", "DestZoneNo", "ACD", "ACT", "EGD", "EGT", "JRD", "JRT", "NTR", "RID", "RIT", "SFQ", "TWT"
]

# =====================
# Sidebar configuration (Destinations & Modes)
# =====================
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
            "Custom Range",
        ],
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
        "Custom Range": (500, 600),
    }
    default_start, default_end = predefined_ranges[skim_type]

    st.sidebar.subheader("Destination Zone Configuration")

    if skim_type == "Single Zone (like Bilbao, Uni Bilbao)":
        st.sidebar.info("💡 For single zones, set start and end to the same zone number")
        dest_start = st.sidebar.number_input(
            "Single Destination Zone ID",
            value=1,
            help="The specific zone ID for destinations like Bilbao, University, etc.",
        )
        dest_end = dest_start
        st.sidebar.write(f"Will analyze accessibility to zone: **{dest_start}**")
    elif skim_type == "Comarca (within normal zones)":
        st.sidebar.info("💡 Comarca zones are within the normal zone range (1-453)")
        dest_start = st.sidebar.number_input(
            "Comarca Zone Start", value=default_start, min_value=1, max_value=453,
            help="Starting zone ID for comarca destinations",
        )
        dest_end = st.sidebar.number_input(
            "Comarca Zone End", value=default_end, min_value=dest_start, max_value=453,
            help="Ending zone ID for comarca destinations",
        )
        st.sidebar.write(f"Will analyze accessibility to comarca zones: **{dest_start}-{dest_end}**")
    else:
        dest_start = st.sidebar.number_input(
            "Destination Zone Start", value=default_start,
            help="Starting zone ID for destinations in this skim matrix",
        )
        dest_end = st.sidebar.number_input(
            "Destination Zone End", value=default_end,
            help="Ending zone ID for destinations in this skim matrix",
        )
        st.sidebar.info(f"Will analyze accessibility to destinations in zones **{dest_start}-{dest_end}**")

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
            "Zone mapping with fallback to nearest",
        ],
        help="Choose how to determine which destinations to analyze for each origin zone",
    )

    mapping_column = None
    if analysis_mode in ["Use zone mapping (if available)", "Both methods", "Zone mapping with fallback to nearest"]:
        mapping_column = st.sidebar.selectbox(
            "Select destination mapping column",
            [None] + destination_columns,
            help="Choose which column contains the destination assignments",
        )

        if mapping_column:
            mapped_zones = zones_df[mapping_column].notna().sum()
            unique_destinations = zones_df[mapping_column].nunique()
            st.sidebar.success(f"✅ {mapped_zones} zones have mapped destinations in '{mapping_column}'")
            st.sidebar.info(f"📊 Maps to {unique_destinations} unique destinations")

            if analysis_mode == "Zone mapping with fallback to nearest":
                unmapped_zones = len(zones_df) - mapped_zones
                st.sidebar.info(f"🔄 {unmapped_zones} zones will use nearest fallback")

    # =====================
    # Run Analysis
    # =====================
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        st.subheader("Processing Skim Matrix...")

        # Robust loader
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
                df = pd.read_csv(io.StringIO(data), names=DEFAULT_COLS, on_bad_lines="skip", thousands=",")
                return df
            except:
                file.seek(0)
                content = file.read().decode("utf-8")
                lines = [line.strip() for line in content.splitlines() if re.match(r'^\d', line)]
                data = "\n".join(lines)
                df = pd.read_csv(io.StringIO(data), names=DEFAULT_COLS, on_bad_lines="skip", thousands=",")
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

        # Utility: enforce unique columns & report duplicates
        def enforce_unique_cols(df: pd.DataFrame, label: str = "DF") -> pd.DataFrame:
            cols = list(map(str, df.columns))
            seen = {}
            new_cols = []
            dups = []
            for c in cols:
                if c in seen:
                    dups.append(c)
                    k = seen[c]
                    new_c = f"{c}__{k}"
                    while new_c in seen:
                        k += 1
                        new_c = f"{c}__{k}"
                    new_cols.append(new_c)
                    seen[c] = k + 1
                    seen[new_c] = 1
                else:
                    new_cols.append(c)
                    seen[c] = 1
            if dups:
                st.warning(f"Duplicate columns resolved in {label}: {sorted(set(dups))}")
            df = df.copy()
            df.columns = new_cols
            return df

        results = enforce_unique_cols(results, label="results:init")

        # Merge with zone names
        zone_names = zones_df[['id', 'name']].rename(columns={'id': 'OrigZoneNo', 'name': 'ZoneName'})
        results = results.merge(zone_names, on='OrigZoneNo', how='left')
        results = enforce_unique_cols(results, label="results:after zone names")

        # Helpers
        def best_row_by_metric(group_df: pd.DataFrame, metric: str):
            if metric not in group_df.columns:
                return None
            valid = group_df.dropna(subset=[metric])
            if valid.empty:
                return None
            if metric_direction.get(metric, 'min') == 'max':
                return valid.loc[valid[metric].idxmax()]
            return valid.loc[valid[metric].idxmin()]

        # =====================
        # Method 1: Nearest within range — for EACH metric
        # =====================
        if analysis_mode in ["Nearest within range", "Both methods"]:
            st.subheader("🔍 Nearest Destination Analysis (per metric)")

            nearest_rows = []
            for orig in results['OrigZoneNo']:
                zone_data = skim_df[skim_df['OrigZoneNo'] == orig]
                row = {"OrigZoneNo": orig}
                for m in ALL_METRICS:
                    if m in skim_df.columns:
                        best = best_row_by_metric(zone_data, m)
                        if best is not None:
                            row[f"NearestZone_{m}"] = int(best['DestZoneNo'])
                            row[f"Nearest_{m}"] = best[m]
                        else:
                            row[f"NearestZone_{m}"] = np.nan
                            row[f"Nearest_{m}"] = np.nan
                nearest_rows.append(row)

            nearest_df = pd.DataFrame(nearest_rows)
            results = results.merge(nearest_df, on='OrigZoneNo', how='left')
            results = enforce_unique_cols(results, label="results:after nearest")

            # Summary stats per metric
            cols = st.columns(4)
            with cols[0]:
                st.metric("Zones with Any Access", int(results[[c for c in results.columns if c.startswith('Nearest_')]].notna().any(axis=1).sum()))
            with cols[1]:
                st.metric("Metrics Evaluated", len([m for m in ALL_METRICS if f"Nearest_{m}" in results.columns]))

            with st.expander("📈 Nearest stats per metric"):
                stat_rows = []
                for m in ALL_METRICS:
                    coln = f"Nearest_{m}"
                    if coln in results.columns:
                        vals = results[coln].dropna()
                        if len(vals) > 0:
                            stat_rows.append({
                                "Metric": m,
                                "Direction": "max" if metric_direction[m] == "max" else "min",
                                "Mean": round(float(vals.mean()), 3),
                                "Min": round(float(vals.min()), 3),
                                "Max": round(float(vals.max()), 3),
                                "Zones": int(vals.count()),
                            })
                if stat_rows:
                    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

        # =====================
        # Method 2: Zone mapping (if available) — compute ALL metrics
        # =====================
        if analysis_mode in ["Use zone mapping (if available)", "Both methods"] and mapping_column:
            st.subheader(f"🗺️ Zone Mapping Analysis ({mapping_column}) — all metrics")

            zone_dest_map = zones_df[['id', mapping_column]].dropna().rename(columns={'id': 'OrigZoneNo', mapping_column: 'DestZoneNo'})
            mapping_rows = []
            in_range = out_range = 0

            for _, r in zone_dest_map.iterrows():
                orig = r['OrigZoneNo']
                dest = r['DestZoneNo']
                if not (dest_start <= dest <= dest_end):
                    out_range += 1
                    continue
                in_range += 1
                trip = skim_df[(skim_df['OrigZoneNo'] == orig) & (skim_df['DestZoneNo'] == dest)]
                row = {"OrigZoneNo": orig, "Mapped_Zone": int(dest)}
                if not trip.empty:
                    for m in ALL_METRICS:
                        if m in trip.columns:
                            row[f"Mapped_{m}"] = trip[m].iloc[0]
                mapping_rows.append(row)

            if mapping_rows:
                mapping_df = pd.DataFrame(mapping_rows)
                results = results.merge(mapping_df, on='OrigZoneNo', how='left')
                results = enforce_unique_cols(results, label="results:after mapping")

            colA, colB = st.columns(2)
            with colA:
                st.metric("Destinations in Range", in_range)
            with colB:
                st.metric("Destinations Outside Range", out_range)

            with st.expander("📈 Mapping stats per metric"):
                stat_rows = []
                for m in ALL_METRICS:
                    coln = f"Mapped_{m}"
                    if coln in results.columns:
                        vals = results[coln].dropna()
                        if len(vals) > 0:
                            stat_rows.append({
                                "Metric": m,
                                "Direction": "max" if metric_direction[m] == "max" else "min",
                                "Mean": round(float(vals.mean()), 3),
                                "Min": round(float(vals.min()), 3),
                                "Max": round(float(vals.max()), 3),
                                "Zones": int(vals.count()),
                            })
                if stat_rows:
                    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

        # =====================
        # Method 3: Zone mapping with fallback to nearest — ALL metrics
        # =====================
        if analysis_mode == "Zone mapping with fallback to nearest" and mapping_column:
            st.subheader(f"🔄 Zone Mapping with Fallback ({mapping_column}) — all metrics")

            zone_dest_map = zones_df[['id', mapping_column]].dropna().rename(columns={'id': 'OrigZoneNo', mapping_column: 'DestZoneNo'})

            # Precompute nearest best per metric for each origin
            nearest_best = {}
            for orig in results['OrigZoneNo']:
                zone_data = skim_df[skim_df['OrigZoneNo'] == orig]
                best_dict = {}
                for m in ALL_METRICS:
                    if m in skim_df.columns:
                        best = best_row_by_metric(zone_data, m)
                        if best is not None:
                            best_dict[m] = {"dest": int(best['DestZoneNo']), "val": best[m]}
                if best_dict:
                    nearest_best[orig] = best_dict

            fallback_rows = []
            used_mapping = used_fallback = no_access = out_range = 0

            for orig in results['OrigZoneNo']:
                map_row = zone_dest_map[zone_dest_map['OrigZoneNo'] == orig]
                if not map_row.empty:
                    dest = int(map_row['DestZoneNo'].iloc[0])
                    if not (dest_start <= dest <= dest_end):
                        out_range += 1
                        # fallback
                        if orig in nearest_best:
                            row = {"OrigZoneNo": orig, "Method_Used": "Nearest (mapped dest outside range)"}
                            for m, info in nearest_best[orig].items():
                                row[f"Fallback_Zone_{m}"] = info['dest']
                                row[f"Fallback_{m}"] = info['val']
                            fallback_rows.append(row)
                            used_fallback += 1
                        else:
                            fallback_rows.append({"OrigZoneNo": orig, "Method_Used": "No access"})
                            no_access += 1
                        continue

                    # inside range → try mapping values
                    trip = skim_df[(skim_df['OrigZoneNo'] == orig) & (skim_df['DestZoneNo'] == dest)]
                    if not trip.empty:
                        row = {"OrigZoneNo": orig, "Method_Used": "Zone mapping"}
                        for m in ALL_METRICS:
                            if m in trip.columns:
                                row[f"Fallback_Zone_{m}"] = dest
                                row[f"Fallback_{m}"] = trip[m].iloc[0]
                        fallback_rows.append(row)
                        used_mapping += 1
                    else:
                        # fallback to nearest
                        if orig in nearest_best:
                            row = {"OrigZoneNo": orig, "Method_Used": "Nearest (mapping failed)"}
                            for m, info in nearest_best[orig].items():
                                row[f"Fallback_Zone_{m}"] = info['dest']
                                row[f"Fallback_{m}"] = info['val']
                            fallback_rows.append(row)
                            used_fallback += 1
                        else:
                            fallback_rows.append({"OrigZoneNo": orig, "Method_Used": "No access"})
                            no_access += 1
                else:
                    # no mapping → fallback
                    if orig in nearest_best:
                        row = {"OrigZoneNo": orig, "Method_Used": "Nearest (no mapping)"}
                        for m, info in nearest_best[orig].items():
                            row[f"Fallback_Zone_{m}"] = info['dest']
                            row[f"Fallback_{m}"] = info['val']
                        fallback_rows.append(row)
                        used_fallback += 1
                    else:
                        fallback_rows.append({"OrigZoneNo": orig, "Method_Used": "No access"})
                        no_access += 1

            if fallback_rows:
                fallback_df = pd.DataFrame(fallback_rows)
                results = results.merge(fallback_df, on='OrigZoneNo', how='left')
                results = enforce_unique_cols(results, label="results:after fallback")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Used Zone Mapping", used_mapping)
            with col2:
                st.metric("Used Nearest Fallback", used_fallback)
            with col3:
                st.metric("No Access", no_access)
            with col4:
                if out_range > 0:
                    st.metric("Dest. Outside Range", out_range)

            with st.expander("📈 Fallback stats per metric"):
                stat_rows = []
                for m in ALL_METRICS:
                    coln = f"Fallback_{m}"
                    if coln in results.columns:
                        vals = results[coln].dropna()
                        if len(vals) > 0:
                            stat_rows.append({
                                "Metric": m,
                                "Direction": "max" if metric_direction[m] == "max" else "min",
                                "Mean": round(float(vals.mean()), 3),
                                "Min": round(float(vals.min()), 3),
                                "Max": round(float(vals.max()), 3),
                                "Zones": int(vals.count()),
                            })
                if stat_rows:
                    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

        # =====================
        # Population (optional)
        # =====================
        if pop_file:
            try:
                pop_df = pd.read_excel(pop_file)
                if 'OrigZoneNo' in pop_df.columns and 'Population' in pop_df.columns:
                    pop_df = pop_df[['OrigZoneNo', 'Population']]
                    results = results.merge(pop_df, on='OrigZoneNo', how='left')
                    results = enforce_unique_cols(results, label="results:after population")
                    results['Population'] = results['Population'].fillna(0).astype(int)
                    st.success(f"✅ Population data added for {len(pop_df)} zones")
                else:
                    st.error("⚠️ Population file must have columns: 'OrigZoneNo' and 'Population'")
            except Exception as e:
                st.error(f"⚠️ Error loading population file: {str(e)}")

        # =====================
        # Display results
        # =====================
        # Ensure unique, stringified column names to avoid Arrow duplicate-name error
        def _make_unique(cols):
            seen = {}
            out = []
            for c in map(str, cols):
                if c not in seen:
                    seen[c] = 1
                    out.append(c)
                else:
                    k = seen[c]
                    out.append(f"{c}__{k}")
                    seen[c] = k + 1
            return out

        results = results.copy()
        results.columns = _make_unique(results.columns)

        st.subheader("🎯 Accessibility Results (All Metrics)")

        display_columns = ['OrigZoneNo', 'ZoneName']
        if 'Population' in results.columns:
            display_columns.append('Population')

        # Gather metric value columns and zone columns
        metric_value_cols = [c for c in results.columns if any(c.startswith(prefix) for prefix in ["Nearest_", "Mapped_", "Fallback_"]) and not c.endswith("_Used") and not c.startswith("Fallback_Zone_") and not c.startswith("NearestZone_")]
        zone_cols = [c for c in results.columns if c.startswith("NearestZone_") or c.startswith("Fallback_Zone_") or c == "Mapped_Zone"]
        method_cols = [c for c in results.columns if c.endswith("_Used") or c == 'Method_Used']

        display_columns.extend(metric_value_cols + zone_cols + method_cols)
        # Drop duplicates from the display list while preserving order
        display_columns = [c for i, c in enumerate(display_columns) if c in results.columns and c not in display_columns[:i]]

        st.dataframe(results[display_columns], use_container_width=True)

        # Download results
        csv_download = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_download,
            file_name=f"accessibility_{skim_type.lower().replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_auto_metrics.csv",
            mime="text/csv",
        )

        # =====================
        # Population accessibility summary with thresholds (ALL metrics)
        # =====================
        if 'Population' in results.columns:
            st.subheader("👥 Population Accessibility Summary (All Metrics)")

            def default_thresholds_for(metric: str):
                if metric in COUNT_METRICS:   # transfers
                    return [0, 1, 2, 3, 4]
                if metric in DIST_METRICS:   # distance (km)
                    return [1, 2, 5, 10, 15, 20]
                if metric in FREQ_METRICS:   # frequency (trips/hour)
                    return [2, 4, 6, 8, 12]
                # times (minutes)
                return [15, 30, 45, 60, 90, 120]

            # Build a long-form summary across methods & metrics
            summary_rows = []
            total_pop = results['Population'].sum()

            # Collect available (method, metric) column pairs
            method_prefixes = ["Nearest", "Mapped", "Fallback"]
            for method in method_prefixes:
                for m in ALL_METRICS:
                    coln = f"{method}_{m}"
                    if coln in results.columns:
                        thresholds = default_thresholds_for(m)
                        for thr in thresholds:
                            if metric_direction[m] == 'max':
                                mask = results[coln] >= thr
                            else:
                                mask = results[coln] <= thr
                            accessible_pop = int(results.loc[mask, 'Population'].sum())
                            accessible_zones = int(mask.sum())
                            total_zones = len(results)
                            pop_pct = 100 * accessible_pop / total_pop if total_pop > 0 else 0.0
                            zone_pct = 100 * accessible_zones / total_zones if total_zones > 0 else 0.0

                            summary_rows.append({
                                "Method": method,
                                "Metric": m,
                                "Direction": "max" if metric_direction[m] == "max" else "min",
                                "Threshold": thr,
                                "Accessible Population": accessible_pop,
                                "Total Population": int(total_pop),
                                "Population (%)": round(pop_pct, 2),
                                "Accessible Zones": accessible_zones,
                                "Total Zones": total_zones,
                                "Zones (%)": round(zone_pct, 2),
                            })

                        # unreachable (NaN)
                        unreachable_mask = results[coln].isna()
                        unreachable_pop = int(results.loc[unreachable_mask, 'Population'].sum())
                        unreachable_zones = int(unreachable_mask.sum())
                        if unreachable_zones > 0:
                            st.warning(
                                f"🚫 **{method} – {m}**: {unreachable_zones} zones ({unreachable_pop:,} people) have no reachable destination"
                            )

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df = enforce_unique_cols(summary_df, label="summary")
                st.dataframe(summary_df, use_container_width=True)

                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary as CSV",
                    data=csv_summary,
                    file_name=f"accessibility_summary_{skim_type.lower().replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_all_metrics.csv",
                    mime="text/csv",
                )

                st.subheader("📊 Quick Visuals")
                st.write("Population accessibility by threshold — select a metric/method below:")

                # Simple selectors to focus a chart
                m_sel = st.selectbox("Metric", [m for m in ALL_METRICS if any(summary_df['Metric'] == m)])
                method_sel = st.selectbox("Method", sorted(summary_df['Method'].unique()))

                focus = summary_df[(summary_df['Metric'] == m_sel) & (summary_df['Method'] == method_sel)]
                if not focus.empty:
                    chart_data_pop = focus.set_index('Threshold')['Population (%)']
                    st.bar_chart(chart_data_pop)

                    chart_data_zones = focus.set_index('Threshold')['Zones (%)']
                    st.bar_chart(chart_data_zones)

        # =====================
        # Multi-metric correlation (across whichever method columns exist)
        # =====================
        st.subheader("📈 Multi-Metric Analysis")
        st.write("**Correlation between metrics across methods** (numeric columns only):")

        numeric_cols = results.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            corr = numeric_cols.corr()
            st.dataframe(corr.round(3), use_container_width=True)

else:
    st.info("👆 Please upload both files to begin analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 📋 Skim Matrix File
            - **CSV format** with multiple travel metrics
            - **Required columns**: `OrigZoneNo, DestZoneNo, ACD, ACT, EGD, EGT, JRD, JRT, NTR, RID, RIT, SFQ, TWT`
            - **ACD**: Access Distance  
            - **ACT**: Access Time (min)  
            - **EGD**: Egress Distance  
            - **EGT**: Egress Time (min)  
            - **RIT**: In-Vehicle Time (min)  
            - **JRD**: Journey Distance (km)  
            - **RID**: Ride Distance  
            - **JRT**: Journey Time (min)  
            - **NTR**: Number of Transfers  
            - **TWT**: Transfer Wait Time (min)  
            - **SFQ**: Service Frequency **(trips/hour)** — fixed as frequency (higher is better)

            **Example:**
            ```
            OrigZoneNo,DestZoneNo,ACD,ACT,EGD,EGT,JRD,JRT,NTR,RID,RIT,SFQ,TWT
            1,501,0.5,3.2,0.6,4.1,7.5,28.5,1,6.9,18.0,6,2.0
            1,502,0.8,5.0,0.4,3.0,9.0,34.0,2,8.6,22.5,4,3.0
            2,501,0.4,2.8,0.3,2.5,5.0,18.0,0,4.6,12.5,10,1.5
            ```
            """
        )

    with col2:
        st.markdown(
            """
            ### 🗺️ Zone Mapping File  
            - **CSV with semicolon separator**
            - **Format**: `id;name;hospital_id;osakidetza_id;comarca_id;bilbao_id;...`
            - Maps each zone to specific destinations

            **New in v7:**
            - ✅ **Auto-metrics**: compute best destination per origin for **every metric**
            - ✅ **Fixed SFQ interpretation**: trips/hour (higher is better)
            - ✅ **Population summaries** for **all metrics** with smart defaults per type
            - ✅ **Unified results** with per-metric best zones and values across methods
            """
        )

# Footer info
with st.expander("ℹ️ How to use this tool (v7)"):
    st.markdown(
        """
        ## Metric Optimization (no selection needed)
        - Most metrics are minimized (lower is better): **JRT, RIT, ACT, EGT, JRD, NTR, TWT, ACD, EGD, RID**
        - **SFQ** is **always frequency (trips/hour)** → **maximize**.

        ## Workflow
        1. Upload skim matrix (with required columns)
        2. Choose destination range/type
        3. Pick analysis mode (nearest, mapping, fallback)
        4. (Optional) Add population to unlock accessibility summaries
        """
    )

with st.expander("📖 Metrics at a glance"):
    st.markdown(
        """
        **JRT** total journey time (min) • **NTR** transfers (count) • **RIT** time on vehicle (min)  
        **ACT/EGT** access/egress walking time (min) • **JRD** distance (km) • **RID** ride distance  
        **SFQ** service frequency (trips/hour, higher better) • **TWT** transfer wait time (min)  
        **ACD/EGD** access/egress distance
        """
    )
