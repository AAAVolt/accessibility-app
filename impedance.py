import streamlit as st
import pandas as pd
import numpy as np
import re

# Optional matplotlib for enhanced styling
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

st.set_page_config(page_title="Accessibility Analysis", layout="wide")

st.title("🗺️ Accessibility Analysis Tool")
st.markdown("Analyze accessibility from origin zones to destination categories using travel time matrices")

# Show installation info if matplotlib is missing
if not MATPLOTLIB_AVAILABLE:
    with st.expander("💡 Optional Enhancement Available"):
        st.info("""
        **Enhanced Styling Available**: Install matplotlib for color-coded result tables:
        ```
        pip install matplotlib
        ```
        The tool works fully without it, but matplotlib adds nice color gradients to results.
        """)

# Sidebar for inputs
st.sidebar.header("Configuration")

# Step 1: Upload travel time matrix
st.sidebar.subheader("1. Travel Time Matrix")
matrix_file = st.sidebar.file_uploader("Upload MTX file (Visum format)", type=['mtx', 'txt'])

# Step 2: Define categories
st.sidebar.subheader("2. Destination Categories")

# Initialize session state for categories
if 'categories' not in st.session_state:
    st.session_state.categories = []

# Add new category
with st.sidebar.expander("➕ Add New Category"):
    cat_name = st.text_input("Category Name", key="new_cat_name")
    cat_lambda = st.number_input("Lambda (λ) - Decay parameter", min_value=0.0, value=0.02, step=0.01, format="%.3f",
                                 key="new_cat_lambda")
    cat_weight = st.number_input("Category Weight (w) - Overall importance", min_value=0.0, value=1.0, step=0.1,
                                 key="new_cat_weight",
                                 help="Weight of this category in total accessibility calculation")
    cat_file = st.file_uploader("Upload Excel with destinations", type=['xlsx', 'xls'], key="new_cat_file")

    st.markdown("**Expected Excel columns:**")
    st.markdown("- `ID`: Zone ID where destination is located")
    st.markdown("- `nombre`: Name of the destination")
    st.markdown("- `attractiveness` (optional): Individual attractiveness value")

    if st.button("Add Category"):
        if cat_name and cat_file is not None:
            df_dest = pd.read_excel(cat_file)
            required_cols = ['ID', 'nombre']

            if all(col in df_dest.columns for col in required_cols):
                # Check if attractiveness column exists, if not create default values
                if 'attractiveness' not in df_dest.columns:
                    df_dest['attractiveness'] = 1.0
                    st.info(
                        f"No 'attractiveness' column found. Using default value 1.0 for all {cat_name} destinations.")

                # Validate attractiveness values
                if df_dest['attractiveness'].isnull().any():
                    df_dest['attractiveness'] = df_dest['attractiveness'].fillna(1.0)
                    st.warning("Some attractiveness values were missing and filled with 1.0")

                if (df_dest['attractiveness'] < 0).any():
                    st.error("Attractiveness values must be non-negative")
                else:
                    st.session_state.categories.append({
                        'name': cat_name,
                        'lambda': cat_lambda,
                        'weight': cat_weight,
                        'destinations': df_dest
                    })

                    total_attractiveness = df_dest['attractiveness'].sum()
                    st.success(
                        f"✅ Category '{cat_name}' added with {len(df_dest)} destinations (total attractiveness: {total_attractiveness:.2f})")
            else:
                missing_cols = [col for col in required_cols if col not in df_dest.columns]
                st.error(f"Excel must contain columns: {', '.join(missing_cols)}")
        else:
            st.warning("Please provide category name and Excel file")

# Display current categories
if st.session_state.categories:
    st.sidebar.subheader("Current Categories")
    for i, cat in enumerate(st.session_state.categories):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.text(f"{cat['name']} (λ={cat['lambda']}, A={cat['weight']})")
            st.caption(f"{len(cat['destinations'])} destinations")
        with col2:
            if st.button("🗑️", key=f"del_{i}"):
                st.session_state.categories.pop(i)
                st.rerun()


def parse_mtx_file(file_content):
    """
    Parse Visum MTX file format properly and extract zone mapping
    """
    try:
        lines = file_content.decode('utf-8').splitlines()
    except UnicodeDecodeError:
        try:
            lines = file_content.decode('latin-1').splitlines()
        except UnicodeDecodeError:
            raise ValueError("Cannot decode file. Please check file encoding.")

    # Find zone mapping and matrix data
    zone_ids = []
    data_start_idx = None
    n_objects = None

    # Parse header to find zone IDs
    in_zone_section = False
    zone_section_start = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Look for number of network objects
        if line.startswith("* Number of network objects"):
            try:
                if i + 1 < len(lines):
                    n_objects = int(lines[i + 1].strip())
                    st.info(f"Found {n_objects} network objects in header")
            except (ValueError, IndexError):
                st.warning("Could not parse number of network objects")

        # Look for network object numbers section
        if line.startswith("* Network object numbers"):
            in_zone_section = True
            zone_section_start = i
            st.info(f"Zone section starts at line {i + 1}")
            continue

        if in_zone_section:
            if line.startswith('*'):
                # End of zone section
                st.info(f"Zone section ends at line {i + 1}, found {len(zone_ids)} zone IDs")
                in_zone_section = False
                continue

            # Parse zone IDs from this line
            tokens = line.split()
            for token in tokens:
                try:
                    zone_id = int(token)
                    zone_ids.append(zone_id)
                except ValueError:
                    # Skip non-numeric tokens
                    continue

        # Matrix data starts after the zone section - look for first row starting with "* Obj"
        if (not in_zone_section and
                zone_section_start is not None and
                i > zone_section_start and
                line.startswith("* Obj") and "Sum" in line):

            # The actual matrix data starts on the next non-comment line
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if (not next_line.startswith('*') and
                        not next_line.startswith('$') and
                        not next_line.startswith('-') and
                        next_line and
                        re.search(r'\d', next_line)):
                    data_start_idx = j
                    st.info(f"Matrix data starts at line {j + 1}")
                    break
            break

    # Validate what we found
    if not zone_ids:
        raise ValueError("Could not find any zone IDs in MTX file. Check file format.")

    if data_start_idx is None:
        raise ValueError("Could not find matrix data section in MTX file")

    # Verify we have the expected number of zones
    if n_objects and len(zone_ids) != n_objects:
        st.warning(f"Expected {n_objects} zones but found {len(zone_ids)} zone IDs")

    # Create zone ID to matrix position mapping
    zone_to_position = {zone_id: pos for pos, zone_id in enumerate(zone_ids)}

    st.success(f"Successfully parsed zone mapping: {len(zone_ids)} zones")
    st.info(f"Zone ID range: {min(zone_ids)} to {max(zone_ids)}")

    # Parse matrix data - each origin's data spans multiple lines
    matrix_data = []
    current_row_values = []
    current_origin = 0
    values_per_line = 10  # Based on our observation
    expected_values_per_origin = len(zone_ids)

    st.info(f"Expected {expected_values_per_origin} values per origin, {values_per_line} values per line")

    i = data_start_idx
    while i < len(lines):
        line = lines[i].strip()

        # Skip comment lines and empty lines
        if line.startswith('*') or not line:
            # Check if we hit a new origin summary
            if line.startswith('* Obj') and current_row_values:
                # We've completed an origin, save its data
                if len(current_row_values) >= expected_values_per_origin:
                    matrix_data.append(current_row_values[:expected_values_per_origin])
                else:
                    # Pad with 999999 if needed
                    padded_row = current_row_values + [999999.0] * (
                            expected_values_per_origin - len(current_row_values))
                    matrix_data.append(padded_row)

                current_row_values = []
                current_origin += 1

                if current_origin % 50 == 0:
                    st.info(f"Parsed {current_origin} origins...")

            i += 1
            continue

        # Parse numeric values from this line
        tokens = line.split()
        for token in tokens:
            if token == '*':
                current_row_values.append(999999.0)
            else:
                try:
                    val = float(token)
                    if val > 999998:
                        val = 999999.0
                    current_row_values.append(val)
                except ValueError:
                    continue

        i += 1

    # Don't forget the last origin if it doesn't end with a summary line
    if current_row_values:
        if len(current_row_values) >= expected_values_per_origin:
            matrix_data.append(current_row_values[:expected_values_per_origin])
        else:
            padded_row = current_row_values + [999999.0] * (expected_values_per_origin - len(current_row_values))
            matrix_data.append(padded_row)

    if not matrix_data:
        raise ValueError("No valid matrix data found")

    st.info(f"Parsed {len(matrix_data)} origin rows")

    # Convert to numpy array
    matrix = np.array(matrix_data)

    st.success(f"✅ Final matrix shape: {matrix.shape}")

    return matrix, zone_ids, zone_to_position


# Main content area
if matrix_file is not None:
    st.header("📊 Travel Time Matrix")

    try:
        # Read the MTX file
        with st.spinner("Loading matrix..."):
            try:
                travel_time_matrix, zone_ids, zone_to_position = parse_mtx_file(matrix_file.getvalue())
            except Exception as parse_error:
                st.error(f"Error parsing MTX file: {str(parse_error)}")
                raise parse_error

            st.success(
                f"✅ Matrix loaded: {travel_time_matrix.shape[0]} origins × {travel_time_matrix.shape[1]} destinations")

            # Validation and debugging info
            with st.expander("📋 Matrix vs Zone ID Validation"):
                st.write(f"**Matrix dimensions:** {travel_time_matrix.shape[0]} × {travel_time_matrix.shape[1]}")
                st.write(f"**Zone IDs parsed:** {len(zone_ids)}")
                st.write(f"**Expected zones from header:** 718")

                if len(zone_ids) != travel_time_matrix.shape[0]:
                    st.warning(f"⚠️ Mismatch: {len(zone_ids)} zone IDs but {travel_time_matrix.shape[0]} matrix rows")
                if len(zone_ids) != travel_time_matrix.shape[1]:
                    st.warning(
                        f"⚠️ Mismatch: {len(zone_ids)} zone IDs but {travel_time_matrix.shape[1]} matrix columns")

                # Show some missing zone IDs if any
                expected_zones = set(range(1, 719))  # 1-718 based on header
                found_zones = set(zone_ids)
                missing_zones = expected_zones - found_zones
                extra_zones = found_zones - expected_zones

                if missing_zones:
                    st.write(f"**Missing zones (first 10):** {sorted(list(missing_zones))[:10]}")
                if extra_zones:
                    st.write(f"**Extra zones (first 10):** {sorted(list(extra_zones))[:10]}")

            # Display matrix info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Number of Origins", travel_time_matrix.shape[0])
            with col2:
                st.metric("Number of Destinations", travel_time_matrix.shape[1])
            with col3:
                # Calculate average excluding unreachable destinations (999999)
                reachable_times = travel_time_matrix[travel_time_matrix < 999999]
                avg_time = np.mean(reachable_times) if len(reachable_times) > 0 else 0
                st.metric("Avg Travel Time (reachable)", f"{avg_time:.2f} min")
            with col4:
                unreachable_pct = np.sum(travel_time_matrix >= 999999) / travel_time_matrix.size * 100
                st.metric("Unreachable (%)", f"{unreachable_pct:.1f}%")

            # Show zone mapping info
            with st.expander("Zone ID Mapping Information"):
                st.write(f"**Zone ID Range:** {min(zone_ids)} to {max(zone_ids)}")
                st.write(f"**Total Zones Found:** {len(zone_ids)}")
                st.write(f"**Matrix Origins:** {travel_time_matrix.shape[0]}")
                st.write(f"**Matrix Destinations:** {travel_time_matrix.shape[1]}")

                # Show some example mappings
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 10 Zone Mappings:**")
                    for i in range(min(10, len(zone_ids))):
                        st.text(f"Zone {zone_ids[i]} → Matrix position {i + 1}")

                with col2:
                    st.write("**Last 10 Zone Mappings:**")
                    start_idx = max(0, len(zone_ids) - 10)
                    for i in range(start_idx, len(zone_ids)):
                        st.text(f"Zone {zone_ids[i]} → Matrix position {i + 1}")

                # Show some key zones if they exist
                key_zones = [700, 733, 905]
                found_zones = []
                for zone in key_zones:
                    if zone in zone_to_position:
                        found_zones.append(f"Zone {zone} → Matrix position {zone_to_position[zone] + 1}")

                if found_zones:
                    st.write("**Key Zone Mappings:**")
                    for mapping in found_zones:
                        st.text(mapping)
                else:
                    st.write("**Key zones (700, 733, 905) not found in zone mapping**")

            # Show sample of matrix
            with st.expander("Preview Travel Time Matrix (first 10x10)"):
                df_preview = pd.DataFrame(
                    travel_time_matrix[:10, :10],
                    index=[f"Origin {zone_ids[i]}" for i in range(min(10, len(zone_ids)))],
                    columns=[f"Dest {zone_ids[i]}" for i in range(min(10, len(zone_ids)))]
                )


                # Format display to show unreachable as "∞"
                def format_time(val):
                    if val >= 999999:
                        return "∞"
                    else:
                        return f"{val:.1f}"


                styled_df = df_preview.style.format(format_time)
                st.dataframe(styled_df)

            # Calculate accessibility if categories are defined
            if st.session_state.categories:
                st.header("🎯 Accessibility Analysis")

                # Options for handling unreachable destinations and origin filtering
                st.subheader("Analysis Options")

                # Origin zone filtering
                with st.expander("🎯 Origin Zone Filtering"):
                    st.write("**Filter which origin zones to include in the analysis:**")
                    col1, col2 = st.columns(2)

                    with col1:
                        min_origin_id = st.number_input(
                            "Minimum Origin Zone ID",
                            min_value=1,
                            value=1,
                            step=1,
                            help="Lowest zone ID to include in accessibility calculation"
                        )

                    with col2:
                        max_origin_id = st.number_input(
                            "Maximum Origin Zone ID",
                            min_value=min_origin_id,
                            value=453,
                            step=1,
                            help="Highest zone ID to include in accessibility calculation"
                        )

                    st.info(f"Will analyze zones from {min_origin_id} to {max_origin_id} (inclusive)")

                # Travel time options
                st.write("**Travel Time Constraints:**")
                col1, col2 = st.columns(2)

                with col1:
                    max_time = st.number_input(
                        "Maximum travel time to consider (minutes)",
                        min_value=1,
                        max_value=999999,
                        value=120,
                        help="Destinations beyond this time will be considered unreachable"
                    )

                with col2:
                    unreachable_penalty = st.number_input(
                        "Travel time for unreachable destinations",
                        min_value=max_time,
                        value=999999,
                        help="High values make unreachable destinations contribute near zero to accessibility"
                    )

                if st.button("Calculate Accessibility", type="primary"):
                    with st.spinner("Calculating accessibility..."):
                        # Initialize results
                        n_origins = travel_time_matrix.shape[0]
                        n_destinations = travel_time_matrix.shape[1]

                        # Create safe indexing - use only as many zone IDs as we have matrix rows
                        available_origin_zones = zone_ids[:n_origins] if len(zone_ids) >= n_origins else zone_ids

                        # If we don't have enough zone IDs, create sequential ones
                        if len(available_origin_zones) < n_origins:
                            st.warning(
                                f"Only {len(zone_ids)} zone IDs available for {n_origins} matrix rows. Using sequential numbering for remaining origins.")
                            missing_count = n_origins - len(available_origin_zones)
                            max_zone_id = max(zone_ids) if zone_ids else 0
                            additional_zones = list(range(max_zone_id + 1, max_zone_id + 1 + missing_count))
                            available_origin_zones.extend(additional_zones)

                        # FILTER ORIGIN ZONES: Only include zones in specified range
                        st.info(f"🔍 Filtering origin zones to ID range {min_origin_id}-{max_origin_id}...")

                        # Create mapping of filtered zones to their matrix positions
                        filtered_origins = []
                        filtered_matrix_indices = []

                        for matrix_idx, zone_id in enumerate(available_origin_zones[:n_origins]):
                            if min_origin_id <= zone_id <= max_origin_id:
                                filtered_origins.append(zone_id)
                                filtered_matrix_indices.append(matrix_idx)

                        if len(filtered_origins) == 0:
                            st.error(
                                f"❌ No origin zones found in the range {min_origin_id}-{max_origin_id}! Check your zone numbering.")
                            st.stop()

                        # Update the number of origins to only filtered ones
                        n_origins_filtered = len(filtered_origins)

                        st.success(
                            f"✅ Found {n_origins_filtered} origin zones in range {min_origin_id}-{max_origin_id} (from total {n_origins} zones)")

                        accessibility_results = pd.DataFrame(index=filtered_origins)
                        accessibility_results.index.name = 'Origin_Zone'

                        st.info(
                            f"Analysis setup: {n_origins_filtered} filtered origins × {n_destinations} destinations")

                        # Process travel time matrix
                        processed_matrix = travel_time_matrix.copy()

                        # Replace unreachable values and apply maximum time constraint
                        processed_matrix = np.where(
                            processed_matrix >= 999999,
                            unreachable_penalty,
                            processed_matrix
                        )
                        processed_matrix = np.where(
                            processed_matrix > max_time,
                            unreachable_penalty,
                            processed_matrix
                        )

                        # Calculate for each category with competitive weighting
                        total_accessibility = np.zeros(n_origins_filtered)

                        for cat in st.session_state.categories:
                            cat_name = cat['name']
                            lambda_val = cat['lambda']
                            cat_weight = cat['weight']  # This is now w_category
                            destinations = cat['destinations']

                            # Step 1: Calculate normalized attractiveness for this category
                            # Â_j = A_j / Σ A_category
                            total_attractiveness = destinations['attractiveness'].sum()

                            if total_attractiveness == 0:
                                st.warning(f"Category '{cat_name}' has zero total attractiveness")
                                continue

                            # Normalize attractiveness within category (competitive weighting)
                            destinations_normalized = destinations.copy()
                            destinations_normalized['normalized_attractiveness'] = (
                                    destinations_normalized['attractiveness'] / total_attractiveness
                            )

                            # Step 2: Apply decay function for each destination in category
                            category_accessibility = np.zeros(n_origins_filtered)
                            valid_destinations = 0

                            for _, dest_row in destinations_normalized.iterrows():
                                dest_id = dest_row['ID']
                                normalized_attr = dest_row['normalized_attractiveness']

                                # Check if destination ID exists in zone mapping
                                if dest_id in zone_to_position:
                                    dest_idx = zone_to_position[dest_id]  # Get matrix position
                                    valid_destinations += 1

                                    # Get travel times from filtered origins to this destination
                                    travel_times = processed_matrix[filtered_matrix_indices, dest_idx]

                                    # Apply decay function with normalized attractiveness:
                                    # Â_j * e^(-λ * c_ij)
                                    category_accessibility += normalized_attr * np.exp(-lambda_val * travel_times)

                            # Step 3: Apply category weight
                            # w_category * Σ (Â_j * e^(-λ * c_ij))
                            weighted_category_accessibility = cat_weight * category_accessibility

                            # Store individual category results
                            accessibility_results[cat_name] = weighted_category_accessibility

                            # Add to total
                            total_accessibility += weighted_category_accessibility

                            # Show mapping results
                            st.info(
                                f"Category '{cat_name}': {valid_destinations}/{len(destinations)} destinations found in matrix")
                            if valid_destinations > 0:
                                st.info(f"  - Total original attractiveness: {total_attractiveness:.2f}")
                                st.info(f"  - Category weight: {cat_weight}")

                        # Store total accessibility
                        accessibility_results['Total_Accessibility'] = total_accessibility

                        # Add normalized columns (0-1 scale using min-max normalization)
                        st.info("Adding normalized columns (0-1 scale)...")
                        for col in accessibility_results.columns:
                            if accessibility_results[col].max() > accessibility_results[col].min():
                                # Min-max normalization: (x - min) / (max - min)
                                normalized_col = f"{col}_Normalized"
                                accessibility_results[normalized_col] = (
                                        (accessibility_results[col] - accessibility_results[col].min()) /
                                        (accessibility_results[col].max() - accessibility_results[col].min())
                                )
                            else:
                                # Handle case where all values are the same
                                normalized_col = f"{col}_Normalized"
                                accessibility_results[normalized_col] = 1.0

                        # Display results
                        st.success("✅ Accessibility calculation complete!")

                        # Summary statistics
                        st.subheader("Summary Statistics")
                        summary_cols = st.columns(len(st.session_state.categories) + 1)

                        for i, cat in enumerate(st.session_state.categories):
                            with summary_cols[i]:
                                cat_values = accessibility_results[cat['name']]
                                normalized_values = accessibility_results[f"{cat['name']}_Normalized"]
                                st.metric(
                                    f"{cat['name']} (w={cat['weight']})",
                                    f"{cat_values.mean():.3f}",
                                    delta=f"Max: {cat_values.max():.3f}"
                                )
                                st.caption(f"Normalized avg: {normalized_values.mean():.3f}")

                        with summary_cols[-1]:
                            total_values = accessibility_results['Total_Accessibility']
                            normalized_total = accessibility_results['Total_Accessibility_Normalized']
                            st.metric(
                                "Total",
                                f"{total_values.mean():.3f}",
                                delta=f"Max: {total_values.max():.3f}"
                            )
                            st.caption(f"Normalized avg: {normalized_total.mean():.3f}")

                        # Show detailed results
                        st.subheader("Detailed Results")

                        # Add tabs for different views
                        tab1, tab2, tab3 = st.tabs(["📊 All Results", "🔢 Original Values", "📈 Normalized Values (0-1)"])

                        with tab1:
                            st.write("**Complete results with both original and normalized values**")
                            display_df = accessibility_results.copy()
                            # Use enhanced styling if matplotlib is available, otherwise simple formatting
                            if MATPLOTLIB_AVAILABLE:
                                try:
                                    st.dataframe(
                                        display_df.style.format("{:.6f}").background_gradient(cmap='YlGnBu'),
                                        height=400
                                    )
                                except Exception:
                                    # Fallback if styling fails
                                    st.dataframe(display_df.style.format("{:.6f}"), height=400)
                            else:
                                st.dataframe(display_df.style.format("{:.6f}"), height=400)

                        with tab2:
                            st.write("**Original accessibility values**")
                            # Show only original columns
                            original_cols = [col for col in accessibility_results.columns if
                                             not col.endswith('_Normalized')]
                            display_df = accessibility_results[original_cols].copy()
                            # Use enhanced styling if matplotlib is available, otherwise simple formatting
                            if MATPLOTLIB_AVAILABLE:
                                try:
                                    st.dataframe(
                                        display_df.style.format("{:.6f}").background_gradient(cmap='YlGnBu'),
                                        height=400
                                    )
                                except Exception:
                                    # Fallback if styling fails
                                    st.dataframe(display_df.style.format("{:.6f}"), height=400)
                            else:
                                st.dataframe(display_df.style.format("{:.6f}"), height=400)

                        with tab3:
                            st.write("**Normalized accessibility values (0-1 scale)**")
                            # Show only normalized columns
                            normalized_cols = [col for col in accessibility_results.columns if
                                               col.endswith('_Normalized')]
                            display_df = accessibility_results[normalized_cols].copy()
                            # Remove '_Normalized' suffix from column names for cleaner display
                            display_df.columns = [col.replace('_Normalized', '') for col in display_df.columns]
                            # Use enhanced styling if matplotlib is available, otherwise simple formatting
                            if MATPLOTLIB_AVAILABLE:
                                try:
                                    st.dataframe(
                                        display_df.style.format("{:.6f}").background_gradient(cmap='YlGnBu'),
                                        height=400
                                    )
                                except Exception:
                                    # Fallback if styling fails
                                    st.dataframe(display_df.style.format("{:.6f}"), height=400)
                            else:
                                st.dataframe(display_df.style.format("{:.6f}"), height=400)

                        # Download results
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            csv_all = accessibility_results.to_csv()
                            st.download_button(
                                label="📥 Download All Results (CSV)",
                                data=csv_all,
                                file_name="accessibility_results_complete.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Original values only
                            original_cols = [col for col in accessibility_results.columns if
                                             not col.endswith('_Normalized')]
                            csv_original = accessibility_results[original_cols].to_csv()
                            st.download_button(
                                label="📥 Download Original Values (CSV)",
                                data=csv_original,
                                file_name="accessibility_results_original.csv",
                                mime="text/csv"
                            )

                        with col3:
                            # Normalized values only
                            normalized_cols = [col for col in accessibility_results.columns if
                                               col.endswith('_Normalized')]
                            normalized_df = accessibility_results[normalized_cols].copy()
                            normalized_df.columns = [col.replace('_Normalized', '') for col in normalized_df.columns]
                            csv_normalized = normalized_df.to_csv()
                            st.download_button(
                                label="📥 Download Normalized Values (CSV)",
                                data=csv_normalized,
                                file_name="accessibility_results_normalized.csv",
                                mime="text/csv"
                            )

                        # Additional analysis
                        st.subheader("Additional Analysis")

                        # Add tabs for original vs normalized rankings
                        rank_tab1, rank_tab2 = st.tabs(["🔢 Original Value Rankings", "📈 Normalized Value Rankings"])

                        with rank_tab1:
                            col1, col2 = st.columns(2)

                            with col1:
                                # Top 10 most accessible origins (original values)
                                st.write("**Top 10 Most Accessible Origins (Original Values)**")
                                top_10 = accessibility_results.nlargest(10, 'Total_Accessibility')[
                                    ['Total_Accessibility']]
                                st.dataframe(top_10.style.format("{:.6f}"))

                            with col2:
                                # Bottom 10 least accessible origins (original values)
                                st.write("**10 Least Accessible Origins (Original Values)**")
                                bottom_10 = accessibility_results.nsmallest(10, 'Total_Accessibility')[
                                    ['Total_Accessibility']]
                                st.dataframe(bottom_10.style.format("{:.6f}"))

                        with rank_tab2:
                            col1, col2 = st.columns(2)

                            with col1:
                                # Top 10 most accessible origins (normalized values)
                                st.write("**Top 10 Most Accessible Origins (Normalized Values)**")
                                top_10_norm = accessibility_results.nlargest(10, 'Total_Accessibility_Normalized')[
                                    ['Total_Accessibility_Normalized']]
                                top_10_norm.columns = ['Total_Accessibility']  # Clean column name for display
                                st.dataframe(top_10_norm.style.format("{:.6f}"))

                            with col2:
                                # Bottom 10 least accessible origins (normalized values)
                                st.write("**10 Least Accessible Origins (Normalized Values)**")
                                bottom_10_norm = accessibility_results.nsmallest(10, 'Total_Accessibility_Normalized')[
                                    ['Total_Accessibility_Normalized']]
                                bottom_10_norm.columns = ['Total_Accessibility']  # Clean column name for display
                                st.dataframe(bottom_10_norm.style.format("{:.6f}"))

            else:
                st.info("👆 Add destination categories in the sidebar to calculate accessibility")

    except Exception as e:
        st.error(f"Error loading matrix: {str(e)}")

        # Show debugging info
        with st.expander("🔍 Debug Information"):
            st.text(f"Error type: {type(e).__name__}")
            st.text(f"Error message: {str(e)}")

            # Show file info
            try:
                file_size = len(matrix_file.getvalue())
                st.text(f"File size: {file_size:,} bytes")

                # Show first few lines
                try:
                    lines = matrix_file.getvalue().decode('utf-8').splitlines()[:20]
                    st.text("First 20 lines of file:")
                    for i, line in enumerate(lines, 1):
                        st.text(f"{i:3d}: {line[:100]}")  # Truncate long lines
                except Exception as decode_error:
                    st.text(f"Could not decode file for preview: {decode_error}")

            except Exception as debug_error:
                st.text(f"Could not get debug info: {debug_error}")

            if hasattr(e, '__traceback__'):
                import traceback

                st.text("Full traceback:")
                st.code(traceback.format_exc())

        st.info("""
        **Troubleshooting Tips:**
        1. Make sure the file is a valid Visum MTX export
        2. Check that the file contains the '* Network object numbers' section
        3. Verify the file is not corrupted or truncated
        4. Try re-exporting the matrix from Visum if the issue persists
        """)

else:
    st.info("👈 Upload a travel time matrix (MTX file) to begin")

    # Instructions
    st.markdown("""
    ### 📋 How to use this tool:

    1. **Upload Travel Time Matrix**: Upload your Visum MTX file containing travel times
       - The tool will automatically detect the matrix data section and zone ID mapping
       - Unreachable destinations (999999 or *) will be handled appropriately
       - You can set maximum travel time limits for the analysis

    2. **Add Destination Categories**: Define categories of destinations (e.g., hospitals, clinics)
       - Upload Excel files with 'ID', 'nombre', and optionally 'attractiveness' columns
       - Zone 'ID' should match the zone numbers in your Visum model (e.g., 700, 733, 905)
       - Set lambda (λ) decay parameter for each category (higher = faster decay)
       - Set category weight (w) for overall importance in total accessibility

    3. **Configure Analysis Options**:
       - **Origin Zone Filtering**: Select which origin zones to include (e.g., 1-453 for specific study area)
       - **Maximum travel time**: Destinations beyond this are considered unreachable
       - **Unreachable penalty**: Travel time assigned to unreachable destinations

    4. **Calculate Accessibility**: The tool uses competitive weighting within categories:

       **Step 1: Normalize attractiveness within each category (destinations compete)**
       - Â_j = A_j / Σ A_category  

       **Step 2: Apply decay function**
       - Category_accessibility = Σ (Â_j · e^(-λ·c_ij))

       **Step 3: Apply category weights**
       - Total_Acc_i = w_hospital · Σ (Â_hospital_j · e^(-λ_hospital·c_ij)) + w_clinic · Σ (Â_clinic_j · e^(-λ_clinic·c_ij))

    5. **Download Results**: Export accessibility scores for all origin zones

    ### 📈 Interpretation:
    - **Higher accessibility scores** = better access to destinations
    - **Lambda (λ)** controls distance decay: higher values mean accessibility drops faster with travel time  
    - **Category weight (w)** adjusts the relative importance of each category in total accessibility
    - **Attractiveness (A)** sets individual destination appeal within its category
    - **Competitive weighting** ensures destinations within each category compete proportionally
    - **Normalized values** (0-1 scale): Use min-max normalization for easier comparison across different scales
      - Value of 0 = lowest accessibility in the study area
      - Value of 1 = highest accessibility in the study area
      - Useful for comparing relative accessibility levels and creating standardized maps
    """)