# Calculated skims:
# ACD	Access distance - Distancia de acceso
# ACT	Access time - Tiempo de acceso
# EGD	Egress distance - Distancia de salida
# EGT	Egress time - Tiempo de salida
# JRD	Journey distance - Distancia del viaje
# JRT	Journey time - Tiempo del viaje
# NTR	Number of transfers - Número de transbordos
# RID	Ride distance - Distancia del trayecto
# RIT	Ride time - Tiempo del trayecto
# SFQ	Service frequency - Frecuencia del servicio
# TWT	Transfer wait time - Tiempo de espera para transbordo

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Análisis Mejorado de Accesibilidad del Transporte Público", layout="wide")
st.title("🚌 Análisis Mejorado de Accesibilidad del Transporte Público")

# Upload files
uploaded_skim_file = st.file_uploader("Subir archivo CSV de matriz de tiempos", type=["csv"], key="skim")
uploaded_zones_file = st.file_uploader("Subir archivo CSV de mapeo zona-destino", type=["csv"], key="zones")
pop_file = st.file_uploader("Subir archivo XLSX de Población (opcional)", type=["xlsx"], key="pop")

# ====== NEW: metric catalog ======
ALL_METRICS = ["JRT", "NTR", "RIT", "ACT", "EGT", "ACD", "EGD", "SFQ", "JRD", "RID", "TWT"]  # display/order helpers
TIME_METRICS = {"JRT", "RIT", "ACT", "EGT", "TWT"}  # minutes
DIST_METRICS = {"JRD", "ACD", "EGD", "RID"}  # distance (km or m)
COUNT_METRICS = {"NTR"}  # integer
FREQ_METRICS = {"SFQ"}  # frequency or headway
DEFAULT_COLS = ["OrigZoneNo", "DestZoneNo", "ACD", "ACT", "EGD", "EGT", "JRD", "JRT", "NTR", "RID", "RIT", "SFQ",
                "TWT"] + ["ACT", "EGT", "", "JRD", "JRT", "NTR", "RIT", "SFQ"]

# Spanish metric descriptions
metric_descriptions = {
    "JRT": "Tiempo del viaje (tiempo total puerta a puerta, min)",
    "NTR": "Número de transbordos",
    "RIT": "Tiempo del trayecto (tiempo en vehículo, min)",
    "ACT": "Tiempo de acceso (tiempo de caminata/aproximación, min)",
    "EGT": "Tiempo de salida (tiempo de caminata/bajada, min)",
    "ACD": "Distancia de acceso",
    "EGD": "Distancia de salida",
    "RID": "Distancia del trayecto",
    "SFQ": "Frecuencia del servicio / Intervalo",
    "JRD": "Distancia del viaje (km)",
    "TWT": "Tiempo de espera para transbordo (min)"
}

if uploaded_skim_file:
    st.sidebar.header("Configuración de Matriz de Tiempos")

    skim_type = st.sidebar.selectbox(
        "¿Qué tipo de destinos contiene esta matriz de tiempos?",
        [
            "Hospitales",
            "Consultorio",
            "Centro de Salud",
            "Residencias",
            "Bachilleres",
            "Hacienda",
            "Catastro",
            "Atencion",
            "Servicios Sociales",
            "Agricultura",
            "Comarca (dentro de zonas normales)",
            "Zona Única (como Bilbao, Uni Bilbao)",
            "Rango Personalizado"
        ]
    )

    predefined_ranges = {
        "Hospitales": (900, 904),
        "Consultorio": (500, 768),
        "Centro de Salud": (500, 768),
        "Residencias": (600, 756),
        "Bachilleres": (800, 848),
        "Hacienda": (500, 511),
        "Catastro": (512, 523),
        "Atencion": (524, 575),
        "Servicios Sociales": (584, 591),
        "Agricultura": (576, 583),
        "Comarca (dentro de zonas normales)": (1, 453),
        "Zona Única (como Bilbao, Uni Bilbao)": (1, 453),
        "Rango Personalizado": (500, 600)
    }
    default_start, default_end = predefined_ranges[skim_type]

    st.sidebar.subheader("Configuración de Zonas de Destino")

    if skim_type == "Zona Única (como Bilbao, Uni Bilbao)":
        st.sidebar.info("💡 Para zonas únicas, establezca inicio y fin con el mismo número de zona")
        dest_start = st.sidebar.number_input(
            "ID de Zona de Destino Única",
            value=1,
            help="El ID específico de zona para destinos como Bilbao, Universidad, etc."
        )
        dest_end = dest_start
        st.sidebar.write(f"Se analizará la accesibilidad a la zona: **{dest_start}**")
    elif skim_type == "Comarca (dentro de zonas normales)":
        st.sidebar.info("💡 Las zonas de comarca están dentro del rango de zonas normales (1-453)")
        dest_start = st.sidebar.number_input(
            "Zona Comarca Inicio", value=default_start, min_value=1, max_value=453,
            help="ID de zona inicial para destinos de comarca"
        )
        dest_end = st.sidebar.number_input(
            "Zona Comarca Fin", value=default_end, min_value=dest_start, max_value=453,
            help="ID de zona final para destinos de comarca"
        )
        st.sidebar.write(f"Se analizará la accesibilidad a zonas de comarca: **{dest_start}-{dest_end}**")
    else:
        dest_start = st.sidebar.number_input(
            "Zona de Destino Inicio", value=default_start,
            help="ID de zona inicial para destinos en esta matriz de tiempos"
        )
        dest_end = st.sidebar.number_input(
            "Zona de Destino Fin", value=default_end,
            help="ID de zona final para destinos en esta matriz de tiempos"
        )
        st.sidebar.info(f"Se analizará la accesibilidad a destinos en zonas **{dest_start}-{dest_end}**")

    # ===== NEW: Interpretation for SFQ =====
    st.sidebar.subheader("Interpretación de Frecuencia del Servicio")
    sfq_mode = st.sidebar.selectbox(
        "¿Cómo debe interpretarse SFQ?",
        ["Frecuencia (viajes/hora) — mayor es mejor", "Intervalo (minutos) — menor es mejor"],
        index=0
    )
    # Directions: min (lower is better) / max (higher is better)
    metric_direction = {m: "min" for m in ALL_METRICS}
    if "Frecuencia" in sfq_mode:
        metric_direction["SFQ"] = "max"
    else:
        metric_direction["SFQ"] = "min"

    # Analysis metric selection
    st.sidebar.subheader("Métricas de Análisis")
    primary_metric = st.sidebar.selectbox(
        "Métrica principal para análisis de accesibilidad",
        ALL_METRICS,
        index=ALL_METRICS.index("JRT"),
        help="Elija qué métrica usar para encontrar el 'mejor' destino"
    )
    st.sidebar.info(f"Usando **{primary_metric}**: {metric_descriptions.get(primary_metric, primary_metric)}")

    # Secondary metrics to include in results
    st.sidebar.subheader("Métricas Adicionales a Incluir")
    secondary_metrics = []
    for metric in ALL_METRICS:
        if metric != primary_metric:
            if st.sidebar.checkbox(f"Incluir {metric} ({metric_descriptions[metric]})", value=True):
                secondary_metrics.append(metric)

if uploaded_zones_file:
    zones_df = pd.read_csv(uploaded_zones_file, sep=';')
    st.sidebar.subheader("Archivo de Mapeo de Zonas")
    st.sidebar.write(f"📍 {len(zones_df)} zonas cargadas")

    id_columns = ['id', 'name']
    destination_columns = [col for col in zones_df.columns if col not in id_columns and not zones_df[col].isna().all()]

    with st.sidebar.expander("Categorías de destino disponibles en archivo de mapeo"):
        for col in destination_columns:
            non_null_count = zones_df[col].notna().sum()
            unique_destinations = zones_df[col].nunique()
            st.write(f"• **{col}**: {non_null_count} zonas mapeadas a {unique_destinations} destinos únicos")
            if unique_destinations == 1:
                single_dest = zones_df[col].dropna().iloc[0]
                st.write(f"  ↳ Todas las zonas mapean al destino: **{single_dest}**")

if uploaded_skim_file and uploaded_zones_file:
    # Origin zones configuration (fixed range)
    st.sidebar.subheader("Configuración de Zonas de Origen")
    origin_start = 1
    origin_end = 453
    st.sidebar.info(f"Zonas de origen: {origin_start}-{origin_end} (fijo)")
    normal_zones = range(origin_start, origin_end + 1)

    # Analysis mode selection
    st.sidebar.subheader("Modo de Análisis")
    analysis_mode = st.sidebar.radio(
        "¿Cómo quiere encontrar los destinos?",
        [
            "Más cercano dentro del rango",
            "Usar mapeo de zonas (si está disponible)",
            "Ambos métodos",
            "Mapeo de zonas con respaldo al más cercano"
        ],
        help="Elija cómo determinar qué destinos analizar para cada zona de origen"
    )

    mapping_column = None
    if analysis_mode in ["Usar mapeo de zonas (si está disponible)", "Ambos métodos", "Mapeo de zonas con respaldo al más cercano"]:
        mapping_column = st.sidebar.selectbox(
            "Seleccionar columna de mapeo de destinos",
            [None] + destination_columns,
            help="Elija qué columna contiene las asignaciones de destinos"
        )

        if mapping_column:
            mapped_zones = zones_df[mapping_column].notna().sum()
            unique_destinations = zones_df[mapping_column].nunique()
            st.sidebar.success(f"✅ {mapped_zones} zonas tienen destinos mapeados en '{mapping_column}'")
            st.sidebar.info(f"📊 Mapea a {unique_destinations} destinos únicos")

            if analysis_mode == "Mapeo de zonas con respaldo al más cercano":
                unmapped_zones = len(zones_df) - mapped_zones
                st.sidebar.info(f"🔄 {unmapped_zones} zonas usarán respaldo al más cercano")

    # Process the analysis
    if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
        st.subheader("Procesando Matriz de Tiempos...")


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
            st.metric("Registros Totales", f"{len(skim_df):,}")
        with col2:
            st.metric("Zonas de Origen", f"{skim_df['OrigZoneNo'].nunique()}")
        with col3:
            st.metric("Zonas de Destino", f"{skim_df['DestZoneNo'].nunique()}")
        with col4:
            available_metrics = [col for col in ALL_METRICS if col in skim_df.columns]
            st.metric("Métricas Disponibles", len(available_metrics))

        with st.expander("📊 Datos de Muestra de Matriz de Tiempos"):
            sample_cols = ['OrigZoneNo', 'DestZoneNo'] + available_metrics
            st.dataframe(skim_df[sample_cols].head(10), use_container_width=True)

        if skim_df.empty:
            st.error("⚠️ ¡No se encontraron datos en la matriz de tiempos para el rango de destinos especificado!")
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
        if analysis_mode in ["Más cercano dentro del rango", "Ambos métodos"]:
            st.subheader(
                f"🔍 Análisis de {skim_type} Más Cercano (por {primary_metric} / {'maximizar' if metric_direction[primary_metric] == 'max' else 'minimizar'})")

            best_destinations = []
            for orig_zone in results['OrigZoneNo']:
                zone_data = skim_df[skim_df['OrigZoneNo'] == orig_zone]
                best_row = find_best_destination(zone_data, primary_metric)

                result_row = {'OrigZoneNo': orig_zone}
                if best_row is not None:
                    result_row[f'Zona_Cercana'] = int(best_row['DestZoneNo'])
                    result_row[f'Cercana_{primary_metric}'] = best_row[primary_metric]
                    for metric in secondary_metrics:
                        if metric in best_row:
                            result_row[f'Cercana_{metric}'] = best_row[metric]
                else:
                    result_row[f'Zona_Cercana'] = np.nan
                    result_row[f'Cercana_{primary_metric}'] = np.nan
                    for metric in secondary_metrics:
                        result_row[f'Cercana_{metric}'] = np.nan
                best_destinations.append(result_row)

            if best_destinations:
                nearest_df = pd.DataFrame(best_destinations)
                results = results.merge(nearest_df, on='OrigZoneNo', how='left')

                valid_vals = results[f'Cercana_{primary_metric}'].dropna()
                if len(valid_vals) > 0:
                    cols = st.columns(len(secondary_metrics) + 4)
                    with cols[0]:
                        st.metric("Zonas con Acceso", f"{len(valid_vals)}")
                    with cols[1]:
                        st.metric(f"Prom {primary_metric}", f"{valid_vals.mean():.2f}")
                    with cols[2]:
                        st.metric(f"Mín {primary_metric}", f"{valid_vals.min():.2f}")
                    with cols[3]:
                        st.metric(f"Máx {primary_metric}", f"{valid_vals.max():.2f}")
                    for i, metric in enumerate(secondary_metrics, 4):
                        colname = f'Cercana_{metric}'
                        if colname in results.columns:
                            vv = results[colname].dropna()
                            if len(vv) > 0 and i < len(cols):
                                with cols[i]:
                                    st.metric(f"Prom {metric}", f"{vv.mean():.2f}")

        # Method 2: Use zone mapping
        if analysis_mode in ["Usar mapeo de zonas (si está disponible)", "Ambos métodos"] and mapping_column:
            st.subheader(f"🗺️ Análisis de Mapeo de Zonas ({mapping_column})")

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

                result_row = {'OrigZoneNo': orig_zone, 'Zona_Mapeada': dest_zone}
                if not travel_row.empty:
                    result_row[f'Mapeada_{primary_metric}'] = travel_row[primary_metric].iloc[0]
                    for metric in secondary_metrics:
                        if metric in travel_row.columns:
                            result_row[f'Mapeada_{metric}'] = travel_row[metric].iloc[0]
                else:
                    result_row[f'Mapeada_{primary_metric}'] = np.nan
                    for metric in secondary_metrics:
                        result_row[f'Mapeada_{metric}'] = np.nan
                mapping_results.append(result_row)

            if mapping_results:
                mapping_df = pd.DataFrame(mapping_results)
                results = results.merge(mapping_df, on='OrigZoneNo', how='left')

                st.info(f"📊 **Resultados de Análisis de Mapeo:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Destinos en Rango", destinations_in_range)
                with col2:
                    st.metric("Destinos Fuera de Rango", destinations_outside_range)
                with col3:
                    valid_mapped = results[f'Mapeada_{primary_metric}'].dropna()
                    st.metric("Mapeos Exitosos", len(valid_mapped))

                if len(valid_mapped) > 0:
                    st.write(
                        f"**Estadísticas de {primary_metric} Mapeado:** Prom: {valid_mapped.mean():.2f}, Mín: {valid_mapped.min():.2f}, Máx: {valid_mapped.max():.2f}")
                    for metric in secondary_metrics:
                        coln = f'Mapeada_{metric}'
                        if coln in results.columns:
                            vv = results[coln].dropna()
                            if len(vv) > 0:
                                st.write(
                                    f"**Estadísticas de {metric}:** Prom: {vv.mean():.2f}, Mín: {vv.min():.2f}, Máx: {vv.max():.2f}")

        # Method 4: Zone mapping with fallback to nearest
        if analysis_mode == "Mapeo de zonas con respaldo al más cercano" and mapping_column:
            st.subheader(f"🔄 Análisis de Mapeo de Zonas con Respaldo ({mapping_column})")

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
                                'Zona_Respaldo': fb['dest_zone'],
                                f'Respaldo_{primary_metric}': fb[primary_metric],
                                'Metodo_Usado': 'Más cercano (dest mapeado fuera de rango)'
                            }
                            for metric in secondary_metrics:
                                if metric in fb:
                                    result_row[f'Respaldo_{metric}'] = fb[metric]
                            fallback_results.append(result_row)
                            zones_used_fallback += 1
                        else:
                            result_row = {'OrigZoneNo': orig_zone, 'Zona_Respaldo': np.nan,
                                          f'Respaldo_{primary_metric}': np.nan, 'Metodo_Usado': 'Sin acceso'}
                            for metric in secondary_metrics:
                                result_row[f'Respaldo_{metric}'] = np.nan
                            fallback_results.append(result_row)
                            zones_no_access += 1
                        continue

                    travel_row = skim_df[(skim_df['OrigZoneNo'] == orig_zone) & (skim_df['DestZoneNo'] == dest_zone)]
                    if not travel_row.empty:
                        result_row = {
                            'OrigZoneNo': orig_zone,
                            'Zona_Respaldo': dest_zone,
                            f'Respaldo_{primary_metric}': travel_row[primary_metric].iloc[0],
                            'Metodo_Usado': 'Mapeo de zonas'
                        }
                        for metric in secondary_metrics:
                            if metric in travel_row.columns:
                                result_row[f'Respaldo_{metric}'] = travel_row[metric].iloc[0]
                        fallback_results.append(result_row)
                        zones_used_mapping += 1
                    else:
                        if orig_zone in nearest_fallback:
                            fb = nearest_fallback[orig_zone]
                            result_row = {
                                'OrigZoneNo': orig_zone,
                                'Zona_Respaldo': fb['dest_zone'],
                                f'Respaldo_{primary_metric}': fb[primary_metric],
                                'Metodo_Usado': 'Más cercano (mapeo falló)'
                            }
                            for metric in secondary_metrics:
                                if metric in fb:
                                    result_row[f'Respaldo_{metric}'] = fb[metric]
                            fallback_results.append(result_row)
                            zones_used_fallback += 1
                        else:
                            result_row = {'OrigZoneNo': orig_zone, 'Zona_Respaldo': np.nan,
                                          f'Respaldo_{primary_metric}': np.nan, 'Metodo_Usado': 'Sin acceso'}
                            for metric in secondary_metrics:
                                result_row[f'Respaldo_{metric}'] = np.nan
                            fallback_results.append(result_row)
                            zones_no_access += 1
                else:
                    if orig_zone in nearest_fallback:
                        fb = nearest_fallback[orig_zone]
                        result_row = {
                            'OrigZoneNo': orig_zone,
                            'Zona_Respaldo': fb['dest_zone'],
                            f'Respaldo_{primary_metric}': fb[primary_metric],
                            'Metodo_Usado': 'Más cercano (sin mapeo)'
                        }
                        for metric in secondary_metrics:
                            if metric in fb:
                                result_row[f'Respaldo_{metric}'] = fb[metric]
                        fallback_results.append(result_row)
                        zones_used_fallback += 1
                    else:
                        result_row = {'OrigZoneNo': orig_zone, 'Zona_Respaldo': np.nan,
                                      f'Respaldo_{primary_metric}': np.nan, 'Metodo_Usado': 'Sin acceso'}
                        for metric in secondary_metrics:
                            result_row[f'Respaldo_{metric}'] = np.nan
                        fallback_results.append(result_row)
                        zones_no_access += 1

            if fallback_results:
                fallback_df = pd.DataFrame(fallback_results)
                results = results.merge(fallback_df, on='OrigZoneNo', how='left')

                st.info(f"🔄 **Resultados de Análisis con Respaldo:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Usó Mapeo de Zonas", zones_used_mapping)
                with col2:
                    st.metric("Usó Respaldo Más Cercano", zones_used_fallback)
                with col3:
                    st.metric("Sin Acceso", zones_no_access)
                with col4:
                    if destinations_outside_range > 0:
                        st.metric("Dest. Fuera de Rango", destinations_outside_range)

                valid_fb = results[f'Respaldo_{primary_metric}'].dropna()
                if len(valid_fb) > 0:
                    st.write(
                        f"**Estadísticas Combinadas de {primary_metric}:** Prom: {valid_fb.mean():.2f}, Mín: {valid_fb.min():.2f}, Máx: {valid_fb.max():.2f}")
                    for metric in secondary_metrics:
                        coln = f'Respaldo_{metric}'
                        if coln in results.columns:
                            vv = results[coln].dropna()
                            if len(vv) > 0:
                                st.write(
                                    f"**Estadísticas de {metric}:** Prom: {vv.mean():.2f}, Mín: {vv.min():.2f}, Máx: {vv.max():.2f}")

                if 'Metodo_Usado' in results.columns:
                    method_counts = results['Metodo_Usado'].value_counts()
                    if not method_counts.empty:
                        st.write("**Desglose de Uso de Métodos:**")
                        for method, count in method_counts.items():
                            percentage = (count / len(results)) * 100
                            st.write(f"• {method}: {count} zonas ({percentage:.1f}%)")

        # Population (optional)
        if pop_file:
            try:
                pop_df = pd.read_excel(pop_file)
                if 'OrigZoneNo' in pop_df.columns and 'Population' in pop_df.columns:
                    pop_df = pop_df[['OrigZoneNo', 'Population']]
                    results = results.merge(pop_df, on='OrigZoneNo', how='left')
                    results['Population'] = results['Population'].fillna(0).astype(int)
                    st.success(f"✅ Datos de población agregados para {len(pop_df)} zonas")
                else:
                    st.error("⚠️ El archivo de población debe tener columnas: 'OrigZoneNo' y 'Population'")
            except Exception as e:
                st.error(f"⚠️ Error al cargar archivo de población: {str(e)}")

        # Display results
        st.subheader("🎯 Resultados de Accesibilidad")

        display_columns = ['OrigZoneNo', 'ZoneName']
        if 'Population' in results.columns:
            display_columns.append('Population')

        metric_columns, zone_columns, method_columns = [], [], []
        for col in results.columns:
            if any(col.endswith(f'_{m}') for m in ALL_METRICS):
                metric_columns.append(col)
            elif col.endswith('_Zone') or 'Zona_' in col:
                zone_columns.append(col)
            elif col.endswith('_Used') or 'Metodo_' in col:
                method_columns.append(col)

        primary_cols = [col for col in metric_columns if col.endswith(f'_{primary_metric}')]
        secondary_cols = [col for col in metric_columns if col not in primary_cols]

        display_columns.extend(primary_cols + secondary_cols + zone_columns + method_columns)
        display_columns = [col for col in display_columns if col in results.columns]

        st.dataframe(results[display_columns], use_container_width=True)

        # Download results
        csv_download = results.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resultados como CSV",
            data=csv_download,
            file_name=f"accesibilidad_{skim_type.lower().replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_resultados.csv",
            mime="text/csv"
        )

        # ===== Population accessibility summary with thresholds =====
        if 'Population' in results.columns:
            st.subheader("👥 Resumen de Accesibilidad Poblacional")


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
                f"Umbrales para {primary_metric} (separados por comas)",
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
                            "Método": method_name,
                            f"Umbral ({primary_metric})": thr,
                            "Población Accesible": int(accessible_pop),
                            "Población Total": int(total_pop),
                            "Población (%)": round(pop_pct, 2),
                            "Zonas Accesibles": accessible_zones,
                            "Zonas Totales": total_zones,
                            "Zonas (%)": round(zone_pct, 2)
                        })

                    # unreachable info
                    unreachable_mask = results[coln].isna()
                    unreachable_pop = int(results.loc[unreachable_mask, 'Population'].sum())
                    unreachable_zones = int(unreachable_mask.sum())
                    if unreachable_zones > 0:
                        st.warning(
                            f"🚫 **{method_name}**: {unreachable_zones} zonas "
                            f"({unreachable_pop:,} personas) no tienen destino alcanzable"
                        )

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(summary_df, use_container_width=True)

                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Resumen como CSV",
                        data=csv_summary,
                        file_name=f"resumen_accesibilidad_{skim_type.lower().replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                        mime="text/csv"
                    )

                    st.subheader("📊 Visualización de Accesibilidad")
                    st.write(f"**Accesibilidad Poblacional por {primary_metric}**")
                    if len(primary_metric_columns) > 1:
                        pivot_df_pop = summary_df.pivot(index=f'Umbral ({primary_metric})', columns='Método',
                                                        values='Población (%)')
                        st.bar_chart(pivot_df_pop)
                    else:
                        chart_data_pop = summary_df.set_index(f'Umbral ({primary_metric})')['Población (%)']
                        st.bar_chart(chart_data_pop)

                    st.write(f"**Accesibilidad de Zonas por {primary_metric}**")
                    if len(primary_metric_columns) > 1:
                        pivot_df_zones = summary_df.pivot(index=f'Umbral ({primary_metric})', columns='Método',
                                                          values='Zonas (%)')
                        st.bar_chart(pivot_df_zones)
                    else:
                        chart_data_zones = summary_df.set_index(f'Umbral ({primary_metric})')['Zonas (%)']
                        st.bar_chart(chart_data_zones)

            except ValueError:
                st.error("⚠️ Formato de umbral inválido. Use números separados por comas como: 15,30,45,60")

        # ===== Multi-metric comparison =====
        if (secondary_metrics and any(col in results.columns for col in metric_columns)) or True:
            st.subheader("📈 Análisis Multi-Métrica")
            st.write("**Correlación entre métricas:**")

            available_metric_cols = [col for col in metric_columns if col in results.columns]
            if len(available_metric_cols) >= 2:
                metric_data = results[available_metric_cols].select_dtypes(include=[np.number])
                if not metric_data.empty:
                    correlation_matrix = metric_data.corr()
                    st.write("Matriz de correlación entre métricas de accesibilidad:")
                    st.dataframe(correlation_matrix.round(3), use_container_width=True)

                    with st.expander("💡 Información de Métricas"):
                        for col in available_metric_cols:
                            if col in results.columns:
                                valid_data = results[col].dropna()
                                if len(valid_data) > 0:
                                    col_name = col.replace('_', ' ').title()
                                    st.write(f"**{col_name}:**")
                                    st.write(f"  • Media: {valid_data.mean():.2f}")
                                    st.write(f"  • Mediana: {valid_data.median():.2f}")
                                    st.write(f"  • Desv. Estándar: {valid_data.std():.2f}")
                                    st.write(f"  • Rango: {valid_data.min():.2f} - {valid_data.max():.2f}")
                                    st.write("")

else:
    st.info("👆 Por favor suba ambos archivos para comenzar el análisis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 Archivo de Matriz de Tiempos
        - **Formato CSV** con múltiples métricas de viaje
        - **Columnas requeridas**: `OrigZoneNo, DestZoneNo, ACD, ACT, EGD, EGT, JRD, JRT, NTR, RID, RIT, SFQ, TWT`
        - **ACD**: Distancia de acceso  
        - **ACT**: Tiempo de acceso (min)  
        - **EGD**: Distancia de salida  
        - **EGT**: Tiempo de salida (min)  
        - **JRD**: Distancia del viaje (km)  
        - **JRT**: Tiempo del viaje (min)  
        - **NTR**: Número de transbordos  
        - **RID**: Distancia del trayecto  
        - **RIT**: Tiempo del trayecto (min)  
        - **TWT**: Tiempo de espera para transbordo (min)  
        - **SFQ**: Frecuencia del servicio **(viajes/hora)** o **Intervalo (min)** — elija interpretación en la barra lateral

        **Ejemplo:**
        ```
OrigZoneNo,DestZoneNo,ACD,ACT,EGD,EGT,JRD,JRT,NTR,RID,RIT,SFQ,TWT
1,501,0.5,3.2,0.6,4.1,7.5,28.5,1,6.9,18.0,6,2.0
1,502,0.8,5.0,0.4,3.0,9.0,34.0,2,8.6,22.5,4,3.0
2,501,0.4,2.8,0.3,2.5,5.0,18.0,0,4.6,12.5,10,1.5
```
        """)

    with col2:
        st.markdown("""
        ### 🗺️ Archivo de Mapeo de Zonas  
        - **CSV con separador punto y coma**
        - **Formato**: `id;name;hospital_id;osakidetza_id;comarca_id;bilbao_id;...`
        - Mapea cada zona a destinos específicos

        **Nuevas Características:**
        - ✅ **Métricas ampliadas**: ACT, EGT, JRD, JRT, NTR, RIT, SFQ
        - ✅ **Optimización de métrica principal** con lógica min/max basada en tipo de métrica (ej. SFQ)
        - ✅ **Resúmenes de umbral basados en población** que soportan ≤ o ≥ dependiendo de la métrica
        - ✅ **Comparaciones visuales y correlaciones**
        """)

# Footer info
with st.expander("ℹ️ Cómo usar esta herramienta mejorada"):
    st.markdown("""
    ## Optimización de Métricas
    - La mayoría de métricas se minimizan (menor es mejor): **JRT, RIT, ACT, EGT, JRD, NTR**
    - **SFQ** puede interpretarse como:
      - **Frecuencia (viajes/hora)** → **maximizar**
      - **Intervalo (minutos)** → **minimizar**
      Configúrelo en la barra lateral.

    ## Flujo de Trabajo
    1. Subir matriz de tiempos (con nuevas columnas)
    2. Elegir rango/tipo de destino
    3. Establecer **interpretación de SFQ**
    4. Seleccionar métricas **principales** y **secundarias**
    5. Elegir modo de análisis (más cercano, mapeo, respaldo)
    6. (Opcional) Agregar población y ejecutar resúmenes/gráficos
    """)

with st.expander("📖 Entendiendo las Métricas"):
    st.markdown("""
    **JRT** Tiempo total del viaje (min) • **NTR** Transbordos (cantidad) • **RIT** Tiempo en vehículo (min)  
    **ACT/EGT** Tiempo de caminata acceso/salida (min) • **JRD** Distancia (km)  
    **SFQ** Servicio ofrecido: frecuencia (viajes/h, mayor mejor) o intervalo (min, menor mejor)
    """)