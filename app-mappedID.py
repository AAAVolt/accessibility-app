# Aplicación Simplificada de Análisis de Accesibilidad
# Enfocada en asignación de zonas a destinos desde archivo CSV

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Análisis de Accesibilidad - Asignación Zona-Destino", layout="wide")
st.title("🚌 Análisis de Accesibilidad del Transporte Público - Asignación Zona-Destino")

# Upload files
uploaded_skim_file = st.file_uploader("Subir archivo CSV de matriz de tiempos", type=["csv"], key="skim")
zone_assignment_file = st.file_uploader("Subir archivo Excel con asignación zona-destino",
                                        type=["xlsx"], key="zone_assignment",
                                        help="Archivo Excel con columnas 'ZONE_ID' y 'DESTINATION_ID' para asignar cada zona a un destino específico")
pop_file = st.file_uploader("Subir archivo XLSX de Población (opcional)", type=["xlsx"], key="pop")

# Metric catalog
ALL_METRICS = ["JRT", "NTR", "RIT", "ACT", "EGT", "ACD", "EGD", "SFQ", "JRD", "RID", "TWT"]
TIME_METRICS = {"JRT", "RIT", "ACT", "EGT", "TWT"}
DIST_METRICS = {"JRD", "ACD", "EGD", "RID"}
COUNT_METRICS = {"NTR"}
FREQ_METRICS = {"SFQ"}
DEFAULT_COLS = ["OrigZoneNo", "DestZoneNo", "ACD", "ACT", "EGD", "EGT", "JRD", "JRT", "NTR", "RID", "RIT", "SFQ", "TWT"]

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

# Check if zone assignment file is uploaded and display info
zone_assignment_dict = None
destination_ids = None

if zone_assignment_file:
    try:
        # Read Excel file
        assignment_df = pd.read_excel(zone_assignment_file)

        # Try to find the required columns (flexible column names)
        zone_column = None
        dest_column = None

        # Possible column names for zone ID
        possible_zone_cols = ['ZONE_ID', 'zone_id', 'Zone_ID', 'ZoneID', 'zona_id', 'OrigZoneNo', 'origin_id', 'ZONE',
                              'Zone']
        for col in possible_zone_cols:
            if col in assignment_df.columns:
                zone_column = col
                break

        # Possible column names for destination ID
        possible_dest_cols = ['DESTINATION_ID', 'destination_id', 'Destination_ID', 'DestinationID', 'destino_id',
                              'DestZoneNo', 'dest_id', 'DESTINATION', 'Destination']
        for col in possible_dest_cols:
            if col in assignment_df.columns:
                dest_column = col
                break

        if zone_column is None or dest_column is None:
            # If standard columns not found, try to use first two columns
            if len(assignment_df.columns) >= 2:
                zone_column = assignment_df.columns[0]
                dest_column = assignment_df.columns[1]
                st.warning(f"⚠️ Usando '{zone_column}' como ZONE_ID y '{dest_column}' como DESTINATION_ID.")
            else:
                raise ValueError("El archivo debe tener al menos 2 columnas")

        # Clean and convert to numeric
        assignment_df[zone_column] = pd.to_numeric(assignment_df[zone_column], errors='coerce')
        assignment_df[dest_column] = pd.to_numeric(assignment_df[dest_column], errors='coerce')
        assignment_df = assignment_df.dropna(subset=[zone_column, dest_column])

        # Create assignment dictionary
        zone_assignment_dict = dict(zip(assignment_df[zone_column].astype(int), assignment_df[dest_column].astype(int)))
        destination_ids = sorted(set(assignment_df[dest_column].astype(int)))

        st.sidebar.header("📋 Archivo de Asignación Zona-Destino")
        st.sidebar.success(f"✅ {len(zone_assignment_dict)} asignaciones cargadas")
        st.sidebar.write(f"**Zonas asignadas:** {len(zone_assignment_dict)}")
        st.sidebar.write(f"**Destinos únicos:** {len(destination_ids)}")
        st.sidebar.write(f"**Rango de destinos:** {min(destination_ids)} - {max(destination_ids)}")

        with st.sidebar.expander("👁️ Ver asignaciones (muestra)"):
            # Show first 10 assignments
            sample_assignments = list(zone_assignment_dict.items())[:10]
            for zone, dest in sample_assignments:
                st.write(f"Zona {zone} → Destino {dest}")
            if len(zone_assignment_dict) > 10:
                st.write(f"... (+{len(zone_assignment_dict) - 10} más)")

    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar archivo de asignaciones: {str(e)}")
        zone_assignment_dict = None

if uploaded_skim_file and zone_assignment_dict:
    st.sidebar.header("Configuración de Análisis")

    # SFQ interpretation
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
    selected_metrics = []
    st.sidebar.write("Seleccionar métricas para incluir en el análisis:")

    for metric in ALL_METRICS:
        if st.sidebar.checkbox(f"{metric} ({metric_descriptions[metric]})", value=(metric == "JRT")):
            selected_metrics.append(metric)

    if not selected_metrics:
        st.sidebar.error("⚠️ Debe seleccionar al menos una métrica")
        st.stop()

    # Origin zones configuration (fixed range)
    st.sidebar.subheader("Configuración de Zonas de Origen")
    origin_start = 1
    origin_end = 453
    st.sidebar.info(f"Zonas de origen: {origin_start}-{origin_end} (fijo)")
    normal_zones = range(origin_start, origin_end + 1)

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

        # Filter to normal zones for origins
        skim_df = skim_df[skim_df['OrigZoneNo'].isin(normal_zones)]

        # Filter to only destination IDs that appear in assignments
        skim_df = skim_df[skim_df['DestZoneNo'].isin(destination_ids)]

        # Display skim matrix info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros Totales", f"{len(skim_df):,}")
        with col2:
            st.metric("Zonas de Origen", f"{skim_df['OrigZoneNo'].nunique()}")
        with col3:
            st.metric("Destinos en Matriz", f"{skim_df['DestZoneNo'].nunique()}")
        with col4:
            available_metrics = [col for col in ALL_METRICS if col in skim_df.columns]
            st.metric("Métricas Disponibles", len(available_metrics))

        st.info(f"🔄 Analizando accesibilidad desde núcleos (1-453) hacia destinos asignados")

        with st.expander("📊 Datos de Muestra de Matriz de Tiempos"):
            sample_cols = ['OrigZoneNo', 'DestZoneNo'] + available_metrics
            st.dataframe(skim_df[sample_cols].head(10), use_container_width=True)

        if skim_df.empty:
            st.error("⚠️ ¡No se encontraron datos en la matriz de tiempos para los destinos asignados!")
            st.stop()

        # Initialize results
        results = []

        # Create zone assignment analysis
        st.subheader("🎯 Análisis de Asignación Zona-Destino")

        for orig_zone in sorted(normal_zones):
            result_row = {'ZONE_ID': orig_zone}

            # Check if this zone has an assigned destination
            if orig_zone in zone_assignment_dict:
                assigned_dest = zone_assignment_dict[orig_zone]
                result_row['DESTINATION_ID'] = assigned_dest

                # Find the travel data for this specific origin-destination pair
                travel_data = skim_df[
                    (skim_df['OrigZoneNo'] == orig_zone) &
                    (skim_df['DestZoneNo'] == assigned_dest)
                    ]

                if not travel_data.empty:
                    # Get the first (and should be only) row
                    travel_row = travel_data.iloc[0]

                    # Add selected metrics to results
                    for metric in selected_metrics:
                        if metric in travel_row and metric in skim_df.columns:
                            result_row[metric] = travel_row[metric]
                        else:
                            result_row[metric] = np.nan

                    result_row['Data_Available'] = True
                else:
                    # No travel data available for this O-D pair
                    for metric in selected_metrics:
                        result_row[metric] = np.nan
                    result_row['Data_Available'] = False
            else:
                # Zone has no assigned destination
                result_row['DESTINATION_ID'] = np.nan
                for metric in selected_metrics:
                    result_row[metric] = np.nan
                result_row['Data_Available'] = False

            results.append(result_row)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate summary statistics
        assigned_zones = results_df['DESTINATION_ID'].notna().sum()
        zones_with_data = results_df['Data_Available'].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Zonas", len(results_df))
        with col2:
            st.metric("Zonas Asignadas", assigned_zones)
        with col3:
            st.metric("Con Datos de Viaje", zones_with_data)
        with col4:
            coverage_pct = (zones_with_data / len(results_df)) * 100 if len(results_df) > 0 else 0
            st.metric("Cobertura (%)", f"{coverage_pct:.1f}%")

        # Show metric statistics
        if selected_metrics:
            st.subheader("📈 Estadísticas de Métricas")

            metric_stats = []
            for metric in selected_metrics:
                if metric in results_df.columns:
                    valid_data = results_df[metric].dropna()
                    if len(valid_data) > 0:
                        metric_stats.append({
                            'Métrica': metric,
                            'Descripción': metric_descriptions.get(metric, metric),
                            'Zonas con Datos': len(valid_data),
                            'Promedio': round(valid_data.mean(), 2),
                            'Mediana': round(valid_data.median(), 2),
                            'Mínimo': round(valid_data.min(), 2),
                            'Máximo': round(valid_data.max(), 2),
                            'Desv. Estándar': round(valid_data.std(), 2)
                        })

            if metric_stats:
                stats_df = pd.DataFrame(metric_stats)
                st.dataframe(stats_df, use_container_width=True)

        # Population (optional)
        if pop_file:
            try:
                pop_df = pd.read_excel(pop_file)
                pop_col_names = ['Population', 'population', 'POPULATION', 'Poblacion', 'poblacion']
                zone_col_names = ['OrigZoneNo', 'ZONE_ID', 'Zone_ID', 'ZoneID', 'zone_id']

                pop_col = None
                zone_col = None

                for col in pop_col_names:
                    if col in pop_df.columns:
                        pop_col = col
                        break

                for col in zone_col_names:
                    if col in pop_df.columns:
                        zone_col = col
                        break

                if pop_col and zone_col:
                    pop_df = pop_df[[zone_col, pop_col]].rename(columns={zone_col: 'ZONE_ID', pop_col: 'Population'})
                    results_df = results_df.merge(pop_df, on='ZONE_ID', how='left')
                    results_df['Population'] = results_df['Population'].fillna(0).astype(int)
                    st.success(f"✅ Datos de población agregados para {len(pop_df)} zonas")
                else:
                    st.error("⚠️ El archivo de población debe tener columnas de zona y población")
            except Exception as e:
                st.error(f"⚠️ Error al cargar archivo de población: {str(e)}")

        # Display results
        st.subheader("📋 Resultados de Asignación Zona-Destino")

        # Prepare display columns
        display_columns = ['ZONE_ID', 'DESTINATION_ID']
        if 'Population' in results_df.columns:
            display_columns.append('Population')

        display_columns.extend(selected_metrics)
        display_columns.append('Data_Available')

        # Filter columns that actually exist
        display_columns = [col for col in display_columns if col in results_df.columns]

        st.dataframe(results_df[display_columns], use_container_width=True)

        # Download results
        csv_download = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resultados como CSV",
            data=csv_download,
            file_name="asignacion_zona_destino_resultados.csv",
            mime="text/csv"
        )

        # Population accessibility summary with thresholds
        if 'Population' in results_df.columns and selected_metrics:
            st.subheader("👥 Resumen de Accesibilidad Poblacional")

            primary_metric = st.selectbox(
                "Seleccionar métrica para análisis de umbrales:",
                selected_metrics,
                index=0
            )


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
                total_pop = results_df['Population'].sum()

                if primary_metric in results_df.columns:
                    for thr in thresholds:
                        if metric_direction.get(primary_metric, "min") == "max":
                            accessible_mask = results_df[primary_metric] >= thr
                        else:
                            accessible_mask = results_df[primary_metric] <= thr

                        accessible_pop = results_df.loc[accessible_mask, 'Population'].sum()
                        accessible_zones = int(accessible_mask.sum())
                        total_zones = len(results_df)
                        pop_pct = 100 * accessible_pop / total_pop if total_pop > 0 else 0
                        zone_pct = 100 * accessible_zones / total_zones if total_zones > 0 else 0

                        summary_rows.append({
                            "Método": "Asignación Zona-Destino",
                            f"Umbral ({primary_metric})": thr,
                            "Población Accesible": int(accessible_pop),
                            "Población Total": int(total_pop),
                            "Población (%)": round(pop_pct, 2),
                            "Zonas Accesibles": accessible_zones,
                            "Zonas Totales": total_zones,
                            "Zonas (%)": round(zone_pct, 2)
                        })

                    # unreachable info
                    unreachable_mask = results_df[primary_metric].isna()
                    unreachable_pop = int(results_df.loc[unreachable_mask, 'Population'].sum())
                    unreachable_zones = int(unreachable_mask.sum())
                    if unreachable_zones > 0:
                        st.warning(
                            f"🚫 **Asignación**: {unreachable_zones} zonas ({unreachable_pop:,} personas) sin datos de accesibilidad")

                    if summary_rows:
                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, use_container_width=True)

                        csv_summary = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resumen como CSV",
                            data=csv_summary,
                            file_name="resumen_accesibilidad_asignacion.csv",
                            mime="text/csv"
                        )

                        st.subheader("📊 Visualización de Accesibilidad")
                        st.write(f"**Accesibilidad Poblacional por {primary_metric}**")
                        chart_data_pop = summary_df.set_index(f'Umbral ({primary_metric})')['Población (%)']
                        st.bar_chart(chart_data_pop)

                        st.write(f"**Accesibilidad de Zonas por {primary_metric}**")
                        chart_data_zones = summary_df.set_index(f'Umbral ({primary_metric})')['Zonas (%)']
                        st.bar_chart(chart_data_zones)

            except ValueError:
                st.error("⚠️ Formato de umbral inválido. Use números separados por comas como: 15,30,45,60")

        # Multi-metric comparison
        if len(selected_metrics) >= 2:
            st.subheader("📈 Análisis Multi-Métrica")
            st.write("**Correlación entre métricas:**")

            metric_data = results_df[selected_metrics].select_dtypes(include=[np.number])
            if not metric_data.empty:
                correlation_matrix = metric_data.corr()
                st.write("Matriz de correlación entre métricas de accesibilidad:")
                st.dataframe(correlation_matrix.round(3), use_container_width=True)

                with st.expander("💡 Información de Métricas"):
                    for metric in selected_metrics:
                        if metric in results_df.columns:
                            valid_data = results_df[metric].dropna()
                            if len(valid_data) > 0:
                                st.write(f"**{metric} - {metric_descriptions.get(metric, metric)}:**")
                                st.write(f"  • Media: {valid_data.mean():.2f}")
                                st.write(f"  • Mediana: {valid_data.median():.2f}")
                                st.write(f"  • Desv. Estándar: {valid_data.std():.2f}")
                                st.write(f"  • Rango: {valid_data.min():.2f} - {valid_data.max():.2f}")
                                st.write("")

else:
    st.info("👆 Por favor suba la matriz de tiempos y el archivo de asignación zona-destino para comenzar el análisis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 Archivo de Matriz de Tiempos
        - **Formato CSV** con múltiples métricas de viaje
        - **Columnas requeridas**: `OrigZoneNo, DestZoneNo, ACD, ACT, EGD, EGT, JRD, JRT, NTR, RID, RIT, SFQ, TWT`

        **Ejemplo:**
        ```
OrigZoneNo,DestZoneNo,ACD,ACT,EGD,EGT,JRD,JRT,NTR,RID,RIT,SFQ,TWT
1,713,0.5,3.2,0.6,4.1,7.5,28.5,1,6.9,18.0,6,2.0
1,714,0.8,5.0,0.4,3.0,9.0,34.0,2,8.6,22.5,4,3.0
2,713,0.4,2.8,0.3,2.5,5.0,18.0,0,4.6,12.5,10,1.5
```
        """)

    with col2:
        st.markdown("""
        ### 🎯 Archivo de Asignación Zona-Destino
        - **Excel con columnas 'ZONE_ID' y 'DESTINATION_ID'**
        - Cada zona se asigna a un destino específico
        - **Ejemplo en Excel:**
        ```
        ZONE_ID | DESTINATION_ID
        1       | 713
        2       | 714
        3       | 713
        4       | 715
        5       | 714
        ```

        **Características de esta versión de asignación:**
        - ✅ **Asignación específica**: Cada zona tiene un destino asignado
        - ✅ **Flexibilidad total**: Cualquier zona puede asignarse a cualquier destino
        - ✅ **Análisis directo**: Obtiene métricas exactas para cada par zona-destino
        - ✅ **Múltiples métricas**: Soporte completo para todas las métricas de transporte
        - ✅ **Análisis poblacional** con umbrales adaptativos
        - ✅ **Descarga de resultados** en formato CSV
        """)

# Footer info
with st.expander("ℹ️ Cómo usar esta herramienta de asignación"):
    st.markdown("""
    ## Flujo de Trabajo de Asignación Zona-Destino

    ### 1. Preparar Archivos
    - **Matriz de tiempos**: CSV con datos de viaje entre zonas
    - **Asignación zona-destino**: Excel con columnas 'ZONE_ID' y 'DESTINATION_ID'
    - **Población** (opcional): XLSX con población por zona

    ### 2. Configurar Análisis
    - Elegir interpretación de SFQ (frecuencia vs intervalo)
    - Seleccionar qué métricas incluir en el análisis
    - La aplicación usará las asignaciones tal como están definidas

    ### 3. Ejecutar y Revisar
    - La aplicación busca datos de viaje para cada par zona-destino asignado
    - Revisa estadísticas de accesibilidad para las asignaciones
    - Descarga resultados para uso posterior

    ## Ventajas de la Asignación Zona-Destino
    - **Control total**: Tú decides exactamente qué destino corresponde a cada zona
    - **Flexibilidad máxima**: Una zona puede asignarse a cualquier destino
    - **Análisis específico**: Obtiene métricas exactas para relaciones predefinidas
    - **Mapeo personalizado**: Perfecto para estudios con criterios específicos de asignación

    ## Casos de Uso Perfectos
    - **Asignación por proximidad**: Cada zona al hospital más cercano
    - **Asignación administrativa**: Cada zona a su centro de servicios correspondiente
    - **Asignación por capacidad**: Distribución balanceada según demanda
    - **Estudios comparativos**: Evaluar diferentes estrategias de asignación
    """)

with st.expander("📖 Entendiendo las Métricas"):
    st.markdown("""
    **JRT** Tiempo total del viaje (min) • **NTR** Transbordos (cantidad) • **RIT** Tiempo en vehículo (min)  
    **ACT/EGT** Tiempo de caminata acceso/salida (min) • **JRD** Distancia (km)  
    **SFQ** Servicio ofrecido: frecuencia (viajes/h, mayor mejor) o intervalo (min, menor mejor)

    ### Formato del Archivo de Asignación
    El archivo Excel debe tener exactamente estas columnas:
    - **ZONE_ID**: Identificador de la zona de origen
    - **DESTINATION_ID**: Identificador del destino asignado

    Cada fila representa una asignación: "La zona X debe usar el destino Y"

    ### Ejemplo de Diferentes Estrategias de Asignación
    - **Proximidad geográfica**: Cada zona al punto más cercano
    - **Balanceado por capacidad**: Distribución equitativa de demanda
    - **Jerárquico**: Zonas residenciales → centros barriales, centros barriales → centro principal
    - **Especializado**: Zonas según tipo de servicio requerido
    """)