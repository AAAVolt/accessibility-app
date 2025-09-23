# Aplicación Simplificada de Análisis de Accesibilidad
# Enfocada en IDs específicos desde archivo CSV

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Análisis de Accesibilidad - IDs Específicos", layout="wide")
st.title("🚌 Análisis de Accesibilidad del Transporte Público - IDs Específicos")

# Upload files
uploaded_skim_file = st.file_uploader("Subir archivo CSV de matriz de tiempos", type=["csv"], key="skim")
custom_ids_file = st.file_uploader("Subir archivo CSV con IDs de destinos específicos",
                                   type=["csv"], key="custom_ids",
                                   help="Archivo CSV con columna 'ID' conteniendo los números de zona de destino específicos")
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

# Check if custom IDs file is uploaded and display info
custom_destination_ids = None
if custom_ids_file:
    try:
        # Try different separators
        try:
            custom_ids_df = pd.read_csv(custom_ids_file, sep=';')
        except:
            custom_ids_file.seek(0)
            custom_ids_df = pd.read_csv(custom_ids_file)

        # Try to find the ID column (flexible column names)
        id_column = None
        possible_id_cols = ['ID', 'id', 'Id', 'zone_id', 'ZoneID', 'zona_id', 'DestZoneNo', 'dest_id']
        for col in possible_id_cols:
            if col in custom_ids_df.columns:
                id_column = col
                break

        if id_column is None:
            # If no standard column found, use the first column
            id_column = custom_ids_df.columns[0]
            st.warning(f"⚠️ No se encontró columna 'ID' estándar. Usando '{id_column}' como columna de IDs.")

        # Clean and convert to numeric
        custom_ids_df[id_column] = pd.to_numeric(custom_ids_df[id_column], errors='coerce')
        custom_ids_df = custom_ids_df.dropna(subset=[id_column])
        custom_destination_ids = sorted(custom_ids_df[id_column].astype(int).unique())

        st.sidebar.header("📋 Archivo de IDs Específicos")
        st.sidebar.success(f"✅ {len(custom_destination_ids)} IDs únicos cargados")
        st.sidebar.write(f"**Rango de IDs:** {min(custom_destination_ids)} - {max(custom_destination_ids)}")

        with st.sidebar.expander("👁️ Ver IDs cargados"):
            # Show first 20 IDs
            if len(custom_destination_ids) <= 20:
                st.write(", ".join(map(str, custom_destination_ids)))
            else:
                st.write(
                    ", ".join(map(str, custom_destination_ids[:20])) + f"... (+{len(custom_destination_ids) - 20} más)")

    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar archivo de IDs: {str(e)}")
        custom_destination_ids = None

if uploaded_skim_file and custom_destination_ids:
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

        # Filter to only custom destination IDs
        skim_df = skim_df[skim_df['DestZoneNo'].isin(custom_destination_ids)]

        # Display skim matrix info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros Totales", f"{len(skim_df):,}")
        with col2:
            st.metric("Zonas de Origen", f"{skim_df['OrigZoneNo'].nunique()}")
        with col3:
            st.metric("Destinos Específicos", f"{skim_df['DestZoneNo'].nunique()}")
        with col4:
            available_metrics = [col for col in ALL_METRICS if col in skim_df.columns]
            st.metric("Métricas Disponibles", len(available_metrics))

        st.info(
            f"📍 Analizando accesibilidad desde núcleos (1-453) hacia {len(custom_destination_ids)} destinos específicos")

        with st.expander("📊 Datos de Muestra de Matriz de Tiempos"):
            sample_cols = ['OrigZoneNo', 'DestZoneNo'] + available_metrics
            st.dataframe(skim_df[sample_cols].head(10), use_container_width=True)

        if skim_df.empty:
            st.error("⚠️ ¡No se encontraron datos en la matriz de tiempos para los destinos especificados!")
            st.stop()

        # Initialize results
        results = pd.DataFrame({'OrigZoneNo': sorted(set(range(1, 454)))})


        # Helper: find best row based on primary metric + direction
        def find_best_destination(group_df, primary_col):
            valid_rows = group_df.dropna(subset=[primary_col])
            if valid_rows.empty:
                return None
            direction = metric_direction.get(primary_col, "min")
            if direction == "max":
                return valid_rows.loc[valid_rows[primary_col].idxmax()]
            return valid_rows.loc[valid_rows[primary_col].idxmin()]


        # Análisis de IDs Específicos Más Cercano
        st.subheader(
            f"🎯 Análisis de Destinos Específicos Más Cercanos (por {primary_metric} / {'maximizar' if metric_direction[primary_metric] == 'max' else 'minimizar'})")

        best_destinations = []
        for orig_zone in results['OrigZoneNo']:
            zone_data = skim_df[skim_df['OrigZoneNo'] == orig_zone]
            best_row = find_best_destination(zone_data, primary_metric)

            result_row = {'OrigZoneNo': orig_zone}
            if best_row is not None:
                result_row['Destino_ID'] = int(best_row['DestZoneNo'])
                result_row[f'Mejor_{primary_metric}'] = best_row[primary_metric]
                for metric in secondary_metrics:
                    if metric in best_row:
                        result_row[f'Mejor_{metric}'] = best_row[metric]
            else:
                result_row['Destino_ID'] = np.nan
                result_row[f'Mejor_{primary_metric}'] = np.nan
                for metric in secondary_metrics:
                    result_row[f'Mejor_{metric}'] = np.nan
            best_destinations.append(result_row)

        if best_destinations:
            nearest_df = pd.DataFrame(best_destinations)
            results = results.merge(nearest_df, on='OrigZoneNo', how='left')

            valid_vals = results[f'Mejor_{primary_metric}'].dropna()
            if len(valid_vals) > 0:
                cols = st.columns(min(len(secondary_metrics) + 4, 6))
                with cols[0]:
                    st.metric("Zonas con Acceso", f"{len(valid_vals)}")
                with cols[1]:
                    st.metric(f"Prom {primary_metric}", f"{valid_vals.mean():.2f}")
                with cols[2]:
                    st.metric(f"Mín {primary_metric}", f"{valid_vals.min():.2f}")
                with cols[3]:
                    st.metric(f"Máx {primary_metric}", f"{valid_vals.max():.2f}")
                for i, metric in enumerate(secondary_metrics, 4):
                    if i < len(cols):
                        colname = f'Mejor_{metric}'
                        if colname in results.columns:
                            vv = results[colname].dropna()
                            if len(vv) > 0:
                                with cols[i]:
                                    st.metric(f"Prom {metric}", f"{vv.mean():.2f}")

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

        display_columns = ['OrigZoneNo']
        if 'Population' in results.columns:
            display_columns.append('Population')

        # Add all metric columns
        metric_columns = [col for col in results.columns if any(col.endswith(f'_{m}') for m in ALL_METRICS)]
        zone_columns = [col for col in results.columns if 'Destino_ID' in col]

        display_columns.extend(zone_columns + metric_columns)
        display_columns = [col for col in display_columns if col in results.columns]

        st.dataframe(results[display_columns], use_container_width=True)

        # Download results
        csv_download = results.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resultados como CSV",
            data=csv_download,
            file_name="accesibilidad_destinos_especificos_resultados.csv",
            mime="text/csv"
        )

        # Population accessibility summary with thresholds
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

                primary_metric_column = f'Mejor_{primary_metric}'

                if primary_metric_column in results.columns:
                    for thr in thresholds:
                        if metric_direction.get(primary_metric, "min") == "max":
                            accessible_mask = results[primary_metric_column] >= thr
                        else:
                            accessible_mask = results[primary_metric_column] <= thr
                        accessible_pop = results.loc[accessible_mask, 'Population'].sum()
                        accessible_zones = int(accessible_mask.sum())
                        total_zones = len(results)
                        pop_pct = 100 * accessible_pop / total_pop if total_pop > 0 else 0
                        zone_pct = 100 * accessible_zones / total_zones if total_zones > 0 else 0

                        summary_rows.append({
                            "Método": "IDs Específicos",
                            f"Umbral ({primary_metric})": thr,
                            "Población Accesible": int(accessible_pop),
                            "Población Total": int(total_pop),
                            "Población (%)": round(pop_pct, 2),
                            "Zonas Accesibles": accessible_zones,
                            "Zonas Totales": total_zones,
                            "Zonas (%)": round(zone_pct, 2)
                        })

                    # unreachable info
                    unreachable_mask = results[primary_metric_column].isna()
                    unreachable_pop = int(results.loc[unreachable_mask, 'Population'].sum())
                    unreachable_zones = int(unreachable_mask.sum())
                    if unreachable_zones > 0:
                        st.warning(
                            f"🚫 **IDs Específicos**: {unreachable_zones} zonas ({unreachable_pop:,} personas) no tienen destino alcanzable")

                    if summary_rows:
                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, use_container_width=True)

                        csv_summary = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resumen como CSV",
                            data=csv_summary,
                            file_name="resumen_accesibilidad_destinos_especificos.csv",
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
        if secondary_metrics and len(
                [col for col in results.columns if any(col.endswith(f'_{m}') for m in ALL_METRICS)]) >= 2:
            st.subheader("📈 Análisis Multi-Métrica")
            st.write("**Correlación entre métricas:**")

            metric_cols = [col for col in results.columns if any(col.endswith(f'_{m}') for m in ALL_METRICS)]
            if len(metric_cols) >= 2:
                metric_data = results[metric_cols].select_dtypes(include=[np.number])
                if not metric_data.empty:
                    correlation_matrix = metric_data.corr()
                    st.write("Matriz de correlación entre métricas de accesibilidad:")
                    st.dataframe(correlation_matrix.round(3), use_container_width=True)

                    with st.expander("💡 Información de Métricas"):
                        for col in metric_cols:
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
    st.info("👆 Por favor suba la matriz de tiempos y el archivo de IDs específicos para comenzar el análisis")

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
        ### 🎯 Archivo de IDs Específicos
        - **CSV con columna 'ID'** conteniendo números de zona específicos
        - Puede tener IDs no consecutivos (ej: 713, 714, 715, 720...)
        - **Ejemplo:**
        ```
        ID
        713
        714
        715
        720
        724
        ```

        **Características de esta versión simplificada:**
        - ✅ **Solo análisis de IDs específicos**: Sin necesidad de archivo de mapeo
        - ✅ **Interfaz simplificada**: Menos opciones, más directo al grano  
        - ✅ **Métricas completas**: Soporte para todas las métricas de transporte
        - ✅ **Análisis poblacional** con umbrales adaptativos
        - ✅ **Descarga de resultados** en formato CSV
        """)

# Footer info
with st.expander("ℹ️ Cómo usar esta herramienta simplificada"):
    st.markdown("""
    ## Flujo de Trabajo Simplificado

    ### 1. Preparar Archivos
    - **Matriz de tiempos**: CSV con datos de viaje entre zonas
    - **IDs específicos**: CSV con columna 'ID' conteniendo destinos de interés
    - **Población** (opcional): XLSX con población por zona

    ### 2. Configurar Análisis
    - Elegir interpretación de SFQ (frecuencia vs intervalo)
    - Seleccionar métrica principal para encontrar el "mejor" destino
    - Incluir métricas adicionales en los resultados

    ### 3. Ejecutar y Revisar
    - La aplicación encuentra automáticamente el destino más cercano/mejor para cada zona
    - Revisa estadísticas de accesibilidad
    - Descarga resultados para uso posterior

    ## Ventajas de esta versión
    - **Más simple**: Sin complejidad de mapeos zona-destino
    - **Más directa**: Solo se enfoca en encontrar el mejor destino de tu lista específica
    - **Más rápida**: Menos configuraciones, análisis más directo
    - **Más clara**: Resultados enfocados en lo que realmente necesitas
    """)

with st.expander("📖 Entendiendo las Métricas"):
    st.markdown("""
    **JRT** Tiempo total del viaje (min) • **NTR** Transbordos (cantidad) • **RIT** Tiempo en vehículo (min)  
    **ACT/EGT** Tiempo de caminata acceso/salida (min) • **JRD** Distancia (km)  
    **SFQ** Servicio ofrecido: frecuencia (viajes/h, mayor mejor) o intervalo (min, menor mejor)

    ### Casos de Uso Perfectos para IDs Específicos
    - **Hospitales de referencia**: Solo hospitales principales de la red
    - **Estaciones intermodales**: Estaciones de metro/tren principales
    - **Centros administrativos**: Oficinas gubernamentales específicas
    - **Centros educativos**: Campus universitarios o centros de FP específicos
    """)