# Aplicación Simplificada de Análisis de Accesibilidad
# Enfocada en optimización por tiempo de viaje (JRT)

import streamlit as st
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Análisis de Accesibilidad Simplificado", layout="wide")
st.title("🚌 Análisis de Accesibilidad - Optimización por Tiempo de Viaje")

# Available metrics from skim results
AVAILABLE_METRICS = [
    "ACD",  # Access distance
    "ACT",  # Access time
    "EGD",  # Egress distance
    "EGT",  # Egress time
    "IVD",  # In-vehicle distance
    "IVT",  # In-vehicle time
    "JRD",  # Journey distance
    "JRT",  # Journey time
    "NTR",  # Number of transfers
    "RID",  # Ride distance
    "RIT",  # Ride time
    "RITA",  # Ride time adapted
    "SFQ",  # Service frequency
    "TWT",  # Transfer wait time
    "WKD",  # Walk distance
    "WKT"  # Walk time
]

DEFAULT_COLS = ["OrigZoneNo", "DestZoneNo"] + AVAILABLE_METRICS

# Spanish metric descriptions
metric_descriptions = {
    "ACD": "Distancia de acceso",
    "ACT": "Tiempo de acceso (min)",
    "EGD": "Distancia de salida",
    "EGT": "Tiempo de salida (min)",
    "IVD": "Distancia en vehículo",
    "IVT": "Tiempo en vehículo (min)",
    "JRD": "Distancia del viaje",
    "JRT": "Tiempo del viaje (min)",
    "NTR": "Número de transbordos",
    "RID": "Distancia del trayecto",
    "RIT": "Tiempo del trayecto (min)",
    "RITA": "Tiempo del trayecto adaptado (min)",
    "SFQ": "Frecuencia del servicio",
    "TWT": "Tiempo de espera para transbordo (min)",
    "WKD": "Distancia de caminata",
    "WKT": "Tiempo de caminata (min)"
}

# Spanish column names for results table
spanish_column_names = {
    "Zona_Origen": "Zona_Origen",
    "Destino_Optimo": "Destino_Óptimo",
    "ACD": "Dist_Acceso",
    "ACT": "Tiempo_Acceso",
    "EGD": "Dist_Salida",
    "EGT": "Tiempo_Salida",
    "IVD": "Dist_Vehiculo",
    "IVT": "Tiempo_Vehiculo",
    "JRD": "Dist_Viaje",
    "JRT": "Tiempo_Viaje",
    "NTR": "Num_Transbordos",
    "RID": "Dist_Trayecto",
    "RIT": "Tiempo_Trayecto",
    "RITA": "Tiempo_Trayecto_Adapt",
    "SFQ": "Frecuencia_Servicio",
    "TWT": "Tiempo_Espera_Transb",
    "WKD": "Dist_Caminata",
    "WKT": "Tiempo_Caminata"
}

# File upload
st.sidebar.header("📁 Subir Archivos")
uploaded_skim_file = st.file_uploader(
    "Subir archivo CSV de matriz de tiempos",
    type=["csv"],
    help="CSV con datos de viaje entre zonas de origen y destino"
)

custom_ids_file = st.file_uploader(
    "Subir archivo CSV con IDs de destinos específicos",
    type=["csv"],
    help="CSV con columna 'ID' conteniendo los números de zona de destino específicos"
)

# Load custom destination IDs
custom_destination_ids = None
if custom_ids_file:
    try:
        # Try different separators
        try:
            custom_ids_df = pd.read_csv(custom_ids_file, sep=';')
        except:
            custom_ids_file.seek(0)
            custom_ids_df = pd.read_csv(custom_ids_file)

        # Try to find the ID column
        id_column = None
        possible_id_cols = ['ID', 'id', 'Id', 'zone_id', 'ZoneID', 'zona_id', 'DestZoneNo', 'dest_id']
        for col in possible_id_cols:
            if col in custom_ids_df.columns:
                id_column = col
                break

        if id_column is None:
            # Use first column if no standard column found
            id_column = custom_ids_df.columns[0]
            st.warning(f"⚠️ No se encontró columna 'ID' estándar. Usando '{id_column}' como columna de IDs.")

        # Clean and convert to numeric
        custom_ids_df[id_column] = pd.to_numeric(custom_ids_df[id_column], errors='coerce')
        custom_ids_df = custom_ids_df.dropna(subset=[id_column])
        custom_destination_ids = sorted(custom_ids_df[id_column].astype(int).unique())

        st.sidebar.success(f"✅ {len(custom_destination_ids)} IDs únicos cargados")
        st.sidebar.write(f"**Rango de IDs:** {min(custom_destination_ids)} - {max(custom_destination_ids)}")

        with st.sidebar.expander("👁️ Ver IDs cargados"):
            if len(custom_destination_ids) <= 20:
                st.write(", ".join(map(str, custom_destination_ids)))
            else:
                st.write(
                    ", ".join(map(str, custom_destination_ids[:20])) + f"... (+{len(custom_destination_ids) - 20} más)")

    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar archivo de IDs: {str(e)}")
        custom_destination_ids = None

# Main analysis
if uploaded_skim_file and custom_destination_ids:
    st.sidebar.header("⚙️ Configuración")

    # Origin zones configuration
    origin_start = st.sidebar.number_input("Zona de origen inicial", value=1, min_value=1)
    origin_end = st.sidebar.number_input("Zona de origen final", value=453, min_value=1)

    if st.sidebar.button("🚀 Ejecutar Análisis de Accesibilidad", type="primary"):

        # Robust CSV loader
        def load_skim(file):
            try:
                # Try standard CSV read first
                df = pd.read_csv(file)
                if "OrigZoneNo" in df.columns and "DestZoneNo" in df.columns:
                    return df
            except:
                pass

            # Fallback: parse lines starting with digits
            file.seek(0)
            content = file.read().decode("utf-8")
            lines = [line.strip() for line in content.splitlines() if re.match(r'^\d', line)]
            data = "\n".join(lines)

            # Determine number of columns from first line
            if lines:
                first_line_cols = len(lines[0].split(','))
                # Use available metrics up to the number of columns
                cols_to_use = ["OrigZoneNo", "DestZoneNo"] + AVAILABLE_METRICS[:first_line_cols - 2]
            else:
                cols_to_use = DEFAULT_COLS

            df = pd.read_csv(
                io.StringIO(data),
                names=cols_to_use,
                on_bad_lines="skip",
                thousands=","
            )
            return df


        # Load and process skim matrix
        with st.spinner("📊 Cargando matriz de tiempos..."):
            skim_df = load_skim(uploaded_skim_file)

            # Clean data types
            skim_df["OrigZoneNo"] = pd.to_numeric(skim_df["OrigZoneNo"], errors='coerce')
            skim_df["DestZoneNo"] = pd.to_numeric(skim_df["DestZoneNo"], errors='coerce')

            # Clean metric columns
            for col in AVAILABLE_METRICS:
                if col in skim_df.columns:
                    skim_df[col] = pd.to_numeric(skim_df[col], errors='coerce')

            # Remove rows with missing origin/destination
            skim_df = skim_df.dropna(subset=["OrigZoneNo", "DestZoneNo"])
            skim_df["OrigZoneNo"] = skim_df["OrigZoneNo"].astype(int)
            skim_df["DestZoneNo"] = skim_df["DestZoneNo"].astype(int)

        st.success(f"✅ Matriz cargada: {len(skim_df):,} registros")

        # Filter for custom destination IDs
        skim_filtered = skim_df[skim_df["DestZoneNo"].isin(custom_destination_ids)].copy()

        if skim_filtered.empty:
            st.error("❌ No se encontraron datos para los IDs de destino especificados.")
        else:
            st.info(
                f"🎯 Datos filtrados: {len(skim_filtered):,} registros para {len(custom_destination_ids)} destinos específicos")

            # Check if JRT column exists
            if "JRT" not in skim_filtered.columns:
                st.error("❌ La columna 'JRT' (Journey Time) no se encontró en los datos.")
            else:
                with st.spinner("🔍 Encontrando destinos óptimos por tiempo de viaje..."):

                    # Filter origins
                    origin_zones = range(origin_start, origin_end + 1)

                    results_list = []

                    for origin in origin_zones:
                        # Get all trips from this origin to custom destinations
                        origin_trips = skim_filtered[skim_filtered["OrigZoneNo"] == origin].copy()

                        if not origin_trips.empty:
                            # Remove rows where JRT is null/invalid
                            valid_trips = origin_trips.dropna(subset=["JRT"])

                            if not valid_trips.empty:
                                # Find the destination with minimum journey time
                                min_jrt_idx = valid_trips["JRT"].idxmin()
                                best_trip = valid_trips.loc[min_jrt_idx]

                                # Create result row with all available metrics
                                result_row = {
                                    "Zona_Origen": origin,
                                    "Destino_Optimo": int(best_trip["DestZoneNo"])
                                }

                                # Add all available metrics
                                for metric in AVAILABLE_METRICS:
                                    if metric in best_trip.index:
                                        result_row[metric] = best_trip[metric]

                                results_list.append(result_row)

                    # Create results dataframe
                    if results_list:
                        results_df = pd.DataFrame(results_list)

                        st.success(f"✅ Análisis completado para {len(results_df)} zonas de origen")

                        # Display results table
                        st.subheader("📋 Resultados de Accesibilidad")
                        st.write("**Destino óptimo (menor tiempo de viaje) y métricas asociadas por zona de origen:**")

                        # Round numeric columns for display
                        display_df = results_df.copy()
                        for col in display_df.columns:
                            if col not in ["Zona_Origen", "Destino_Optimo"] and display_df[col].dtype in ['float64',
                                                                                                          'int64']:
                                if display_df[col].dtype == 'float64':
                                    display_df[col] = display_df[col].round(2)

                        # Rename columns to Spanish for display
                        display_df = display_df.rename(columns=spanish_column_names)

                        st.dataframe(display_df, use_container_width=True)

                        # Summary statistics
                        st.subheader("📊 Estadísticas Resumen")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Destinos más utilizados:**")
                            dest_counts = results_df["Destino_Optimo"].value_counts().head(10)
                            st.write(dest_counts)

                        with col2:
                            if "JRT" in results_df.columns:
                                jrt_stats = results_df["JRT"].describe()
                                st.write("**Estadísticas Tiempo de Viaje (Tiempo_Viaje):**")
                                st.write(f"• Media: {jrt_stats['mean']:.2f} min")
                                st.write(f"• Mediana: {jrt_stats['50%']:.2f} min")
                                st.write(f"• Mínimo: {jrt_stats['min']:.2f} min")
                                st.write(f"• Máximo: {jrt_stats['max']:.2f} min")

                        # Download button - keep original column names for CSV compatibility
                        csv_data = results_df.to_csv(index=False)  # Use original results_df, not display_df
                        st.download_button(
                            label="📥 Descargar Resultados como CSV",
                            data=csv_data,
                            file_name="resultados_accesibilidad_tiempo_viaje.csv",
                            mime="text/csv",
                            help="El CSV mantiene los códigos originales de métricas para compatibilidad"
                        )

                        # Detailed metrics info
                        with st.expander("📈 Información Detallada de Métricas"):
                            available_metrics_in_data = [col for col in results_df.columns if col in AVAILABLE_METRICS]

                            if available_metrics_in_data:
                                st.write("**Métricas disponibles en los resultados:**")
                                for metric in available_metrics_in_data:
                                    if metric in metric_descriptions:
                                        st.write(f"• **{metric}**: {metric_descriptions[metric]}")

                                st.write("\n**Estadísticas por métrica:**")
                                metrics_stats = results_df[available_metrics_in_data].describe().round(2)
                                st.dataframe(metrics_stats, use_container_width=True)

                    else:
                        st.warning("⚠️ No se encontraron rutas válidas para las zonas de origen especificadas.")

else:
    # Instructions when no files uploaded
    st.info("👆 Por favor suba la matriz de tiempos y el archivo de IDs específicos para comenzar el análisis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 Archivo de Matriz de Tiempos
        Formato CSV con las siguientes columnas posibles:

        **Columnas principales:**
        - `OrigZoneNo, DestZoneNo` (requeridas)
        - `JRT` (Journey Time - requerida para optimización)

        **Métricas adicionales disponibles:**
        - `ACD, ACT` - Distancia y tiempo de acceso
        - `EGD, EGT` - Distancia y tiempo de salida  
        - `IVD, IVT` - Distancia y tiempo en vehículo
        - `JRD` - Distancia del viaje
        - `NTR` - Número de transbordos
        - `RID, RIT, RITA` - Distancia y tiempo del trayecto
        - `SFQ` - Frecuencia del servicio
        - `TWT` - Tiempo de espera para transbordo
        - `WKD, WKT` - Distancia y tiempo de caminata
        """)

    with col2:
        st.markdown("""
        ### 🎯 Archivo de IDs Específicos
        CSV simple con columna 'ID' conteniendo números de zona de destino:

        **Ejemplo:**
        ```
        ID
        713
        714
        715
        720
        724
        ```

        ### 🚀 ¿Qué hace esta aplicación?

        1. **Carga** la matriz de tiempos y destinos específicos
        2. **Encuentra** para cada zona de origen el destino con menor tiempo de viaje (JRT)
        3. **Extrae** todas las métricas disponibles para esa combinación óptima
        4. **Genera** una tabla de resultados de accesibilidad
        5. **Permite descargar** los resultados en CSV
        """)

# Footer with metric explanations
with st.expander("📖 Descripción de las Métricas"):
    st.markdown("""
    ### Métricas de Accesibilidad Disponibles

    **Tiempos (minutos):**
    - **JRT**: Tiempo total del viaje (puerta a puerta) - *MÉTRICA PRINCIPAL*
    - **ACT**: Tiempo de acceso/caminata al transporte
    - **EGT**: Tiempo de salida/caminata desde transporte
    - **IVT**: Tiempo en vehículo
    - **RIT**: Tiempo del trayecto
    - **RITA**: Tiempo del trayecto adaptado
    - **TWT**: Tiempo de espera para transbordo
    - **WKT**: Tiempo de caminata total

    **Distancias:**
    - **JRD**: Distancia total del viaje
    - **ACD**: Distancia de acceso
    - **EGD**: Distancia de salida
    - **IVD**: Distancia en vehículo
    - **RID**: Distancia del trayecto
    - **WKD**: Distancia de caminata

    **Otros:**
    - **NTR**: Número de transbordos requeridos
    - **SFQ**: Frecuencia/intervalo del servicio

    La aplicación optimiza por **JRT** (tiempo total) y reporta todas las demás métricas para la ruta óptima encontrada.
    """)