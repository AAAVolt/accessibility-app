# Aplicación Simplificada de Análisis de Accesibilidad
# Enfocada en optimización por tiempo de viaje (JRT) con comparación de transporte privado

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
    "PJT",  # Perceived journey time
    "RID",  # Ride distance
    "RIT",  # Ride time
    "RITA",  # Ride time adapted
    "SFQ",  # Service frequency
    "TWT",  # Transfer wait time
    "WKD",  # Walk distance
    "WKT"  # Walk time
]

# Car matrix metrics
CAR_METRICS = ["TT0", "VP0", "DIS"]

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
    "WKT": "Tiempo de caminata (min)",
    "PJT": "Tiempo de viaje percibido (min)",
    "TT0": "Tiempo viaje coche fuera de pico (min)",
    "VP0": "Valor del tiempo fuera de pico",
    "DIS": "Distancia (km)"
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
    "WKT": "Tiempo_Caminata",
    "PJT": "Tiempo_Viaje_Percibido",
    "TT0": "TT_Coche_NoPico",
    "VP0": "Valor_NoPico",
    "DIS": "Distancia_km"
}

# File upload
st.sidebar.header("📁 Subir Archivos")
uploaded_skim_file = st.file_uploader(
    "Subir archivo CSV de matriz de tiempos (Transporte Público)",
    type=["csv"],
    help="CSV con datos de viaje entre zonas de origen y destino"
)

custom_ids_file = st.file_uploader(
    "Subir archivo CSV con IDs de destinos específicos",
    type=["csv"],
    help="CSV con columna 'ID' conteniendo los números de zona de destino específicos"
)

car_matrix_file = st.file_uploader(
    "Subir archivo CSV de matriz de coche (Opcional)",
    type=["csv"],
    help="CSV con datos de viaje en coche privado: OrigZoneNo, DestZoneNo, TT0, VP0, DIS"
)

# Load custom destination IDs
custom_destination_ids = None
if custom_ids_file:
    try:
        try:
            custom_ids_df = pd.read_csv(custom_ids_file, sep=';')
        except:
            custom_ids_file.seek(0)
            custom_ids_df = pd.read_csv(custom_ids_file)

        id_column = None
        possible_id_cols = ['ID', 'id', 'Id', 'zone_id', 'ZoneID', 'zona_id', 'DestZoneNo', 'dest_id']
        for col in possible_id_cols:
            if col in custom_ids_df.columns:
                id_column = col
                break

        if id_column is None:
            id_column = custom_ids_df.columns[0]
            st.warning(f"⚠️ No se encontró columna 'ID' estándar. Usando '{id_column}' como columna de IDs.")

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

# Load car matrix
car_df = None
if car_matrix_file:
    try:
        car_df = pd.read_csv(car_matrix_file)

        # Clean data types for car matrix
        car_df["OrigZoneNo"] = pd.to_numeric(car_df["OrigZoneNo"], errors='coerce')
        car_df["DestZoneNo"] = pd.to_numeric(car_df["DestZoneNo"], errors='coerce')

        for col in CAR_METRICS:
            if col in car_df.columns:
                car_df[col] = pd.to_numeric(car_df[col], errors='coerce')

        car_df = car_df.dropna(subset=["OrigZoneNo", "DestZoneNo"])
        car_df["OrigZoneNo"] = car_df["OrigZoneNo"].astype(int)
        car_df["DestZoneNo"] = car_df["DestZoneNo"].astype(int)

        st.sidebar.success(f"✅ Matriz de coche cargada: {len(car_df):,} registros")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar matriz de coche: {str(e)}")
        car_df = None

# Main analysis
if uploaded_skim_file and custom_destination_ids:
    st.sidebar.header("⚙️ Configuración")

    origin_start = st.sidebar.number_input("Zona de origen inicial", value=1, min_value=1)
    origin_end = st.sidebar.number_input("Zona de origen final", value=453, min_value=1)

    if st.sidebar.button("🚀 Ejecutar Análisis de Accesibilidad", type="primary"):

        def load_skim(file):
            try:
                df = pd.read_csv(file)
                if "OrigZoneNo" in df.columns and "DestZoneNo" in df.columns:
                    return df
            except:
                pass

            file.seek(0)
            content = file.read().decode("utf-8-sig")
            all_lines = content.splitlines()

            # --- Try to extract column names from the Visum $Relations header ---
            # e.g. "$Relations:OrigZoneNo,DestZoneNo,ACD,ACT,IVT,JRT,..."
            cols_from_relations = None
            for line in all_lines:
                stripped = line.strip()
                if stripped.startswith("$Relations:"):
                    cols_part = stripped[len("$Relations:"):]
                    cols_from_relations = [c.strip() for c in cols_part.split(",")]
                    break

            # Keep only genuine data lines: must start with an integer followed by a comma
            data_lines = [l.strip() for l in all_lines if re.match(r'^\d+,', l.strip())]
            data = "\n".join(data_lines)

            if cols_from_relations:
                # Use the exact columns declared in the file — no positional guessing
                cols_to_use = cols_from_relations
            elif data_lines:
                # Last resort: assign positionally from AVAILABLE_METRICS
                first_line_cols = len(data_lines[0].split(','))
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


        with st.spinner("📊 Cargando matriz de tiempos..."):
            skim_df = load_skim(uploaded_skim_file)

            skim_df["OrigZoneNo"] = pd.to_numeric(skim_df["OrigZoneNo"], errors='coerce')
            skim_df["DestZoneNo"] = pd.to_numeric(skim_df["DestZoneNo"], errors='coerce')

            for col in AVAILABLE_METRICS:
                if col in skim_df.columns:
                    skim_df[col] = pd.to_numeric(skim_df[col], errors='coerce')

            skim_df = skim_df.dropna(subset=["OrigZoneNo", "DestZoneNo"])
            skim_df["OrigZoneNo"] = skim_df["OrigZoneNo"].astype(int)
            skim_df["DestZoneNo"] = skim_df["DestZoneNo"].astype(int)

        st.success(f"✅ Matriz TP cargada: {len(skim_df):,} registros")

        skim_filtered = skim_df[skim_df["DestZoneNo"].isin(custom_destination_ids)].copy()

        if skim_filtered.empty:
            st.error("❌ No se encontraron datos para los IDs de destino especificados.")
        else:
            st.info(
                f"🎯 Datos filtrados TP: {len(skim_filtered):,} registros para {len(custom_destination_ids)} destinos específicos")

            if "JRT" not in skim_filtered.columns:
                st.error("❌ La columna 'JRT' (Journey Time) no se encontró en los datos.")
            else:
                with st.spinner("🔍 Encontrando destinos óptimos por tiempo de viaje..."):

                    origin_zones = range(origin_start, origin_end + 1)
                    results_list = []

                    for origin in origin_zones:
                        origin_trips = skim_filtered[skim_filtered["OrigZoneNo"] == origin].copy()

                        if not origin_trips.empty:
                            valid_trips = origin_trips.dropna(subset=["JRT"])

                            if not valid_trips.empty:
                                min_jrt_idx = valid_trips["JRT"].idxmin()
                                best_trip = valid_trips.loc[min_jrt_idx]

                                result_row = {
                                    "Zona_Origen": origin,
                                    "Destino_Optimo": int(best_trip["DestZoneNo"])
                                }

                                for metric in AVAILABLE_METRICS:
                                    if metric in best_trip.index:
                                        result_row[metric] = best_trip[metric]

                                # Add car matrix data if available
                                if car_df is not None:
                                    car_data = car_df[
                                        (car_df["OrigZoneNo"] == origin) &
                                        (car_df["DestZoneNo"] == int(best_trip["DestZoneNo"]))
                                        ]

                                    if not car_data.empty:
                                        car_row = car_data.iloc[0]
                                        for car_metric in CAR_METRICS:
                                            if car_metric in car_row.index:
                                                result_row[car_metric] = car_row[car_metric]
                                    else:
                                        for car_metric in CAR_METRICS:
                                            result_row[car_metric] = np.nan

                                results_list.append(result_row)

                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        st.success(f"✅ Análisis completado para {len(results_df)} zonas de origen")

                        # Display results table
                        st.subheader("📋 Resultados de Accesibilidad")
                        st.write("**Destino óptimo (menor tiempo de viaje) y métricas asociadas por zona de origen:**")

                        display_df = results_df.copy()
                        for col in display_df.columns:
                            if col not in ["Zona_Origen", "Destino_Optimo"] and display_df[col].dtype in ['float64',
                                                                                                          'int64']:
                                if display_df[col].dtype == 'float64':
                                    display_df[col] = display_df[col].round(2)

                        display_df = display_df.rename(columns=spanish_column_names)
                        st.dataframe(display_df, use_container_width=True)

                        # Comparison tab if car data available
                        if car_df is not None:
                            st.subheader("🚗 Comparación Transporte Público vs Coche Privado")

                            # Create comparison dataframe
                            comparison_df = results_df[["Zona_Origen", "Destino_Optimo", "JRT"]].copy()

                            if "PJT" in results_df.columns:
                                comparison_df["PJT"] = results_df["PJT"]

                            if "TT0" in results_df.columns:
                                comparison_df["TT0_Coche"] = results_df["TT0"]
                                comparison_df["Diferencia_NoPico"] = comparison_df["JRT"] - comparison_df["TT0_Coche"]
                                if "PJT" in comparison_df.columns:
                                    comparison_df["Diferencia_PJT_Coche"] = comparison_df["PJT"] - comparison_df["TT0_Coche"]

                            if "DIS" in results_df.columns:
                                comparison_df["Distancia"] = results_df["DIS"]

                            # Display comparison
                            display_comparison = comparison_df.copy()
                            for col in display_comparison.columns:
                                if display_comparison[col].dtype == 'float64':
                                    display_comparison[col] = display_comparison[col].round(2)

                            display_comparison = display_comparison.rename(columns={
                                "Zona_Origen": "Zona Origen",
                                "Destino_Optimo": "Destino",
                                "JRT": "TP Tiempo (min)",
                                "PJT": "PJT Tiempo Percibido (min)",
                                "TT0_Coche": "Coche Fuera Pico (min)",
                                "Diferencia_NoPico": "Diferencia JRT-Coche NP (min)",
                                "Diferencia_PJT_Coche": "Diferencia PJT-Coche NP (min)",
                                "Distancia": "Distancia (km)"
                            })

                            st.dataframe(display_comparison, use_container_width=True)

                            # Comparison statistics
                            st.subheader("📊 Estadísticas de Comparación")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if "TT0_Coche" in comparison_df.columns:
                                    avg_tp = comparison_df["JRT"].mean()
                                    avg_car_np = comparison_df["TT0_Coche"].mean()
                                    st.metric("TP Promedio (min)", f"{avg_tp:.1f}")
                                    st.metric("Coche Fuera Pico Promedio (min)", f"{avg_car_np:.1f}")

                            with col3:
                                if "Diferencia_NoPico" in comparison_df.columns:
                                    tp_faster = (comparison_df["Diferencia_NoPico"] < 0).sum()
                                    car_faster = (comparison_df["Diferencia_NoPico"] > 0).sum()
                                    st.metric("TP más rápido", tp_faster)
                                    st.metric("Coche más rápido", car_faster)

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
                                st.write("**Estadísticas Tiempo de Viaje TP (JRT):**")
                                st.write(f"• Media: {jrt_stats['mean']:.2f} min")
                                st.write(f"• Mediana: {jrt_stats['50%']:.2f} min")
                                st.write(f"• Mínimo: {jrt_stats['min']:.2f} min")
                                st.write(f"• Máximo: {jrt_stats['max']:.2f} min")

                            if "PJT" in results_df.columns:
                                pjt_stats = results_df["PJT"].describe()
                                st.write("**Estadísticas Tiempo de Viaje Percibido (PJT):**")
                                st.write(f"• Media: {pjt_stats['mean']:.2f} min")
                                st.write(f"• Mediana: {pjt_stats['50%']:.2f} min")
                                st.write(f"• Mínimo: {pjt_stats['min']:.2f} min")
                                st.write(f"• Máximo: {pjt_stats['max']:.2f} min")

                        # Download buttons
                        col1, col2 = st.columns(2)

                        with col1:
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Resultados como CSV",
                                data=csv_data,
                                file_name="resultados_accesibilidad_tiempo_viaje.csv",
                                mime="text/csv",
                                help="El CSV mantiene los códigos originales de métricas para compatibilidad"
                            )

                        with col2:
                            if car_df is not None:
                                comparison_csv = comparison_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar Comparación TP vs Coche",
                                    data=comparison_csv,
                                    file_name="comparacion_tp_coche.csv",
                                    mime="text/csv"
                                )

                        # Detailed metrics info
                        with st.expander("📈 Información Detallada de Métricas"):
                            available_metrics_in_data = [col for col in results_df.columns if col in AVAILABLE_METRICS]
                            available_car_metrics = [col for col in results_df.columns if col in CAR_METRICS]

                            if available_metrics_in_data:
                                st.write("**Métricas de Transporte Público:**")
                                for metric in available_metrics_in_data:
                                    if metric in metric_descriptions:
                                        st.write(f"• **{metric}**: {metric_descriptions[metric]}")

                            if available_car_metrics:
                                st.write("\n**Métricas de Coche Privado:**")
                                for metric in available_car_metrics:
                                    if metric in metric_descriptions:
                                        st.write(f"• **{metric}**: {metric_descriptions[metric]}")

                            st.write("\n**Estadísticas por métrica:**")
                            all_numeric_cols = available_metrics_in_data + available_car_metrics
                            if all_numeric_cols:
                                metrics_stats = results_df[all_numeric_cols].describe().round(2)
                                st.dataframe(metrics_stats, use_container_width=True)

                    else:
                        st.warning("⚠️ No se encontraron rutas válidas para las zonas de origen especificadas.")

else:
    st.info("👆 Por favor suba la matriz de tiempos y el archivo de IDs específicos para comenzar el análisis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 Archivo de Matriz de Tiempos (Transporte Público)
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
        - `PJT` - Tiempo de viaje percibido
        """)

    with col2:
        st.markdown("""
        ### 🚗 Archivo de Matriz de Coche (Opcional)
        CSV con columnas:
        - `OrigZoneNo, DestZoneNo` (requeridas)
        - `TT0` - Tiempo viaje fuera de pico (min)
        - `VP0` - Valor del tiempo fuera de pico
        - `DIS` - Distancia (km)

        ### 🎯 Archivo de IDs Específicos
        CSV simple con columna 'ID':
        ```
        ID
        713
        714
        715
        ```
        """)

with st.expander("📖 Descripción de las Métricas"):
    st.markdown("""
    ### Métricas de Accesibilidad Disponibles

    **Transporte Público - Tiempos (minutos):**
    - **JRT**: Tiempo total del viaje (puerta a puerta) - *MÉTRICA PRINCIPAL*
    - **ACT**: Tiempo de acceso/caminata al transporte
    - **EGT**: Tiempo de salida/caminata desde transporte
    - **IVT**: Tiempo en vehículo
    - **RIT**: Tiempo del trayecto
    - **RITA**: Tiempo del trayecto adaptado
    - **TWT**: Tiempo de espera para transbordo
    - **WKT**: Tiempo de caminata total
    - **PJT**: Tiempo de viaje percibido (incluye penalizaciones por espera, transbordos y caminata)

    **Transporte Público - Distancias:**
    - **JRD**: Distancia total del viaje
    - **ACD**: Distancia de acceso
    - **EGD**: Distancia de salida
    - **IVD**: Distancia en vehículo
    - **RID**: Distancia del trayecto
    - **WKD**: Distancia de caminata

    **Transporte Público - Otros:**
    - **NTR**: Número de transbordos requeridos
    - **SFQ**: Frecuencia/intervalo del servicio

    **Coche Privado:**
    - **TT0**: Tiempo de viaje fuera de pico (min)
    - **VP0**: Valor del tiempo fuera de pico
    - **DIS**: Distancia del recorrido (km)

    La aplicación optimiza por **JRT** y permite comparar los tiempos de viaje del transporte público (JRT y PJT) con los tiempos en coche privado.
    """)