import pandas as pd
import numpy as np


def clean_accessibility_data(input_file, output_file=None):
    """
    Clean and process accessibility data with the following logic:

    1. Use Access_Time_Minutes and Access_Distance_Meters if not empty/NaN
       Otherwise use Mejor_ACT and Mejor_ACD

    2. Use Egress_Time_Minutes and Egress_Distance_Meters if not empty/NaN
       Otherwise use Mejor_EGT and Mejor_EGD

    3. Recalculate journey time when using calculated access/egress times:
       New_Journey_Time = Mejor_JRT - Mejor_ACT - Mejor_EGT + Final_Access_Time + Final_Egress_Time

    4. Rename columns to Spanish with clear names
    """

    print("Leyendo datos de accesibilidad...")
    df = pd.read_csv(input_file)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Create a copy to work with
    df_clean = df.copy()

    # Helper function to check if value is valid (not NaN, not None, not 0)
    def is_valid_value(value):
        return not pd.isna(value) and value is not None and value != 0

    # Process access time and distance
    print("\nProcesando tiempos y distancias de acceso...")
    df_clean['Tiempo_Acceso_Final'] = np.nan
    df_clean['Distancia_Acceso_Final'] = np.nan
    df_clean['Fuente_Acceso'] = ''

    calculated_access_count = 0
    mapped_access_count = 0

    for idx, row in df_clean.iterrows():
        # Check if calculated access values are available and valid
        if (is_valid_value(row.get('Access_Time_Minutes')) and
                is_valid_value(row.get('Access_Distance_Meters'))):
            df_clean.loc[idx, 'Tiempo_Acceso_Final'] = row['Access_Time_Minutes']
            df_clean.loc[idx, 'Distancia_Acceso_Final'] = row['Access_Distance_Meters']
            df_clean.loc[idx, 'Fuente_Acceso'] = 'Calculado'
            calculated_access_count += 1
        else:
            # Use mapped values (changed from Mapeada_* to Mejor_*)
            df_clean.loc[idx, 'Tiempo_Acceso_Final'] = row.get('Mejor_ACT', np.nan)
            df_clean.loc[idx, 'Distancia_Acceso_Final'] = row.get('Mejor_ACD', np.nan) * 1000  # Convert km to meters
            df_clean.loc[idx, 'Fuente_Acceso'] = 'Mapeado'
            mapped_access_count += 1

    print(f"Acceso - Calculados: {calculated_access_count}, Mapeados: {mapped_access_count}")

    # Process egress time and distance
    print("Procesando tiempos y distancias de salida...")
    df_clean['Tiempo_Salida_Final'] = np.nan
    df_clean['Distancia_Salida_Final'] = np.nan
    df_clean['Fuente_Salida'] = ''

    calculated_egress_count = 0
    mapped_egress_count = 0

    for idx, row in df_clean.iterrows():
        # Check if calculated egress values are available and valid
        if (is_valid_value(row.get('Egress_Time_Minutes')) and
                is_valid_value(row.get('Egress_Distance_Meters'))):
            df_clean.loc[idx, 'Tiempo_Salida_Final'] = row['Egress_Time_Minutes']
            df_clean.loc[idx, 'Distancia_Salida_Final'] = row['Egress_Distance_Meters']
            df_clean.loc[idx, 'Fuente_Salida'] = 'Calculado'
            calculated_egress_count += 1
        else:
            # Use mapped values (changed from Mapeada_* to Mejor_*)
            df_clean.loc[idx, 'Tiempo_Salida_Final'] = row.get('Mejor_EGT', np.nan)
            df_clean.loc[idx, 'Distancia_Salida_Final'] = row.get('Mejor_EGD', np.nan) * 1000  # Convert km to meters
            df_clean.loc[idx, 'Fuente_Salida'] = 'Mapeado'
            mapped_egress_count += 1

    print(f"Salida - Calculados: {calculated_egress_count}, Mapeados: {mapped_egress_count}")

    # Recalculate journey time when using calculated access/egress times
    # Formula: JRT (Journey Time) = RIT (Ride Time) + ACT (Access Time) + EGT (Egress Time)
    print("Recalculando tiempos de viaje...")
    df_clean['Tiempo_Viaje_Ajustado'] = np.nan

    recalculated_count = 0
    original_count = 0

    for idx, row in df_clean.iterrows():
        # Check if we used any calculated values
        used_calculated_access = row['Fuente_Acceso'] == 'Calculado'
        used_calculated_egress = row['Fuente_Salida'] == 'Calculado'

        if used_calculated_access or used_calculated_egress:
            # Get the ride time (RIT) from mapped data - this stays constant
            ride_time = row.get('Mejor_RIT', 0)

            # Get final access and egress times (already calculated above)
            new_access = row['Tiempo_Acceso_Final'] if not pd.isna(row['Tiempo_Acceso_Final']) else 0
            new_egress = row['Tiempo_Salida_Final'] if not pd.isna(row['Tiempo_Salida_Final']) else 0

            # Recalculated journey time = Ride Time + New Access Time + New Egress Time
            # JRT = RIT + ACT + EGT
            adjusted_time = ride_time + new_access + new_egress
            df_clean.loc[idx, 'Tiempo_Viaje_Ajustado'] = max(0, adjusted_time)  # Ensure non-negative
            recalculated_count += 1
        else:
            # Use original mapped journey time
            df_clean.loc[idx, 'Tiempo_Viaje_Ajustado'] = row.get('Mejor_JRT', np.nan)
            original_count += 1

    print(f"Tiempos de viaje - Recalculados: {recalculated_count}, Originales: {original_count}")

    # Create final cleaned dataframe with Spanish column names
    print("Creando dataset final con nombres en español...")

    cleaned_df = pd.DataFrame({
        'Zona_Origen': df_clean['Origin'],
        'Poblacion_Origen': df_clean.get('Origin_Population', np.nan),
        'Zona_Destino': df_clean['Destination'],
        'Usa_Transporte_Publico': df_clean.get('Uses_Public_Transport', True),
        'Parada_Origen': df_clean.get('FromStopPointNo', np.nan),
        'Parada_Destino': df_clean.get('ToStopPointNo', np.nan),

        # Final calculated/mapped values
        'Tiempo_Acceso_Minutos': df_clean['Tiempo_Acceso_Final'],
        'Tiempo_Salida_Minutos': df_clean['Tiempo_Salida_Final'],
        'Distancia_Acceso_Metros': df_clean['Distancia_Acceso_Final'],
        'Distancia_Salida_Metros': df_clean['Distancia_Salida_Final'],
        'Fuente_Datos_Acceso': df_clean['Fuente_Acceso'],
        'Fuente_Datos_Salida': df_clean['Fuente_Salida'],

        # Journey metrics - with correct interpretations (changed from Mapeada_* to Mejor_*)
        'Tiempo_Viaje_Total_Minutos': df_clean['Tiempo_Viaje_Ajustado'],
        # JRT: Journey Time (includes access + ride + egress)
        'Tiempo_Trayecto_Minutos': df_clean.get('Mejor_RIT', np.nan),  # RIT: Ride Time only (vehicle in motion)
        'Numero_Transbordos': df_clean.get('Mejor_NTR', np.nan),
        'Distancia_Viaje_Total_Km': df_clean.get('Mejor_JRD', np.nan),
        'Distancia_Trayecto_Km': df_clean.get('Mejor_RID', np.nan),
        'Tiempo_Espera_Transbordo_Minutos': df_clean.get('Mejor_TWT', np.nan),
        'Frecuencia_Servicio': df_clean.get('Mejor_SFQ', np.nan),

        # Original mapped values for reference (changed from Mapeada_* to Mejor_*)
        'Tiempo_Acceso_Original': df_clean.get('Mejor_ACT', np.nan),
        'Tiempo_Salida_Original': df_clean.get('Mejor_EGT', np.nan),
        'Distancia_Acceso_Original_Km': df_clean.get('Mejor_ACD', np.nan),
        'Distancia_Salida_Original_Km': df_clean.get('Mejor_EGD', np.nan),
        'Tiempo_Viaje_Original_Minutos': df_clean.get('Mejor_JRT', np.nan),
    })

    # Calculate total door-to-door time (should equal Tiempo_Viaje_Total_Minutos when properly calculated)
    print("Calculando tiempo total puerta a puerta...")
    # Note: For journeys with calculated access/egress, Tiempo_Viaje_Total_Minutos already includes
    # access + ride + egress, so we use that directly
    # For journeys with only mapped data, we need to sum the components

    cleaned_df['Tiempo_Total_Puerta_a_Puerta_Minutos'] = cleaned_df['Tiempo_Viaje_Total_Minutos'].copy()

    # Verification column: manual calculation for comparison
    cleaned_df['Verificacion_Tiempo_Total'] = (
            cleaned_df['Tiempo_Acceso_Minutos'].fillna(0) +
            cleaned_df['Tiempo_Trayecto_Minutos'].fillna(0) +
            cleaned_df['Tiempo_Salida_Minutos'].fillna(0) +
            cleaned_df['Tiempo_Espera_Transbordo_Minutos'].fillna(0)
    )

    # Replace 0 with NaN where appropriate
    for col in ['Tiempo_Acceso_Minutos', 'Tiempo_Salida_Minutos', 'Tiempo_Total_Puerta_a_Puerta_Minutos']:
        cleaned_df.loc[cleaned_df[col] == 0, col] = np.nan

    print(f"\nDataset limpio creado con {len(cleaned_df)} filas y {len(cleaned_df.columns)} columnas")

    # Save to file if output file specified
    if output_file:
        cleaned_df.to_csv(output_file, index=False)
        print(f"Datos guardados en: {output_file}")

    # Print summary statistics
    print_summary_statistics(cleaned_df)

    return cleaned_df


def print_summary_statistics(df):
    """Print summary statistics of the cleaned data"""
    print("\n=== RESUMEN ESTADÍSTICO ===")

    print(f"\nDatos generales:")
    print(f"Total de pares origen-destino: {len(df)}")
    print(f"Zonas de origen únicas: {df['Zona_Origen'].nunique()}")
    print(f"Zonas de destino únicas: {df['Zona_Destino'].nunique()}")

    # Data source breakdown
    print(f"\nFuentes de datos:")
    if 'Fuente_Datos_Acceso' in df.columns:
        access_sources = df['Fuente_Datos_Acceso'].value_counts()
        print("Datos de acceso:")
        for source, count in access_sources.items():
            print(f"  {source}: {count} ({count / len(df) * 100:.1f}%)")

    if 'Fuente_Datos_Salida' in df.columns:
        egress_sources = df['Fuente_Datos_Salida'].value_counts()
        print("Datos de salida:")
        for source, count in egress_sources.items():
            print(f"  {source}: {count} ({count / len(df) * 100:.1f}%)")

    # Time statistics
    print(f"\nEstadísticas de tiempo (minutos):")
    time_cols = ['Tiempo_Acceso_Minutos', 'Tiempo_Salida_Minutos',
                 'Tiempo_Viaje_Total_Minutos', 'Tiempo_Total_Puerta_a_Puerta_Minutos']

    for col in time_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(f"  {col}:")
                print(f"    Promedio: {valid_data.mean():.2f}")
                print(f"    Mediana: {valid_data.median():.2f}")
                print(f"    Rango: {valid_data.min():.2f} - {valid_data.max():.2f}")
                print(f"    Datos válidos: {len(valid_data)} ({len(valid_data) / len(df) * 100:.1f}%)")

    # Transport mode breakdown
    if 'Usa_Transporte_Publico' in df.columns:
        transport_modes = df['Usa_Transporte_Publico'].value_counts()
        print(f"\nModos de transporte:")
        for mode, count in transport_modes.items():
            mode_name = "Transporte público" if mode else "Solo caminata"
            print(f"  {mode_name}: {count} ({count / len(df) * 100:.1f}%)")


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = r"C:\Users\avoltan\PyCharmMiscProject\accessibility\accessibility_with_times_cruces.csv"
    output_file = "datos_accesibilidad_cruces.csv"

    try:
        # Clean the data
        cleaned_data = clean_accessibility_data(input_file, output_file)

        print(f"\n¡Proceso completado exitosamente!")
        print(f"Archivo de salida: {output_file}")

        # Show sample of cleaned data
        print(f"\nMuestra de datos limpios (primeras 3 filas):")
        sample_cols = ['Zona_Origen', 'Zona_Destino', 'Tiempo_Acceso_Minutos',
                       'Tiempo_Viaje_Total_Minutos', 'Tiempo_Salida_Minutos',
                       'Tiempo_Total_Puerta_a_Puerta_Minutos', 'Fuente_Datos_Acceso', 'Fuente_Datos_Salida']

        available_cols = [col for col in sample_cols if col in cleaned_data.columns]
        print(cleaned_data[available_cols].head(3).to_string(index=False))

    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {input_file}")
        print("Por favor, verifica la ruta del archivo.")
    except Exception as e:
        print(f"Error procesando los datos: {e}")
        import traceback

        traceback.print_exc()