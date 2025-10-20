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
       New_Journey_Time = Mejor_RIT + Final_Access_Time + Final_Egress_Time
       (Note: RIT already includes in-vehicle time, TWT is separate if needed)

    4. **NEW: Recalculate journey distance when using calculated access/egress distances:**
       New_Journey_Distance = (Final_Access_Distance + Final_Egress_Distance) / 1000 + Mejor_RID
       (Note: RID already includes distance during transfers)

    5. Rename columns to Spanish with clear names
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
            # Use mapped values
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
            # Use mapped values
            df_clean.loc[idx, 'Tiempo_Salida_Final'] = row.get('Mejor_EGT', np.nan)
            df_clean.loc[idx, 'Distancia_Salida_Final'] = row.get('Mejor_EGD', np.nan) * 1000  # Convert km to meters
            df_clean.loc[idx, 'Fuente_Salida'] = 'Mapeado'
            mapped_egress_count += 1

    print(f"Salida - Calculados: {calculated_egress_count}, Mapeados: {mapped_egress_count}")

    # Recalculate journey time when using calculated access/egress times
    print("Recalculando tiempos de viaje...")
    df_clean['Tiempo_Viaje_Ajustado'] = np.nan

    recalculated_time_count = 0
    original_time_count = 0

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
            # Note: TWT (transfer wait time) is already included in RIT
            adjusted_time = ride_time + new_access + new_egress
            df_clean.loc[idx, 'Tiempo_Viaje_Ajustado'] = max(0, adjusted_time)
            recalculated_time_count += 1
        else:
            # Use original mapped journey time
            df_clean.loc[idx, 'Tiempo_Viaje_Ajustado'] = row.get('Mejor_JRT', np.nan)
            original_time_count += 1

    print(f"Tiempos de viaje - Recalculados: {recalculated_time_count}, Originales: {original_time_count}")

    # **NEW: Recalculate journey distance when using calculated access/egress distances**
    print("Recalculando distancias de viaje...")
    df_clean['Distancia_Viaje_Ajustada'] = np.nan

    recalculated_distance_count = 0
    original_distance_count = 0

    for idx, row in df_clean.iterrows():
        # Check if we used any calculated values
        used_calculated_access = row['Fuente_Acceso'] == 'Calculado'
        used_calculated_egress = row['Fuente_Salida'] == 'Calculado'

        if used_calculated_access or used_calculated_egress:
            # Get the ride distance (RID) from mapped data - this stays constant (in km)
            ride_distance_km = row.get('Mejor_RID', 0)

            # Get final access and egress distances in meters (already calculated above)
            new_access_m = row['Distancia_Acceso_Final'] if not pd.isna(row['Distancia_Acceso_Final']) else 0
            new_egress_m = row['Distancia_Salida_Final'] if not pd.isna(row['Distancia_Salida_Final']) else 0

            # Convert access and egress from meters to kilometers
            new_access_km = new_access_m / 1000
            new_egress_km = new_egress_m / 1000

            # Recalculated journey distance = Access + Ride + Egress (all in km)
            # JRD = ACD + RID + EGD
            # Note: RID already includes distance covered during transfers
            adjusted_distance = new_access_km + ride_distance_km + new_egress_km
            df_clean.loc[idx, 'Distancia_Viaje_Ajustada'] = max(0, adjusted_distance)
            recalculated_distance_count += 1
        else:
            # Use original mapped journey distance
            df_clean.loc[idx, 'Distancia_Viaje_Ajustada'] = row.get('Mejor_JRD', np.nan)
            original_distance_count += 1

    print(f"Distancias de viaje - Recalculadas: {recalculated_distance_count}, Originales: {original_distance_count}")

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

        # Journey metrics - with correct interpretations
        'Tiempo_Viaje_Total_Minutos': df_clean['Tiempo_Viaje_Ajustado'],
        'Tiempo_Trayecto_Minutos': df_clean.get('Mejor_RIT', np.nan),
        # RIT: Ride Time (includes in-vehicle + transfer wait)
        'Distancia_Viaje_Total_Km': df_clean['Distancia_Viaje_Ajustada'],  # **NOW RECALCULATED**
        'Distancia_Trayecto_Km': df_clean.get('Mejor_RID', np.nan),
        # RID: Ride Distance (includes distance during transfers)

        'Numero_Transbordos': df_clean.get('Mejor_NTR', np.nan),
        'Tiempo_Espera_Transbordo_Minutos': df_clean.get('Mejor_TWT', np.nan),
        'Frecuencia_Servicio': df_clean.get('Mejor_SFQ', np.nan),

        # Original mapped values for reference
        'Tiempo_Acceso_Original': df_clean.get('Mejor_ACT', np.nan),
        'Tiempo_Salida_Original': df_clean.get('Mejor_EGT', np.nan),
        'Distancia_Acceso_Original_Km': df_clean.get('Mejor_ACD', np.nan),
        'Distancia_Salida_Original_Km': df_clean.get('Mejor_EGD', np.nan),
        'Tiempo_Viaje_Original_Minutos': df_clean.get('Mejor_JRT', np.nan),
        'Distancia_Viaje_Original_Km': df_clean.get('Mejor_JRD', np.nan),  # **ADDED**
    })

    # Calculate verification columns
    print("Calculando columnas de verificación...")

    # Time verification (should equal Tiempo_Viaje_Total_Minutos)
    cleaned_df['Verificacion_Tiempo_Total'] = (
            cleaned_df['Tiempo_Acceso_Minutos'].fillna(0) +
            cleaned_df['Tiempo_Trayecto_Minutos'].fillna(0) +
            cleaned_df['Tiempo_Salida_Minutos'].fillna(0)
    )

    # **NEW: Distance verification (should equal Distancia_Viaje_Total_Km)**
    cleaned_df['Verificacion_Distancia_Total_Km'] = (
            cleaned_df['Distancia_Acceso_Metros'].fillna(0) / 1000 +  # Convert meters to km
            cleaned_df['Distancia_Trayecto_Km'].fillna(0) +
            cleaned_df['Distancia_Salida_Metros'].fillna(0) / 1000  # Convert meters to km
    )

    # Replace 0 with NaN where appropriate
    for col in ['Tiempo_Acceso_Minutos', 'Tiempo_Salida_Minutos', 'Tiempo_Viaje_Total_Minutos',
                'Distancia_Acceso_Metros', 'Distancia_Salida_Metros', 'Distancia_Viaje_Total_Km']:
        if col in cleaned_df.columns:
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
                 'Tiempo_Trayecto_Minutos', 'Tiempo_Viaje_Total_Minutos']

    for col in time_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(f"  {col}:")
                print(f"    Promedio: {valid_data.mean():.2f}")
                print(f"    Mediana: {valid_data.median():.2f}")
                print(f"    Rango: {valid_data.min():.2f} - {valid_data.max():.2f}")
                print(f"    Datos válidos: {len(valid_data)} ({len(valid_data) / len(df) * 100:.1f}%)")

    # **NEW: Distance statistics**
    print(f"\nEstadísticas de distancia:")
    distance_cols = ['Distancia_Acceso_Metros', 'Distancia_Salida_Metros',
                     'Distancia_Trayecto_Km', 'Distancia_Viaje_Total_Km']

    for col in distance_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                unit = "metros" if "Metros" in col else "km"
                print(f"  {col}:")
                print(f"    Promedio: {valid_data.mean():.2f} {unit}")
                print(f"    Mediana: {valid_data.median():.2f} {unit}")
                print(f"    Rango: {valid_data.min():.2f} - {valid_data.max():.2f} {unit}")
                print(f"    Datos válidos: {len(valid_data)} ({len(valid_data) / len(df) * 100:.1f}%)")

    # Transport mode breakdown
    if 'Usa_Transporte_Publico' in df.columns:
        transport_modes = df['Usa_Transporte_Publico'].value_counts()
        print(f"\nModos de transporte:")
        for mode, count in transport_modes.items():
            mode_name = "Transporte público" if mode else "Solo caminata"
            print(f"  {mode_name}: {count} ({count / len(df) * 100:.1f}%)")

    # **NEW: Verification statistics**
    print(f"\nVerificación de cálculos:")

    # Check time verification
    if 'Verificacion_Tiempo_Total' in df.columns and 'Tiempo_Viaje_Total_Minutos' in df.columns:
        time_diff = (df['Tiempo_Viaje_Total_Minutos'] - df['Verificacion_Tiempo_Total']).abs()
        valid_time_diff = time_diff.dropna()
        if len(valid_time_diff) > 0:
            print(f"  Diferencia tiempo (absoluta):")
            print(f"    Promedio: {valid_time_diff.mean():.4f} minutos")
            print(f"    Máxima: {valid_time_diff.max():.4f} minutos")
            matches = (valid_time_diff < 0.01).sum()
            print(
                f"    Coincidencias exactas: {matches}/{len(valid_time_diff)} ({matches / len(valid_time_diff) * 100:.1f}%)")

    # Check distance verification
    if 'Verificacion_Distancia_Total_Km' in df.columns and 'Distancia_Viaje_Total_Km' in df.columns:
        distance_diff = (df['Distancia_Viaje_Total_Km'] - df['Verificacion_Distancia_Total_Km']).abs()
        valid_distance_diff = distance_diff.dropna()
        if len(valid_distance_diff) > 0:
            print(f"  Diferencia distancia (absoluta):")
            print(f"    Promedio: {valid_distance_diff.mean():.4f} km")
            print(f"    Máxima: {valid_distance_diff.max():.4f} km")
            matches = (valid_distance_diff < 0.001).sum()
            print(
                f"    Coincidencias exactas: {matches}/{len(valid_distance_diff)} ({matches / len(valid_distance_diff) * 100:.1f}%)")


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = r"C:\Users\avoltan\PyCharmMiscProject\accessibility\results\accessibility_with_times_leioa.csv"
    output_file = "results/datos_accesibilidad_leioa.csv"

    try:
        # Clean the data
        cleaned_data = clean_accessibility_data(input_file, output_file)

        print(f"\n¡Proceso completado exitosamente!")
        print(f"Archivo de salida: {output_file}")

        # Show sample of cleaned data
        print(f"\nMuestra de datos limpios (primeras 3 filas):")
        sample_cols = ['Zona_Origen', 'Zona_Destino',
                       'Tiempo_Acceso_Minutos', 'Tiempo_Trayecto_Minutos', 'Tiempo_Salida_Minutos',
                       'Tiempo_Viaje_Total_Minutos', 'Verificacion_Tiempo_Total',
                       'Distancia_Acceso_Metros', 'Distancia_Trayecto_Km', 'Distancia_Salida_Metros',
                       'Distancia_Viaje_Total_Km', 'Verificacion_Distancia_Total_Km',
                       'Fuente_Datos_Acceso', 'Fuente_Datos_Salida']

        available_cols = [col for col in sample_cols if col in cleaned_data.columns]
        print(cleaned_data[available_cols].head(3).to_string(index=False))

    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {input_file}")
        print("Por favor, verifica la ruta del archivo.")
    except Exception as e:
        print(f"Error procesando los datos: {e}")
        import traceback

        traceback.print_exc()