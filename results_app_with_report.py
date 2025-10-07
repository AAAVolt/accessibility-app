import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from io import BytesIO
from datetime import datetime
import zipfile
import plotly.io as pio


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_excel_report(df_filtered, zone_populations, total_population, poi_df, poor_zones, df_all=None,
                        zone_need=None):
    """Generate an Excel workbook with key metrics and tables"""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # ===== SHEET 1: KEY METRICS =====
        zones_under_30 = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] <= 30]['Zona_Origen'].unique()
        pop_under_30 = zone_populations[zone_populations.index.isin(zones_under_30)].sum()
        under_30_pct = (pop_under_30 / total_population * 100) if total_population > 0 else 0

        # Calculate accessibility summary
        zone_best_access = df_filtered.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'min',
            'Poblacion_Origen': 'first'
        }).reset_index()

        zone_best_access['Best_Category'] = pd.cut(
            zone_best_access['Tiempo_Viaje_Total_Minutos'],
            bins=[0, 30, 45, 60, float('inf')],
            labels=['Excellent (<30min)', 'Good (30-45min)', 'Fair (45-60min)', 'Poor (>60min)']
        )

        pop_by_category = zone_best_access.groupby('Best_Category')['Poblacion_Origen'].sum()
        category_pcts = (pop_by_category / total_population * 100) if total_population > 0 else pd.Series()

        # Calculate trip need statistics if available
        if df_all is not None and 'Necesita_viaje' in df_all.columns and zone_need is not None:
            zones_with_need = zone_need['Necesita_viaje'].sum()
            total_zones = len(zone_need)
            zones_need_pct = (zones_with_need / total_zones * 100) if total_zones > 0 else 0

            pop_with_need = zone_need[zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
            total_pop = zone_need['Poblacion_Origen'].sum()
            pop_with_need_pct = (pop_with_need / total_pop * 100) if total_pop > 0 else 0

            pois_with_need = df_all[df_all['Necesita_viaje'] == 1]['Zona_Destino'].nunique()
            total_pois_all = df_all['Zona_Destino'].nunique()
            pois_need_pct = (pois_with_need / total_pois_all * 100) if total_pois_all > 0 else 0
        else:
            zones_need_pct = None
            pop_with_need_pct = None
            pois_need_pct = None

        key_metrics_data = {
            'Metric': [
                'Average Travel Time (min)',
                'Average Transfers',
                'Population with <30min access (%)',
                'Total POIs',
                '',
                'ACCESSIBILITY SUMMARY',
                'Excellent (<30min) - Population',
                'Excellent (<30min) - Percentage',
                'Good (30-45min) - Population',
                'Good (30-45min) - Percentage',
                'Fair (45-60min) - Population',
                'Fair (45-60min) - Percentage',
                'Poor (>60min) - Population',
                'Poor (>60min) - Percentage',
            ],
            'Value': [
                f"{df_filtered['Tiempo_Viaje_Total_Minutos'].mean():.1f}",
                f"{df_filtered['Numero_Transbordos'].mean():.2f}",
                f"{under_30_pct:.1f}",
                df_filtered['Zona_Destino'].nunique(),
                '',
                '',
                f"{pop_by_category.get('Excellent (<30min)', 0):,.0f}",
                f"{category_pcts.get('Excellent (<30min)', 0):.1f}%",
                f"{pop_by_category.get('Good (30-45min)', 0):,.0f}",
                f"{category_pcts.get('Good (30-45min)', 0):.1f}%",
                f"{pop_by_category.get('Fair (45-60min)', 0):,.0f}",
                f"{category_pcts.get('Fair (45-60min)', 0):.1f}%",
                f"{pop_by_category.get('Poor (>60min)', 0):,.0f}",
                f"{category_pcts.get('Poor (>60min)', 0):.1f}%",
            ]
        }

        # Add trip need metrics if available
        if zones_need_pct is not None:
            key_metrics_data['Metric'].extend([
                '',
                'TRIP NEED STATISTICS',
                'Zones with Need of Travel with BizkaiBus (%)',
                'Population with Need to travel with Bizkaibus (%)',
                'POIs with need (%)'
            ])
            key_metrics_data['Value'].extend([
                '',
                '',
                f"{zones_need_pct:.1f}",
                f"{pop_with_need_pct:.1f}",
                f"{pois_need_pct:.1f}"
            ])

        pd.DataFrame(key_metrics_data).to_excel(writer, sheet_name='Key Metrics', index=False)

        # ===== SHEET 2: POI STATISTICS =====
        poi_df.to_excel(writer, sheet_name='POI Statistics', index=False)

        # ===== SHEET 3: MOST CRITICAL ZONES =====
        if len(poor_zones) > 0:
            critical_zones = poor_zones[['Zone Name', 'Population', 'Avg Travel Time']].head(15).copy()
            critical_zones.columns = ['Zone Name', 'Residents', 'Avg Travel Time (min)']
            critical_zones.to_excel(writer, sheet_name='Most Critical Zones', index=False)

    output.seek(0)
    return output


def create_all_charts(df_filtered, zone_populations, total_population, poi_df, poor_zones):
    """Generate all chart images and return as dictionary"""
    charts = {}

    # Calculate common data needed for multiple charts
    zone_best_access = df_filtered.groupby('Zona_Origen').agg({
        'Tiempo_Viaje_Total_Minutos': 'min',
        'Poblacion_Origen': 'first'
    }).reset_index()

    zone_best_access['Best_Category'] = pd.cut(
        zone_best_access['Tiempo_Viaje_Total_Minutos'],
        bins=[0, 30, 45, 60, float('inf')],
        labels=['🟢 Excellent (<30min)', '🟡 Good (30-45min)',
                '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
    )

    pop_by_category = zone_best_access.groupby('Best_Category')['Poblacion_Origen'].sum()
    category_pcts = (pop_by_category / total_population * 100) if total_population > 0 else pd.Series()

    categories = ['🟢 Excellent (<30min)', '🟡 Good (30-45min)', '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
    populations = [pop_by_category.get(cat, 0) for cat in categories]
    percentages = [category_pcts.get(cat, 0) for cat in categories]

    # CHART 1: Population by Accessibility Category
    fig1 = px.bar(
        x=categories,
        y=populations,
        color=categories,
        color_discrete_map={
            '🟢 Excellent (<30min)': '#2ecc71',
            '🟡 Good (30-45min)': '#f1c40f',
            '🟠 Fair (45-60min)': '#e67e22',
            '🔴 Poor (>60min)': '#e74c3c'
        },
        text=[f"{int(pop):,}<br><b>{pct:.1f}%</b>" for pop, pct in zip(populations, percentages)],
        title="Distribution of Population by Best Accessibility to Any POI"
    )
    fig1.update_traces(textposition='outside', textfont_size=10)
    fig1.update_layout(
        xaxis_title="Accessibility Category",
        yaxis_title="Population (residents)",
        showlegend=False,
        height=500,
        width=1200,
        font=dict(size=10),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9))
    )
    charts['01_population_accessibility_bar.png'] = pio.to_image(fig1, format='png', width=1200, height=500)

    # CHART 2: Travel Time Distribution (Population-Weighted)
    bins = np.linspace(df_filtered['Tiempo_Viaje_Total_Minutos'].min(),
                       df_filtered['Tiempo_Viaje_Total_Minutos'].max(), 41)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_populations = []
    for i in range(len(bins) - 1):
        mask = (df_filtered['Tiempo_Viaje_Total_Minutos'] >= bins[i]) & \
               (df_filtered['Tiempo_Viaje_Total_Minutos'] < bins[i + 1])
        zones_in_bin = df_filtered[mask]['Zona_Origen'].unique()
        pop_in_bin = zone_populations[zone_populations.index.isin(zones_in_bin)].sum()
        bin_populations.append(pop_in_bin / total_population * 100)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bin_centers,
        y=bin_populations,
        marker_color='#EBEBEB',
        width=(bins[1] - bins[0]) * 0.9,
        showlegend=False
    ))

    travel_times_list = []
    for _, row in df_filtered.iterrows():
        travel_times_list.extend([row['Tiempo_Viaje_Total_Minutos']] * int(row['Poblacion_Origen'] / 100))

    if len(travel_times_list) > 0:
        travel_times_weighted = np.array(travel_times_list)
        kde = stats.gaussian_kde(travel_times_weighted)
        x_range = np.linspace(df_filtered['Tiempo_Viaje_Total_Minutos'].min(),
                              df_filtered['Tiempo_Viaje_Total_Minutos'].max(), 200)
        y_kde = kde(x_range)
        y_kde_scaled = y_kde * max(bin_populations) / y_kde.max() * 0.8

        fig2.add_trace(go.Scatter(
            x=x_range,
            y=y_kde_scaled,
            mode='lines',
            line=dict(color='#C00000', width=3),
            showlegend=False
        ))

        pop_under_30_pct = category_pcts.get('🟢 Excellent (<30min)', 0)
        pop_under_45_pct = pop_under_30_pct + category_pcts.get('🟡 Good (30-45min)', 0)
        pop_under_60_pct = pop_under_45_pct + category_pcts.get('🟠 Fair (45-60min)', 0)

        fig2.add_vline(x=30, line_dash="dash", line_color="green",
                       annotation_text=f"30 min ({pop_under_30_pct:.1f}%)", annotation_font_size=9)
        fig2.add_vline(x=45, line_dash="dash", line_color="orange",
                       annotation_text=f"45 min ({pop_under_45_pct:.1f}%)", annotation_font_size=9)
        fig2.add_vline(x=60, line_dash="dash", line_color="red",
                       annotation_text=f"60 min ({pop_under_60_pct:.1f}%)", annotation_font_size=9)

    fig2.update_layout(
        title="Travel Time Distribution (Population-Weighted)",
        xaxis_title="Total Travel Time (minutes)",
        yaxis_title="Percentage of Population (%)",
        height=400,
        width=1200,
        font=dict(size=10),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9))
    )
    charts['02_travel_time_population_weighted.png'] = pio.to_image(fig2, format='png', width=1200, height=400)

    # CHART 3: Travel Time Distribution by Zones
    fig3 = px.histogram(
        zone_best_access,
        x='Tiempo_Viaje_Total_Minutos',
        nbins=40,
        histnorm='percent',
        color_discrete_sequence=['#EBEBEB'],
        title="Travel Time Distribution by Zones"
    )

    travel_times_zones = zone_best_access['Tiempo_Viaje_Total_Minutos'].dropna()
    kde = stats.gaussian_kde(travel_times_zones)
    x_range = np.linspace(travel_times_zones.min(), travel_times_zones.max(), 200)
    y_kde = kde(x_range)
    y_kde_normalized = y_kde * 100 * (travel_times_zones.max() - travel_times_zones.min()) / 40

    fig3.add_trace(go.Scatter(
        x=x_range,
        y=y_kde_normalized,
        mode='lines',
        line=dict(color='#C00000', width=3),
        showlegend=False
    ))

    total_zones = len(zone_best_access)
    zones_under_30_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 30).sum() / total_zones * 100
    zones_under_45_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 45).sum() / total_zones * 100
    zones_under_60_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 60).sum() / total_zones * 100

    fig3.add_vline(x=30, line_dash="dash", line_color="green",
                   annotation_text=f"30 min ({zones_under_30_pct:.1f}%)", annotation_font_size=9)
    fig3.add_vline(x=45, line_dash="dash", line_color="orange",
                   annotation_text=f"45 min ({zones_under_45_pct:.1f}%)", annotation_font_size=9)
    fig3.add_vline(x=60, line_dash="dash", line_color="red",
                   annotation_text=f"60 min ({zones_under_60_pct:.1f}%)", annotation_font_size=9)

    fig3.update_layout(
        xaxis_title="Total Travel Time (minutes)",
        yaxis_title="Percentage of Zones (%)",
        height=400,
        width=1200,
        font=dict(size=10),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9))
    )
    charts['03_travel_time_by_zones.png'] = pio.to_image(fig3, format='png', width=1200, height=400)

    # CHART 4: Best Connected POIs
    best_pois = poi_df.head(5)
    fig4 = px.bar(
        best_pois,
        x='Avg Time (weighted)',
        y='POI',
        orientation='h',
        color='Excellent (%)',
        color_continuous_scale='Greens',
        text='Avg Time (weighted)',
        title="Best Connected POIs"
    )
    fig4.update_traces(texttemplate='%{text:.1f} min', textposition='outside', textfont_size=9)
    fig4.update_layout(
        height=300,
        yaxis={'categoryorder': 'total ascending'},
        width=1200,
        font=dict(size=10),
        xaxis=dict(tickfont=dict(size=9))
    )
    charts['04_best_connected_pois.png'] = pio.to_image(fig4, format='png', width=1200, height=300)

    # CHART 5: Worst Connected POIs
    worst_pois = poi_df.tail(5)
    fig5 = px.bar(
        worst_pois,
        x='Avg Time (weighted)',
        y='POI',
        orientation='h',
        color='Poor (%)',
        color_continuous_scale='Reds',
        text='Avg Time (weighted)',
        title="Worst Connected POIs"
    )
    fig5.update_traces(texttemplate='%{text:.1f} min', textposition='outside', textfont_size=9)
    fig5.update_layout(
        height=300,
        yaxis={'categoryorder': 'total descending'},
        width=1200,
        font=dict(size=10),
        xaxis=dict(tickfont=dict(size=9))
    )
    charts['05_worst_connected_pois.png'] = pio.to_image(fig5, format='png', width=1200, height=300)

    # CHART 6: POI Accessibility Breakdown
    poi_df_sorted = poi_df.sort_values('Avg Time (weighted)')

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        name='🟢 Excellent (<30min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Excellent (%)'],
        orientation='h',
        marker_color='#2ecc71',
        text=poi_df_sorted['Excellent (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%',
        textfont_size=8
    ))
    fig6.add_trace(go.Bar(
        name='🟡 Good (30-45min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Good (%)'],
        orientation='h',
        marker_color='#f1c40f',
        text=poi_df_sorted['Good (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%',
        textfont_size=8
    ))
    fig6.add_trace(go.Bar(
        name='🟠 Fair (45-60min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Fair (%)'],
        orientation='h',
        marker_color='#e67e22',
        text=poi_df_sorted['Fair (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%',
        textfont_size=8
    ))
    fig6.add_trace(go.Bar(
        name='🔴 Poor (>60min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Poor (%)'],
        orientation='h',
        marker_color='#e74c3c',
        text=poi_df_sorted['Poor (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%',
        textfont_size=8
    ))
    fig6.update_layout(
        barmode='stack',
        title='POI Accessibility Quality Distribution (Population-Weighted)',
        xaxis_title='Percentage of Population (%)',
        yaxis_title='Destination POI',
        height=max(400, len(poi_df) * 25),
        width=1200,
        font=dict(size=9),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        legend=dict(font=dict(size=8))
    )
    charts['06_poi_accessibility_breakdown.png'] = pio.to_image(fig6, format='png', width=1200,
                                                                height=max(400, len(poi_df) * 25))

    # CHART 7: Priority Zones (if any exist)
    if len(poor_zones) > 0:
        top_poor = poor_zones.head(15)
        fig7 = px.bar(
            top_poor,
            x='Population',
            y='Zone Name',
            orientation='h',
            color='Avg Travel Time',
            color_continuous_scale='Reds',
            hover_data=['Avg Transfers', 'POIs Served'],
            text='Population',
            title="High-Population Zones with Worst Accessibility"
        )
        fig7.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont_size=9)
        fig7.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Population (residents)',
            yaxis_title='Origin Zone',
            width=1200,
            font=dict(size=10),
            xaxis=dict(tickfont=dict(size=9)),
        )
        charts['07_priority_zones.png'] = pio.to_image(fig7, format='png', width=1200, height=500)

    return charts
    """Generate all chart images and return as dictionary"""
    charts = {}

    # Calculate common data needed for multiple charts
    zone_best_access = df_filtered.groupby('Zona_Origen').agg({
        'Tiempo_Viaje_Total_Minutos': 'min',
        'Poblacion_Origen': 'first'
    }).reset_index()

    zone_best_access['Best_Category'] = pd.cut(
        zone_best_access['Tiempo_Viaje_Total_Minutos'],
        bins=[0, 30, 45, 60, float('inf')],
        labels=['🟢 Excellent (<30min)', '🟡 Good (30-45min)',
                '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
    )

    pop_by_category = zone_best_access.groupby('Best_Category')['Poblacion_Origen'].sum()
    category_pcts = (pop_by_category / total_population * 100) if total_population > 0 else pd.Series()

    categories = ['🟢 Excellent (<30min)', '🟡 Good (30-45min)', '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
    populations = [pop_by_category.get(cat, 0) for cat in categories]
    percentages = [category_pcts.get(cat, 0) for cat in categories]

    # CHART 1: Population by Accessibility Category
    fig1 = px.bar(
        x=categories,
        y=populations,
        color=categories,
        color_discrete_map={
            '🟢 Excellent (<30min)': '#2ecc71',
            '🟡 Good (30-45min)': '#f1c40f',
            '🟠 Fair (45-60min)': '#e67e22',
            '🔴 Poor (>60min)': '#e74c3c'
        },
        text=[f"{int(pop):,}<br><b>{pct:.1f}%</b>" for pop, pct in zip(populations, percentages)],
        title="Distribution of Population by Best Accessibility to Any POI"
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(xaxis_title="Accessibility Category", yaxis_title="Population (residents)",
                       showlegend=False, height=500, width=1200)
    charts['01_population_accessibility_bar.png'] = pio.to_image(fig1, format='png', width=1200, height=500)

    # CHART 2: Travel Time Distribution (Population-Weighted)
    bins = np.linspace(df_filtered['Tiempo_Viaje_Total_Minutos'].min(),
                       df_filtered['Tiempo_Viaje_Total_Minutos'].max(), 41)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_populations = []
    for i in range(len(bins) - 1):
        mask = (df_filtered['Tiempo_Viaje_Total_Minutos'] >= bins[i]) & \
               (df_filtered['Tiempo_Viaje_Total_Minutos'] < bins[i + 1])
        zones_in_bin = df_filtered[mask]['Zona_Origen'].unique()
        pop_in_bin = zone_populations[zone_populations.index.isin(zones_in_bin)].sum()
        bin_populations.append(pop_in_bin / total_population * 100)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bin_centers,
        y=bin_populations,
        marker_color='#EBEBEB',
        width=(bins[1] - bins[0]) * 0.9,
        showlegend=False
    ))

    travel_times_list = []
    for _, row in df_filtered.iterrows():
        travel_times_list.extend([row['Tiempo_Viaje_Total_Minutos']] * int(row['Poblacion_Origen'] / 100))

    if len(travel_times_list) > 0:
        travel_times_weighted = np.array(travel_times_list)
        kde = stats.gaussian_kde(travel_times_weighted)
        x_range = np.linspace(df_filtered['Tiempo_Viaje_Total_Minutos'].min(),
                              df_filtered['Tiempo_Viaje_Total_Minutos'].max(), 200)
        y_kde = kde(x_range)
        y_kde_scaled = y_kde * max(bin_populations) / y_kde.max() * 0.8

        fig2.add_trace(go.Scatter(
            x=x_range,
            y=y_kde_scaled,
            mode='lines',
            line=dict(color='#C00000', width=3),
            showlegend=False
        ))

        pop_under_30_pct = category_pcts.get('🟢 Excellent (<30min)', 0)
        pop_under_45_pct = pop_under_30_pct + category_pcts.get('🟡 Good (30-45min)', 0)
        pop_under_60_pct = pop_under_45_pct + category_pcts.get('🟠 Fair (45-60min)', 0)

        fig2.add_vline(x=30, line_dash="dash", line_color="green",
                       annotation_text=f"30 min ({pop_under_30_pct:.1f}%)")
        fig2.add_vline(x=45, line_dash="dash", line_color="orange",
                       annotation_text=f"45 min ({pop_under_45_pct:.1f}%)")
        fig2.add_vline(x=60, line_dash="dash", line_color="red",
                       annotation_text=f"60 min ({pop_under_60_pct:.1f}%)")

    fig2.update_layout(
        title="Travel Time Distribution (Population-Weighted)",
        xaxis_title="Total Travel Time (minutes)",
        yaxis_title="Percentage of Population (%)",
        height=400, width=1200
    )
    charts['02_travel_time_population_weighted.png'] = pio.to_image(fig2, format='png', width=1200, height=400)

    # CHART 3: Travel Time Distribution by Zones
    fig3 = px.histogram(
        zone_best_access,
        x='Tiempo_Viaje_Total_Minutos',
        nbins=40,
        histnorm='percent',
        color_discrete_sequence=['#EBEBEB'],
        title="Travel Time Distribution by Zones"
    )

    travel_times_zones = zone_best_access['Tiempo_Viaje_Total_Minutos'].dropna()
    kde = stats.gaussian_kde(travel_times_zones)
    x_range = np.linspace(travel_times_zones.min(), travel_times_zones.max(), 200)
    y_kde = kde(x_range)
    y_kde_normalized = y_kde * 100 * (travel_times_zones.max() - travel_times_zones.min()) / 40

    fig3.add_trace(go.Scatter(
        x=x_range,
        y=y_kde_normalized,
        mode='lines',
        line=dict(color='#C00000', width=3),
        showlegend=False
    ))

    total_zones = len(zone_best_access)
    zones_under_30_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 30).sum() / total_zones * 100
    zones_under_45_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 45).sum() / total_zones * 100
    zones_under_60_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 60).sum() / total_zones * 100

    fig3.add_vline(x=30, line_dash="dash", line_color="green",
                   annotation_text=f"30 min ({zones_under_30_pct:.1f}% zones)")
    fig3.add_vline(x=45, line_dash="dash", line_color="orange",
                   annotation_text=f"45 min ({zones_under_45_pct:.1f}% zones)")
    fig3.add_vline(x=60, line_dash="dash", line_color="red",
                   annotation_text=f"60 min ({zones_under_60_pct:.1f}% zones)")

    fig3.update_layout(
        xaxis_title="Total Travel Time (minutes)",
        yaxis_title="Percentage of Zones (%)",
        height=400, width=1200
    )
    charts['03_travel_time_by_zones.png'] = pio.to_image(fig3, format='png', width=1200, height=400)

    # CHART 4: Best Connected POIs
    best_pois = poi_df.head(5)
    fig4 = px.bar(
        best_pois,
        x='Avg Time (weighted)',
        y='POI',
        orientation='h',
        color='Excellent (%)',
        color_continuous_scale='Greens',
        text='Avg Time (weighted)',
        title="Best Connected POIs"
    )
    fig4.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
    fig4.update_layout(height=300, yaxis={'categoryorder': 'total ascending'}, width=1200)
    charts['04_best_connected_pois.png'] = pio.to_image(fig4, format='png', width=1200, height=300)

    # CHART 5: Worst Connected POIs
    worst_pois = poi_df.tail(5)
    fig5 = px.bar(
        worst_pois,
        x='Avg Time (weighted)',
        y='POI',
        orientation='h',
        color='Poor (%)',
        color_continuous_scale='Reds',
        text='Avg Time (weighted)',
        title="Worst Connected POIs"
    )
    fig5.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
    fig5.update_layout(height=300, yaxis={'categoryorder': 'total descending'}, width=1200)
    charts['05_worst_connected_pois.png'] = pio.to_image(fig5, format='png', width=1200, height=300)

    # CHART 6: POI Accessibility Breakdown
    poi_df_sorted = poi_df.sort_values('Avg Time (weighted)')

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        name='🟢 Excellent (<30min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Excellent (%)'],
        orientation='h',
        marker_color='#2ecc71',
        text=poi_df_sorted['Excellent (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%'
    ))
    fig6.add_trace(go.Bar(
        name='🟡 Good (30-45min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Good (%)'],
        orientation='h',
        marker_color='#f1c40f',
        text=poi_df_sorted['Good (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%'
    ))
    fig6.add_trace(go.Bar(
        name='🟠 Fair (45-60min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Fair (%)'],
        orientation='h',
        marker_color='#e67e22',
        text=poi_df_sorted['Fair (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%'
    ))
    fig6.add_trace(go.Bar(
        name='🔴 Poor (>60min)',
        y=poi_df_sorted['POI'],
        x=poi_df_sorted['Poor (%)'],
        orientation='h',
        marker_color='#e74c3c',
        text=poi_df_sorted['Poor (%)'].round(1),
        textposition='inside',
        texttemplate='%{text:.0f}%'
    ))
    fig6.update_layout(
        barmode='stack',
        title='POI Accessibility Quality Distribution (Population-Weighted)',
        xaxis_title='Percentage of Population (%)',
        yaxis_title='Destination POI',
        height=max(400, len(poi_df) * 25),
        width=1200
    )
    charts['06_poi_accessibility_breakdown.png'] = pio.to_image(fig6, format='png', width=1200,
                                                                height=max(400, len(poi_df) * 25))

    # CHART 7: Priority Zones (if any exist)
    if len(poor_zones) > 0:
        top_poor = poor_zones.head(15)
        fig7 = px.bar(
            top_poor,
            x='Population',
            y='Zone Name',
            orientation='h',
            color='Avg Travel Time',
            color_continuous_scale='Reds',
            hover_data=['Avg Transfers', 'POIs Served'],
            text='Population',
            title="High-Population Zones with Worst Accessibility"
        )
        fig7.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig7.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Population (residents)',
            yaxis_title='Origin Zone',
            width=1200
        )
        charts['07_priority_zones.png'] = pio.to_image(fig7, format='png', width=1200, height=500)

    return charts


# ============================================================================
# MAIN APP
# ============================================================================

st.set_page_config(page_title="Bizkaia Accessibility Analysis", layout="wide", page_icon="🚌")

# Title and introduction
st.title("🚌 Bizkaia Public Transport Accessibility Analysis")
st.markdown("""
**Policy-focused analysis of Bizkaibus service accessibility**  
*Identifying priority areas for service improvement based on population and travel time*
""")

# Methodology expander
with st.expander("📋 Methodology & Data Structure", expanded=False):
    st.markdown("""
    ### Data Columns Used in Analysis:

    **Origin Information:**
    - `Zona_Origen` → Origin zone ("núcleo") ID
    - `Zona_Origen_nombre` → Human-readable name of the origin zone
    - `Poblacion_Origen` → **Number of people living in the origin zone**

    **Destination Information:**
    - `Zona_Destino` → Destination zone ID (Point of Interest)
    - `Zona_Destino_nombre` → Human-readable name of the destination POI

    **Travel Time Metrics:**
    - `Tiempo_Viaje_Total_Minutos` → **Total door-to-door travel time**
    - `Tiempo_Trayecto_Minutos` → In-vehicle travel time only
    - `Numero_Transbordos` → Number of transfers required

    **Trip Necessity:**
    - `Necesita_viaje` → Whether a trip is actually needed (1 = needed, 0 = not needed)

    ### Accessibility Categories:
    - 🟢 **Excellent**: < 30 minutes
    - 🟡 **Good**: 30-45 minutes
    - 🟠 **Fair**: 45-60 minutes
    - 🔴 **Poor**: > 60 minutes

    ### Key Calculations:
    1. **Population-weighted accessibility**: All metrics weighted by population
    2. **Distance efficiency**: Travel time per kilometer
    3. **POI accessibility score**: Population-weighted average travel time
    """)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # Handle trip necessity
    if 'Necesita_viaje' in df.columns:
        df_all = df.copy()
        total_possible_trips = len(df)
        unnecessary_trips = (df['Necesita_viaje'] == 0).sum()
        necessary_trips = (df['Necesita_viaje'] == 1).sum()
        unnecessary_pct = (unnecessary_trips / total_possible_trips * 100) if total_possible_trips > 0 else 0
        df = df[df['Necesita_viaje'] == 1].copy()
    else:
        df_all = df.copy()
        unnecessary_trips = 0
        necessary_trips = len(df)
        unnecessary_pct = 0

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    transport_filter = st.sidebar.radio(
        "Transport Mode",
        ["Public transport only", "All trips"],
        index=0
    )

    if transport_filter == "Public transport only":
        df_filtered = df[df['Usa_Transporte_Publico'] == True].copy()
    else:
        df_filtered = df.copy()

    # Calculate population metrics
    zone_populations = df_filtered.groupby('Zona_Origen')['Poblacion_Origen'].first()
    total_population = zone_populations.sum()

    st.sidebar.markdown(f"**Analyzing {len(df_filtered):,} origin-destination pairs**")
    st.sidebar.markdown(f"**Total population: {total_population:,.0f} residents**")
    st.sidebar.markdown(f"**Origin zones: {df_filtered['Zona_Origen'].nunique()}**")

    if 'Necesita_viaje' in df_all.columns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Trip Necessity:**")
        st.sidebar.markdown(f"✅ Necessary trips: {necessary_trips:,} ({100 - unnecessary_pct:.1f}%)")
        st.sidebar.markdown(f"❌ Unnecessary trips: {unnecessary_trips:,} ({unnecessary_pct:.1f}%)")

    # Main KPIs
    st.header("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_travel_time = df_filtered['Tiempo_Viaje_Total_Minutos'].mean()
        st.metric("Avg Travel Time", f"{avg_travel_time:.1f} min")

    with col2:
        avg_transfers = df_filtered['Numero_Transbordos'].mean()
        st.metric("Avg Transfers", f"{avg_transfers:.2f}")

    with col3:
        zones_under_30 = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] <= 30]['Zona_Origen'].unique()
        pop_under_30 = zone_populations[zone_populations.index.isin(zones_under_30)].sum()
        under_30_pct = (pop_under_30 / total_population * 100) if total_population > 0 else 0
        st.metric("Population with <30min access", f"{under_30_pct:.1f}%")

    with col4:
        total_pois = df_filtered['Zona_Destino'].nunique()
        st.metric("Total POIs", f"{total_pois}")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Accessibility Overview",
        "📍 POI Analysis",
        "⚠️ Priority Areas",
        "📊 Distance Efficiency",
        "🔄 Trip need Patterns"
    ])

    # TAB 1: Accessibility Overview
    with tab1:
        st.subheader("Population Accessibility Distribution")
        st.markdown("*What percentage of the population has good vs poor connections to POIs?*")

        df_filtered['Accessibility_Category'] = pd.cut(
            df_filtered['Tiempo_Viaje_Total_Minutos'],
            bins=[0, 30, 45, 60, float('inf')],
            labels=['🟢 Excellent (<30min)', '🟡 Good (30-45min)',
                    '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
        )

        zone_best_access = df_filtered.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'min',
            'Poblacion_Origen': 'first'
        }).reset_index()

        zone_best_access['Best_Category'] = pd.cut(
            zone_best_access['Tiempo_Viaje_Total_Minutos'],
            bins=[0, 30, 45, 60, float('inf')],
            labels=['🟢 Excellent (<30min)', '🟡 Good (30-45min)',
                    '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
        )

        pop_by_category = zone_best_access.groupby('Best_Category')['Poblacion_Origen'].sum()
        category_pcts = (pop_by_category / total_population * 100) if total_population > 0 else pd.Series()

        col1, col2 = st.columns([2, 1])

        with col1:
            categories = ['🟢 Excellent (<30min)', '🟡 Good (30-45min)', '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
            populations = [pop_by_category.get(cat, 0) for cat in categories]
            percentages = [category_pcts.get(cat, 0) for cat in categories]

            fig1 = px.bar(
                x=categories,
                y=populations,
                color=categories,
                color_discrete_map={
                    '🟢 Excellent (<30min)': '#2ecc71',
                    '🟡 Good (30-45min)': '#f1c40f',
                    '🟠 Fair (45-60min)': '#e67e22',
                    '🔴 Poor (>60min)': '#e74c3c'
                },
                text=[f"{int(pop):,}<br><b>{pct:.1f}%</b>" for pop, pct in zip(populations, percentages)]
            )
            fig1.update_layout(
                title="Distribution of Population by Best Accessibility to Any POI",
                xaxis_title="Accessibility Category",
                yaxis_title="Population (residents)",
                showlegend=False,
                height=500
            )
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("### Summary")
            for cat, pop, pct in zip(categories, populations, percentages):
                if '🟢' in cat or '🟡' in cat:
                    st.success(f"**{cat}**\n\n{pop:,.0f} residents ({pct:.1f}%)")
                elif '🟠' in cat:
                    st.warning(f"**{cat}**\n\n{pop:,.0f} residents ({pct:.1f}%)")
                else:
                    st.error(f"**{cat}**\n\n{pop:,.0f} residents ({pct:.1f}%)")

        # Travel time distributions
        st.markdown("---")
        st.markdown("### Travel Time Distribution (Population-Weighted)")

        bins = np.linspace(df_filtered['Tiempo_Viaje_Total_Minutos'].min(),
                           df_filtered['Tiempo_Viaje_Total_Minutos'].max(), 41)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_populations = []
        for i in range(len(bins) - 1):
            mask = (df_filtered['Tiempo_Viaje_Total_Minutos'] >= bins[i]) & \
                   (df_filtered['Tiempo_Viaje_Total_Minutos'] < bins[i + 1])
            zones_in_bin = df_filtered[mask]['Zona_Origen'].unique()
            pop_in_bin = zone_populations[zone_populations.index.isin(zones_in_bin)].sum()
            bin_populations.append(pop_in_bin / total_population * 100)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=bin_centers,
            y=bin_populations,
            marker_color='#EBEBEB',
            width=(bins[1] - bins[0]) * 0.9,
            showlegend=False
        ))

        travel_times_list = []
        for _, row in df_filtered.iterrows():
            travel_times_list.extend([row['Tiempo_Viaje_Total_Minutos']] * int(row['Poblacion_Origen'] / 100))

        if len(travel_times_list) > 0:
            travel_times_weighted = np.array(travel_times_list)
            kde = stats.gaussian_kde(travel_times_weighted)
            x_range = np.linspace(df_filtered['Tiempo_Viaje_Total_Minutos'].min(),
                                  df_filtered['Tiempo_Viaje_Total_Minutos'].max(), 200)
            y_kde = kde(x_range)
            y_kde_scaled = y_kde * max(bin_populations) / y_kde.max() * 0.8

            fig2.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

            pop_under_30_pct = category_pcts.get('🟢 Excellent (<30min)', 0)
            pop_under_45_pct = pop_under_30_pct + category_pcts.get('🟡 Good (30-45min)', 0)
            pop_under_60_pct = pop_under_45_pct + category_pcts.get('🟠 Fair (45-60min)', 0)

            fig2.add_vline(x=30, line_dash="dash", line_color="green",
                           annotation_text=f"30 min ({pop_under_30_pct:.1f}%)")
            fig2.add_vline(x=45, line_dash="dash", line_color="orange",
                           annotation_text=f"45 min ({pop_under_45_pct:.1f}%)")
            fig2.add_vline(x=60, line_dash="dash", line_color="red",
                           annotation_text=f"60 min ({pop_under_60_pct:.1f}%)")

        fig2.update_layout(
            xaxis_title="Total Travel Time (minutes)",
            yaxis_title="Percentage of Population (%)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### Travel Time Distribution by Zones")

        fig3 = px.histogram(
            zone_best_access,
            x='Tiempo_Viaje_Total_Minutos',
            nbins=40,
            histnorm='percent',
            color_discrete_sequence=['#EBEBEB']
        )

        travel_times_zones = zone_best_access['Tiempo_Viaje_Total_Minutos'].dropna()
        kde = stats.gaussian_kde(travel_times_zones)
        x_range = np.linspace(travel_times_zones.min(), travel_times_zones.max(), 200)
        y_kde = kde(x_range)
        y_kde_normalized = y_kde * 100 * (travel_times_zones.max() - travel_times_zones.min()) / 40

        fig3.add_trace(go.Scatter(
            x=x_range,
            y=y_kde_normalized,
            mode='lines',
            line=dict(color='#C00000', width=3),
            showlegend=False
        ))

        total_zones = len(zone_best_access)
        zones_under_30_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 30).sum() / total_zones * 100
        zones_under_45_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 45).sum() / total_zones * 100
        zones_under_60_pct = (zone_best_access['Tiempo_Viaje_Total_Minutos'] <= 60).sum() / total_zones * 100

        fig3.add_vline(x=30, line_dash="dash", line_color="green",
                       annotation_text=f"30 min<br>({zones_under_30_pct:.1f}% zones)")
        fig3.add_vline(x=45, line_dash="dash", line_color="orange",
                       annotation_text=f"45 min<br>({zones_under_45_pct:.1f}% zones)")
        fig3.add_vline(x=60, line_dash="dash", line_color="red",
                       annotation_text=f"60 min<br>({zones_under_60_pct:.1f}% zones)")

        fig3.update_layout(
            xaxis_title="Total Travel Time (minutes)",
            yaxis_title="Percentage of Zones (%)",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

    # TAB 2: POI Analysis
    with tab2:
        st.subheader("Point of Interest (POI) Accessibility")
        st.markdown("*Which destinations are well-connected and which need improvement?*")

        poi_accessibility = []
        for poi in df_filtered['Zona_Destino_nombre'].unique():
            poi_data = df_filtered[df_filtered['Zona_Destino_nombre'] == poi]

            zone_data = poi_data.groupby('Zona_Origen').agg({
                'Poblacion_Origen': 'first',
                'Tiempo_Viaje_Total_Minutos': 'mean'
            })

            total_pop_poi = zone_data['Poblacion_Origen'].sum()

            pop_excellent = zone_data[zone_data['Tiempo_Viaje_Total_Minutos'] <= 30]['Poblacion_Origen'].sum()
            pop_good = zone_data[(zone_data['Tiempo_Viaje_Total_Minutos'] > 30) &
                                 (zone_data['Tiempo_Viaje_Total_Minutos'] <= 45)]['Poblacion_Origen'].sum()
            pop_fair = zone_data[(zone_data['Tiempo_Viaje_Total_Minutos'] > 45) &
                                 (zone_data['Tiempo_Viaje_Total_Minutos'] <= 60)]['Poblacion_Origen'].sum()
            pop_poor = zone_data[zone_data['Tiempo_Viaje_Total_Minutos'] > 60]['Poblacion_Origen'].sum()

            weighted_avg_time = (zone_data['Tiempo_Viaje_Total_Minutos'] * zone_data[
                'Poblacion_Origen']).sum() / total_pop_poi if total_pop_poi > 0 else 0

            poi_accessibility.append({
                'POI': poi,
                'Avg Time (weighted)': weighted_avg_time,
                'Total Population': total_pop_poi,
                'Excellent (%)': (pop_excellent / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Good (%)': (pop_good / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Fair (%)': (pop_fair / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Poor (%)': (pop_poor / total_pop_poi * 100) if total_pop_poi > 0 else 0
            })

        poi_df = pd.DataFrame(poi_accessibility).sort_values('Avg Time (weighted)')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Best Connected POIs")
            best_pois = poi_df.head(5)
            fig4 = px.bar(
                best_pois,
                x='Avg Time (weighted)',
                y='POI',
                orientation='h',
                color='Excellent (%)',
                color_continuous_scale='Greens',
                text='Avg Time (weighted)',
                hover_data={'Total Population': ':,.0f'}
            )
            fig4.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
            fig4.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.markdown("#### 🔴 Worst Connected POIs")
            worst_pois = poi_df.tail(5)
            fig5 = px.bar(
                worst_pois,
                x='Avg Time (weighted)',
                y='POI',
                orientation='h',
                color='Poor (%)',
                color_continuous_scale='Reds',
                text='Avg Time (weighted)',
                hover_data={'Total Population': ':,.0f'}
            )
            fig5.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
            fig5.update_layout(height=300, yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Accessibility Breakdown by POI")

        poi_df_sorted = poi_df.sort_values('Avg Time (weighted)')

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            name='🟢 Excellent (<30min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Excellent (%)'],
            orientation='h',
            marker_color='#2ecc71',
            text=poi_df_sorted['Excellent (%)'].round(1),
            textposition='inside',
            texttemplate='%{text:.0f}%'
        ))
        fig6.add_trace(go.Bar(
            name='🟡 Good (30-45min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Good (%)'],
            orientation='h',
            marker_color='#f1c40f',
            text=poi_df_sorted['Good (%)'].round(1),
            textposition='inside',
            texttemplate='%{text:.0f}%'
        ))
        fig6.add_trace(go.Bar(
            name='🟠 Fair (45-60min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Fair (%)'],
            orientation='h',
            marker_color='#e67e22',
            text=poi_df_sorted['Fair (%)'].round(1),
            textposition='inside',
            texttemplate='%{text:.0f}%'
        ))
        fig6.add_trace(go.Bar(
            name='🔴 Poor (>60min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Poor (%)'],
            orientation='h',
            marker_color='#e74c3c',
            text=poi_df_sorted['Poor (%)'].round(1),
            textposition='inside',
            texttemplate='%{text:.0f}%'
        ))
        fig6.update_layout(
            barmode='stack',
            title='POI Accessibility Quality Distribution',
            xaxis_title='Percentage of Population (%)',
            yaxis_title='Destination POI',
            height=max(400, len(poi_df) * 25),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("#### Detailed POI Statistics")
        display_df = poi_df[
            ['POI', 'Avg Time (weighted)', 'Total Population', 'Excellent (%)', 'Good (%)', 'Fair (%)', 'Poor (%)']
        ].copy()
        st.dataframe(
            display_df.style.format({
                'Avg Time (weighted)': '{:.1f} min',
                'Total Population': '{:,.0f}',
                'Excellent (%)': '{:.1f}%',
                'Good (%)': '{:.1f}%',
                'Fair (%)': '{:.1f}%',
                'Poor (%)': '{:.1f}%'
            }).background_gradient(subset=['Avg Time (weighted)'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    # TAB 3: Priority Areas
    with tab3:
        st.subheader("Priority Areas for Service Improvement")
        st.markdown("*Origin zones with poor accessibility affecting the most residents*")

        zone_priority = df_filtered.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Zona_Origen_nombre': 'first',
            'Zona_Destino': 'count',
            'Numero_Transbordos': 'mean',
            'Poblacion_Origen': 'first'
        }).reset_index()
        zone_priority.columns = ['Zone_ID', 'Avg Travel Time', 'Zone Name', 'POIs Served', 'Avg Transfers',
                                 'Population']

        poor_zones = zone_priority[zone_priority['Avg Travel Time'] > 45].sort_values('Population', ascending=False)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### 🔴 High-Population Zones with Worst Accessibility")
            if len(poor_zones) > 0:
                st.markdown(
                    f"**{len(poor_zones)} zones** with **{poor_zones['Population'].sum():,.0f} residents** have average travel times exceeding 45 minutes")

                top_poor = poor_zones.head(15)
                fig7 = px.bar(
                    top_poor,
                    x='Population',
                    y='Zone Name',
                    orientation='h',
                    color='Avg Travel Time',
                    color_continuous_scale='Reds',
                    hover_data=['Avg Transfers', 'POIs Served'],
                    text='Population'
                )
                fig7.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig7.update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title='Population (residents)',
                    yaxis_title='Origin Zone'
                )
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.success("✅ No zones with poor average accessibility found!")

        with col2:
            st.markdown("#### Key Impact Metrics")

            if len(poor_zones) > 0:
                avg_poor = poor_zones['Avg Travel Time'].mean()
                avg_transfers_poor = poor_zones['Avg Transfers'].mean()
                total_affected_pop = poor_zones['Population'].sum()
                pct_affected = (total_affected_pop / total_population * 100) if total_population > 0 else 0

                st.error(f"**Avg travel time in poor zones:** {avg_poor:.1f} minutes")
                st.warning(f"**Avg transfers needed:** {avg_transfers_poor:.2f}")
                st.info(f"**Population affected:** {total_affected_pop:,.0f} residents ({pct_affected:.1f}% of total)")

                st.markdown("#### Most Critical Zones")
                st.markdown("*(by population affected)*")
                for idx, row in poor_zones.head(5).iterrows():
                    st.write(f"**{row['Zone Name']}**")
                    st.write(f"→ {row['Population']:,.0f} residents")
                    st.write(f"→ {row['Avg Travel Time']:.1f} min avg")
                    st.markdown("---")
            else:
                st.success("✅ All zones have good accessibility!")

        # Problem matrix: POI vs Origin zones with poor access
        st.markdown("---")
        st.markdown("#### Problem Matrix: Which high-population zones struggle to reach their POI?")
        st.markdown("*Showing zones with highest population impact*")

        # Filter for problematic connections (>45 min)
        problem_connections = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] > 45].copy()

        if len(problem_connections) > 0:
            # Get zones with highest population
            zone_pops = problem_connections.groupby(['Zona_Origen', 'Zona_Origen_nombre']).agg({
                'Poblacion_Origen': 'first'
            }).reset_index().sort_values('Poblacion_Origen', ascending=False)

            top_problem_zones = zone_pops.head(10)['Zona_Origen_nombre'].values

            # Get POIs with most problematic connections
            top_problem_pois = problem_connections['Zona_Destino_nombre'].value_counts().head(8).index

            # Create pivot table
            problem_matrix = problem_connections[
                (problem_connections['Zona_Origen_nombre'].isin(top_problem_zones)) &
                (problem_connections['Zona_Destino_nombre'].isin(top_problem_pois))
                ].pivot_table(
                values='Tiempo_Viaje_Total_Minutos',
                index='Zona_Origen_nombre',
                columns='Zona_Destino_nombre',
                aggfunc='mean'
            )

            if not problem_matrix.empty:
                fig8 = px.imshow(
                    problem_matrix,
                    color_continuous_scale='RdYlGn_r',
                    aspect='auto',
                    labels=dict(x="Destination POI", y="Origin Zone (high population)", color="Travel Time (min)")
                )
                fig8.update_layout(height=500)
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("No significant problem patterns found.")
        else:
            st.success("✅ No problematic connections found!")

    # TAB 4: Distance Efficiency
    with tab4:
        st.subheader("Distance vs Travel Time Efficiency")
        st.markdown("*Identifying where service is slow despite short distances*")

        df_filtered['Time_per_Km'] = df_filtered['Tiempo_Viaje_Total_Minutos'] / df_filtered['Distancia_Viaje_Total_Km']
        df_filtered['Time_per_Km'] = df_filtered['Time_per_Km'].replace([np.inf, -np.inf], np.nan)

        fig9 = px.scatter(
            df_filtered.sample(min(1000, len(df_filtered))),
            x='Distancia_Viaje_Total_Km',
            y='Tiempo_Viaje_Total_Minutos',
            color='Numero_Transbordos',
            hover_data=['Zona_Origen_nombre', 'Zona_Destino_nombre', 'Poblacion_Origen'],
            opacity=0.6,
            color_continuous_scale='Reds',
            labels={
                'Distancia_Viaje_Total_Km': 'Distance (km)',
                'Tiempo_Viaje_Total_Minutos': 'Travel Time (min)',
                'Numero_Transbordos': 'Transfers'
            }
        )

        max_dist = df_filtered['Distancia_Viaje_Total_Km'].max()
        fig9.add_trace(go.Scatter(
            x=[0, max_dist],
            y=[0, max_dist * 3],
            mode='lines',
            name='Ideal (20 km/h avg)',
            line=dict(color='green', dash='dash', width=2)
        ))

        fig9.update_layout(
            title='Travel Time vs Distance (points above green line are slower than 20 km/h average)',
            height=500
        )
        st.plotly_chart(fig9, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_time_per_km = df_filtered['Time_per_Km'].mean()
            st.metric("Avg Time per Km", f"{avg_time_per_km:.1f} min/km")

        with col2:
            avg_speed = 60 / avg_time_per_km if avg_time_per_km > 0 else 0
            st.metric("Avg Effective Speed", f"{avg_speed:.1f} km/h")

        with col3:
            inefficient_pct = (df_filtered['Time_per_Km'] > 3).sum() / len(df_filtered) * 100
            st.metric("Slow Connections", f"{inefficient_pct:.1f}%",
                      help="Connections slower than 20 km/h (>3 min/km)")

        # Zone efficiency comparison
        st.markdown("---")
        st.markdown("#### Travel Efficiency by Zone")
        st.markdown("*Top 20 zones by population, sorted by efficiency*")

        zone_efficiency = df_filtered.groupby('Zona_Origen').agg({
            'Time_per_Km': 'mean',
            'Distancia_Viaje_Total_Km': 'mean',
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Poblacion_Origen': 'first',
            'Zona_Origen_nombre': 'first'
        }).reset_index()

        zone_efficiency.columns = ['Zone_ID', 'Min per Km', 'Avg Distance (km)', 'Avg Time (min)', 'Population',
                                   'Zone Name']
        zone_efficiency['Effective Speed (km/h)'] = 60 / zone_efficiency['Min per Km']

        # Get top 20 by population
        top_zones_by_pop = zone_efficiency.nlargest(20, 'Population')

        fig10 = px.bar(
            top_zones_by_pop.sort_values('Min per Km', ascending=False),
            x='Zone Name',
            y='Min per Km',
            color='Min per Km',
            color_continuous_scale='RdYlGn_r',
            hover_data={
                'Effective Speed (km/h)': ':.1f',
                'Avg Distance (km)': ':.1f',
                'Avg Time (min)': ':.1f',
                'Population': ':,.0f'
            }
        )
        fig10.update_layout(
            title='Travel Time Efficiency by Zone (Top 20 by population, lower = better)',
            xaxis_title='Origin Zone',
            yaxis_title='Average Minutes per Kilometer',
            height=400
        )
        fig10.update_xaxes(tickangle=45)
        st.plotly_chart(fig10, use_container_width=True)

        # Data table
        st.markdown("#### Zone Efficiency Rankings (Top 20 by Population)")
        display_efficiency = top_zones_by_pop.sort_values('Min per Km')
        st.dataframe(
            display_efficiency[['Zone Name', 'Population', 'Min per Km', 'Avg Distance (km)',
                                'Avg Time (min)', 'Effective Speed (km/h)']].style.format({
                'Population': '{:,.0f}',
                'Min per Km': '{:.2f}',
                'Avg Distance (km)': '{:.1f}',
                'Avg Time (min)': '{:.1f}',
                'Effective Speed (km/h)': '{:.1f}'
            }).background_gradient(subset=['Min per Km'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True
        )

    # TAB 5: Trip Need Patterns
    with tab5:
        st.subheader("Trip Need Analysis")
        st.markdown("*Understanding which origin-destination pairs have actual travel need*")

        if 'Necesita_viaje' in df_all.columns:
            zone_need = df_all.groupby('Zona_Origen').agg({
                'Necesita_viaje': 'sum',
                'Poblacion_Origen': 'first',
                'Zona_Origen_nombre': 'first'
            }).reset_index()
            zone_need['Necesita_viaje'] = zone_need['Necesita_viaje'] > 0

            pop_with_need = zone_need[zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
            pop_without_need = zone_need[~zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
            total_pop = zone_need['Poblacion_Origen'].sum()
            pop_with_need_pct = (pop_with_need / total_pop * 100) if total_pop > 0 else 0

            st.markdown("### Overall Need Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                zones_with_need = zone_need['Necesita_viaje'].sum()
                total_zones = len(zone_need)
                zones_need_pct = (zones_with_need / total_zones * 100) if total_zones > 0 else 0
                st.metric("Zones with Need of Travel with BizkaiBus", f"{zones_need_pct:.1f}%",
                          help=f"{zones_with_need} out of {total_zones} zones need to reach at least one POI")

            with col2:
                st.metric("Population with Need to travel with Bizkaibus", f"{pop_with_need_pct:.1f}%",
                          help=f"{pop_with_need:,.0f} out of {total_pop:,.0f} residents live in zones that need trips")

            with col3:
                pois_with_need = df_all[df_all['Necesita_viaje'] == 1]['Zona_Destino'].nunique()
                total_pois_all = df_all['Zona_Destino'].nunique()
                pois_need_pct = (pois_with_need / total_pois_all * 100) if total_pois_all > 0 else 0
                st.metric("POIs with need", f"{pois_need_pct:.1f}%",
                          help=f"{pois_with_need} out of {total_pois_all} POIs have at least one origin zone that needs them")

            # Population breakdown by need
            st.markdown("---")
            st.markdown("### Population Distribution by Trip Necessity")

            col1, col2 = st.columns(2)

            with col1:
                # Pie chart for population
                pop_data = pd.DataFrame({
                    'Category': ['Residents needing trips', 'Residents not needing trips'],
                    'Population': [pop_with_need, pop_without_need],
                    'Percentage': [pop_with_need_pct, 100 - pop_with_need_pct]
                })

                fig_pop = px.pie(
                    pop_data,
                    values='Population',
                    names='Category',
                    color='Category',
                    color_discrete_map={
                        'Residents needing trips': '#3498db',
                        'Residents not needing trips': '#95a5a6'
                    },
                    hole=0.4
                )
                fig_pop.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    texttemplate='<b>%{label}</b><br>%{value:,.0f} residents<br>(%{percent})'
                )
                fig_pop.update_layout(
                    title='Population by Trip Necessity',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_pop, use_container_width=True)

            with col2:
                # Show zones with no need
                zones_no_need = zone_need[~zone_need['Necesita_viaje']].sort_values('Poblacion_Origen',
                                                                                    ascending=False)

                if len(zones_no_need) > 0:
                    st.markdown("#### Zones with No Trip Need")
                    st.markdown(
                        f"*{len(zones_no_need)} zones with {pop_without_need:,.0f} residents don't need to reach any POI*")

                    if len(zones_no_need) > 10:
                        st.markdown("**Top 10 by population:**")
                        display_zones = zones_no_need.head(10)
                    else:
                        display_zones = zones_no_need

                    for idx, row in display_zones.iterrows():
                        st.write(f"**{row['Zona_Origen_nombre']}**: {row['Poblacion_Origen']:,.0f} residents")
                else:
                    st.success("All zones have trip need to at least one POI!")

            # Need by POI - with population weighting
            st.markdown("---")
            st.markdown("### Trip Need by Destination POI")
            st.markdown("*Showing both trip-based and population-based need*")

            # Calculate both trip need and population need for each POI
            poi_need_analysis = []

            for poi in df_all['Zona_Destino_nombre'].unique():
                poi_data = df_all[df_all['Zona_Destino_nombre'] == poi]

                # Trip-based metrics
                trips_needed = poi_data['Necesita_viaje'].sum()
                total_possible = len(poi_data)
                trip_need_rate = (trips_needed / total_possible * 100) if total_possible > 0 else 0

                # Population-based metrics
                zones_needing = poi_data[poi_data['Necesita_viaje'] == 1]['Zona_Origen'].unique()
                pop_needing = zone_need[zone_need['Zona_Origen'].isin(zones_needing)]['Poblacion_Origen'].sum()
                pop_need_rate = (pop_needing / total_pop * 100) if total_pop > 0 else 0

                poi_need_analysis.append({
                    'POI': poi,
                    'Trips_Needed': trips_needed,
                    'Total_Possible_Trips': total_possible,
                    'Trip_Need_Rate_%': trip_need_rate,
                    'Population_Needing': pop_needing,
                    'Population_Need_Rate_%': pop_need_rate
                })

            need_by_poi = pd.DataFrame(poi_need_analysis).sort_values('Population_Need_Rate_%',
                                                                      ascending=False)

            # Dual bar chart
            fig_need = go.Figure()

            top_15 = need_by_poi.head(15)

            fig_need.add_trace(go.Bar(
                name='% of Trips',
                y=top_15['POI'],
                x=top_15['Trip_Need_Rate_%'],
                orientation='h',
                marker_color='#3498db',
                text=top_15['Trip_Need_Rate_%'].round(1),
                texttemplate='%{text:.1f}%',
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Trip need: %{x:.1f}%<br>Trips Needed: %{customdata[0]:,.0f}<extra></extra>',
                customdata=top_15[['Trips_Needed']].values
            ))

            fig_need.add_trace(go.Bar(
                name='% of Population',
                y=top_15['POI'],
                x=top_15['Population_Need_Rate_%'],
                orientation='h',
                marker_color='#e74c3c',
                text=top_15['Population_Need_Rate_%'].round(1),
                texttemplate='%{text:.1f}%',
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Population need: %{x:.1f}%<br>Residents: %{customdata[0]:,.0f}<extra></extra>',
                customdata=top_15[['Population_Needing']].values
            ))

            fig_need.update_layout(
                title='Top 15 POIs by Need Rate (Trip-based vs Population-based)',
                xaxis_title='Need Rate (%)',
                yaxis_title='Destination POI',
                height=600,
                yaxis={'categoryorder': 'total ascending'},
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_need, use_container_width=True)

            # Show detailed table
            st.markdown("#### Detailed Need Statistics")
            display_need = need_by_poi[['POI', 'Trips_Needed', 'Total_Possible_Trips', 'Trip_Need_Rate_%',
                                        'Population_Needing', 'Population_Need_Rate_%']].copy()
            st.dataframe(
                display_need.style.format({
                    'Trips_Needed': '{:,.0f}',
                    'Total_Possible_Trips': '{:,.0f}',
                    'Trip_Need_Rate_%': '{:.1f}%',
                    'Population_Needing': '{:,.0f}',
                    'Population_Need_Rate_%': '{:.1f}%'
                }).background_gradient(subset=['Trip_Need_Rate_%'], cmap='Blues')
                .background_gradient(subset=['Population_Need_Rate_%'], cmap='Reds'),
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.warning("⚠️ 'Necesita_viaje' column not found in the dataset")

    # ========================================================================
    # DOWNLOAD SECTION - AFTER ALL DATA IS COMPUTED
    # ========================================================================

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Report")


    def create_complete_report_package():
        """Create ZIP with Excel + all chart images"""
        zip_buffer = BytesIO()

        # Prepare zone_need data if needed
        if 'Necesita_viaje' in df_all.columns:
            zone_need_for_report = df_all.groupby('Zona_Origen').agg({
                'Necesita_viaje': 'sum',
                'Poblacion_Origen': 'first',
                'Zona_Origen_nombre': 'first'
            }).reset_index()
            zone_need_for_report['Necesita_viaje'] = zone_need_for_report['Necesita_viaje'] > 0
        else:
            zone_need_for_report = None

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add Excel report
            excel_report = create_excel_report(df_filtered, zone_populations, total_population, poi_df, poor_zones,
                                               df_all, zone_need_for_report)
            zip_file.writestr(f"bizkaia_report_{datetime.now().strftime('%Y%m%d')}.xlsx", excel_report.getvalue())

            # Add all chart images
            charts = create_all_charts(df_filtered, zone_populations, total_population, poi_df, poor_zones)
            for filename, image_data in charts.items():
                zip_file.writestr(filename, image_data)

        zip_buffer.seek(0)
        return zip_buffer


    # Generate and offer download
    report_package = create_complete_report_package()

    st.sidebar.download_button(
        label="📦 Download Complete Report",
        data=report_package,
        file_name=f"bizkaia_accessibility_report_{datetime.now().strftime('%Y%m%d')}.zip",
        mime="application/zip",
        help="Includes Excel report + all chart images (PNG)"
    )

    st.sidebar.markdown("*ZIP includes:*")
    st.sidebar.markdown("- 📊 Excel report (3 sheets)")
    st.sidebar.markdown("- 📈 7 high-resolution charts (PNG)")