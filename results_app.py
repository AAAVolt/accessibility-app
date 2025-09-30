import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

st.set_page_config(page_title="Bizkaia Accessibility Analysis", layout="wide", page_icon="🚌")

# Title and introduction
st.title("🚌 Bizkaia Public Transport Accessibility Analysis")
st.markdown("""
**Policy-focused analysis of Bizkaibus service accessibility**  
*Identifying priority areas for service improvement based on population and travel time*
""")
st.set_page_config(page_title="Bizkaia Accessibility Analysis", layout="wide", page_icon="🚌")


# Methodology expander
with st.expander("📋 Methodology & Data Structure", expanded=False):
    st.markdown("""
    ### Data Columns Used in Analysis:

    **Origin Information:**
    - `Zona_Origen` → Origin zone ("núcleo") ID - unique identifier for each neighborhood/area
    - `Zona_Origen_nombre` → Human-readable name of the origin zone/núcleo
    - `Poblacion_Origen` → **Number of people living in the origin zone** (used for population-weighted metrics)

    **Destination Information:**
    - `Zona_Destino` → Destination zone ID (Point of Interest)
    - `Zona_Destino_nombre` → Human-readable name of the destination POI

    **Travel Time Metrics:**
    - `Tiempo_Viaje_Total_Minutos` → **Total door-to-door travel time** (main accessibility metric)
    - `Tiempo_Trayecto_Minutos` → In-vehicle travel time only
    - `Tiempo_Acceso_Minutos` → Walking time from origin to boarding stop
    - `Tiempo_Salida_Minutos` → Walking time from alighting stop to final destination
    - `Tiempo_Espera_Transbordo_Minutos` → Total waiting time at transfers

    **Distance Metrics:**
    - `Distancia_Viaje_Total_Km` → Total door-to-door distance (walking + in-vehicle)
    - `Distancia_Trayecto_Km` → In-vehicle distance only
    - `Distancia_Acceso_Metros` → Walking distance to boarding stop
    - `Distancia_Salida_Metros` → Walking distance from alighting stop

    **Service Quality:**
    - `Numero_Transbordos` → Number of transfers required
    - `Usa_Transporte_Publico` → Whether public transport is used (True/False)
    - `Frecuencia_Servicio` → Service headway (minutes between vehicles)
    - `Parada_Origen` → Boarding stop/station
    - `Parada_Destino` → Alighting stop/station

    ### Accessibility Categories:
    We classify connections into 4 categories based on **total travel time**:
    - 🟢 **Excellent**: < 30 minutes
    - 🟡 **Good**: 30-45 minutes
    - 🟠 **Fair**: 45-60 minutes
    - 🔴 **Poor**: > 60 minutes

    ### Key Calculations:

    1. **Population-weighted accessibility**: Each origin zone has a population (`Poblacion_Origen`). We weight all metrics by this population to understand how many actual residents experience good vs poor connections. For example, if 1,000 residents live in a zone with excellent access and 500 residents live in a zone with poor access, then 66.7% of the population has excellent access.

    2. **Distance efficiency**: We calculate travel time per kilometer (Tiempo_Viaje_Total_Minutos / Distancia_Viaje_Total_Km) to identify where service is inefficient despite short distances.

    3. **POI accessibility score**: For each POI, we calculate population-weighted average travel time from all origin zones. This shows which destinations are accessible to the most people.

    ### Important Notes:
    - Each row represents one origin-destination pair (one possible trip)
    - **All percentages reflect actual population impact**, not just geographic coverage
    - When we say "X% of population has good access", we mean X% of total residents across all zones
    - Priority areas are identified by combining poor accessibility with high population counts
    """)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)

    # Data cleaning
    df.columns = df.columns.str.strip()

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Filter by public transport usage
    transport_filter = st.sidebar.radio(
        "Transport Mode",
        ["Public transport only", "All trips"],
        index=0
    )

    if transport_filter == "Public transport only":
        df_filtered = df[df['Usa_Transporte_Publico'] == True].copy()
    else:
        df_filtered = df.copy()

    # Calculate total population (sum of unique origin zones)
    zone_populations = df_filtered.groupby('Zona_Origen')['Poblacion_Origen'].first()
    total_population = zone_populations.sum()

    st.sidebar.markdown(f"**Analyzing {len(df_filtered):,} origin-destination pairs**")
    st.sidebar.markdown(f"**Total population: {total_population:,.0f} residents**")
    st.sidebar.markdown(f"**Origin zones: {df_filtered['Zona_Origen'].nunique()}**")

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
        # Population with connections under 30 min
        zones_under_30 = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] <= 30]['Zona_Origen'].unique()
        pop_under_30 = zone_populations[zone_populations.index.isin(zones_under_30)].sum()
        under_30_pct = (pop_under_30 / total_population * 100) if total_population > 0 else 0
        st.metric("Population with <30min access", f"{under_30_pct:.1f}%")

    with col4:
        total_pois = df_filtered['Zona_Destino'].nunique()
        st.metric("Total POIs", f"{total_pois}")

    # Create tabs for policy-focused analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Accessibility Overview",
        "📍 POI Analysis",
        "⚠️ Priority Areas",
        "📊 Distance Efficiency"
    ])

    # TAB 1: Accessibility Overview
    with tab1:
        st.subheader("Population Accessibility Distribution")
        st.markdown("*What percentage of the population has good vs poor connections to POIs?*")

        # Create accessibility categories
        df_filtered['Accessibility_Category'] = pd.cut(
            df_filtered['Tiempo_Viaje_Total_Minutos'],
            bins=[0, 30, 45, 60, float('inf')],
            labels=['🟢 Excellent (<30min)', '🟡 Good (30-45min)',
                    '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
        )

        # Calculate POPULATION-WEIGHTED percentages
        # For each zone, determine its best accessibility category (minimum travel time)
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
            # Bar chart with population-weighted percentages
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
                height=400
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

        # Travel time distribution histogram
        st.markdown("---")
        st.markdown("### Travel Time Distribution")
        fig2 = px.histogram(
            df_filtered,
            x='Tiempo_Viaje_Total_Minutos',
            nbins=40,
            histnorm='percent',
            color_discrete_sequence=['#C00000']
        )
        travel_times = df_filtered['Tiempo_Viaje_Total_Minutos'].dropna()
        kde = stats.gaussian_kde(travel_times)
        x_range = np.linspace(travel_times.min(), travel_times.max(), 200)
        y_kde = kde(x_range)
        # Normalize to match histogram scale (percent)
        y_kde_normalized = y_kde * 100 * (travel_times.max() - travel_times.min()) / 40  # 40 = nbins

        fig2.add_trace(go.Scatter(
            x=x_range,
            y=y_kde_normalized,
            mode='lines',
            name='Density Curve',
            line=dict(color='blue', width=3),
            showlegend=False
        ))
        fig2.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="30 min")
        fig2.add_vline(x=45, line_dash="dash", line_color="orange", annotation_text="45 min")
        fig2.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="60 min")
        fig2.update_layout(
            xaxis_title="Total Travel Time (minutes)",
            yaxis_title="Percentage of Connections (%)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)



    # TAB 2: POI Analysis
    with tab2:
        st.subheader("Point of Interest (POI) Accessibility")
        st.markdown("*Which destinations are well-connected and which need improvement?*")

        # POI statistics with POPULATION weighting
        poi_accessibility = []
        for poi in df_filtered['Zona_Destino_nombre'].unique():
            poi_data = df_filtered[df_filtered['Zona_Destino_nombre'] == poi]

            # Get unique zones and their populations
            zone_data = poi_data.groupby('Zona_Origen').agg({
                'Poblacion_Origen': 'first',
                'Tiempo_Viaje_Total_Minutos': 'mean'
            })

            total_pop_poi = zone_data['Poblacion_Origen'].sum()

            # Calculate population in each time category
            pop_excellent = zone_data[zone_data['Tiempo_Viaje_Total_Minutos'] <= 30]['Poblacion_Origen'].sum()
            pop_good = zone_data[(zone_data['Tiempo_Viaje_Total_Minutos'] > 30) &
                                 (zone_data['Tiempo_Viaje_Total_Minutos'] <= 45)]['Poblacion_Origen'].sum()
            pop_poor = zone_data[zone_data['Tiempo_Viaje_Total_Minutos'] > 60]['Poblacion_Origen'].sum()

            # Calculate population-weighted average travel time
            weighted_avg_time = (zone_data['Tiempo_Viaje_Total_Minutos'] * zone_data[
                'Poblacion_Origen']).sum() / total_pop_poi if total_pop_poi > 0 else 0

            poi_accessibility.append({
                'POI': poi,
                'Avg Time (weighted)': weighted_avg_time,
                'Total Population': total_pop_poi,
                'Excellent (%)': (pop_excellent / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Good (%)': (pop_good / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Poor (%)': (pop_poor / total_pop_poi * 100) if total_pop_poi > 0 else 0
            })

        poi_df = pd.DataFrame(poi_accessibility).sort_values('Avg Time (weighted)')

        # Top and bottom POIs
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Best Connected POIs")
            st.markdown("*(lowest population-weighted travel time)*")
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
            st.markdown("*(highest population-weighted travel time)*")
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

        # Detailed POI breakdown
        st.markdown("---")
        st.markdown("#### Accessibility Breakdown by POI")
        st.markdown("*Percentage of **population** with excellent, good, or poor access to each POI*")

        # Stacked bar chart
        fig6 = go.Figure()

        poi_df_sorted = poi_df.sort_values('Avg Time (weighted)')

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
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig6, use_container_width=True)

        # Data table
        st.markdown("#### Detailed POI Statistics")
        display_df = poi_df[
            ['POI', 'Avg Time (weighted)', 'Total Population', 'Excellent (%)', 'Good (%)', 'Poor (%)']].copy()
        st.dataframe(
            display_df.style.format({
                'Avg Time (weighted)': '{:.1f} min',
                'Total Population': '{:,.0f}',
                'Excellent (%)': '{:.1f}%',
                'Good (%)': '{:.1f}%',
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

        # Calculate zone-level statistics
        zone_priority = df_filtered.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Zona_Origen_nombre': 'first',
            'Zona_Destino': 'count',
            'Numero_Transbordos': 'mean',
            'Poblacion_Origen': 'first'
        }).reset_index()
        zone_priority.columns = ['Zone_ID', 'Avg Travel Time', 'Zone Name', 'POIs Served', 'Avg Transfers',
                                 'Population']

        # Identify poor connections (>45min average) sorted by population
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

                # Top affected zones list
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
        st.markdown("#### Problem Matrix: Which high-population zones struggle to reach which POIs?")
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

        # Calculate time per km
        df_filtered['Time_per_Km'] = df_filtered['Tiempo_Viaje_Total_Minutos'] / df_filtered['Distancia_Viaje_Total_Km']
        df_filtered['Time_per_Km'] = df_filtered['Time_per_Km'].replace([np.inf, -np.inf], np.nan)

        # Scatter plot: Distance vs Time
        fig9 = px.scatter(
            df_filtered.sample(min(1000, len(df_filtered))),  # Sample for performance
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

        # Add reference line for ideal speed (20 km/h average = 3 min/km)
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

        # Efficiency metrics
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

else:
    st.info("👆 Please upload an Excel file to begin the analysis")

    st.markdown("""
    ### This tool analyzes:

    1. **📊 Accessibility Overview**
       - What % of **population** has good/poor connections to any POI
       - Travel time distribution across the network
       - Zone-level performance comparison (weighted by population)

    2. **📍 POI Analysis**
       - Which destinations are well/poorly connected (population-weighted metrics)
       - % of **population** that can reach each POI within different time thresholds
       - Accessibility quality breakdown showing which POIs serve the most people

    3. **⚠️ Priority Areas**
       - Zones with worst accessibility and **highest population impact**
       - Total residents affected by poor connections
       - Problem matrix showing which high-population zones struggle with which POIs

    4. **📊 Distance Efficiency**
       - Are travel times reasonable given distances?
       - Which high-population areas have inefficient service (slow despite short distances)
       - Zone efficiency rankings for areas with most residents

    ### Key Metrics Used:
    - **Travel Time**: `Tiempo_Viaje_Total_Minutos` (total door-to-door time)
    - **Origin Zone**: `Zona_Origen` (zone ID), `Zona_Origen_nombre` (zone name)
    - **Population**: `Poblacion_Origen` (number of residents in each origin zone)
    - **Destination**: `Zona_Destino_nombre` (POI name)
    - **Distance**: `Distancia_Viaje_Total_Km` (total trip distance)
    - **Transfers**: `Numero_Transbordos` (number of transfers required)

    ### Important:
    All percentages and statistics are **population-weighted**, meaning they reflect the actual number of residents affected, not just geographic coverage. This ensures policy decisions prioritize areas where the most people are impacted.
    """)