import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
    - `Zona_Origen` → Origin zone ID (unique identifier for each neighborhood/area)
    - `Zona_Origen_nombre` → Human-readable name of the origin zone/neighborhood
    - `Poblacion_Origen` → **Population living in the origin zone** (number of residents - used for weighting accessibility metrics)

    **Destination Information:**
    - `Zona_Destino_nombre` → Point of Interest (POI) destination name
    - `Zona_Destino` → Destination zone ID

    **Travel Time Metrics:**
    - `Tiempo_Viaje_Total_Minutos` → **Total door-to-door travel time** (main metric for accessibility)
    - `Tiempo_Trayecto_Minutos` → In-vehicle time only
    - `Tiempo_Acceso_Minutos` → Walking time from origin to bus stop
    - `Tiempo_Salida_Minutos` → Walking time from bus stop to destination
    - `Tiempo_Espera_Transbordo_Minutos` → Waiting time for transfers

    **Distance Metrics:**
    - `Distancia_Viaje_Total_Km` → Total trip distance (walking + vehicle)
    - `Distancia_Trayecto_Km` → In-vehicle distance

    **Service Quality:**
    - `Numero_Transbordos` → Number of transfers required
    - `Usa_Transporte_Publico` → Whether public transport is used (True/False)
    - `Frecuencia_Servicio` → Service headway (minutes between buses)

    ### Accessibility Categories:
    We classify connections into 4 categories based on **total travel time**:
    - 🟢 **Excellent**: < 30 minutes
    - 🟡 **Good**: 30-45 minutes
    - 🟠 **Fair**: 45-60 minutes
    - 🔴 **Poor**: > 60 minutes

    ### Key Calculations:

    1. **Population-weighted accessibility**: We multiply each origin zone's accessibility metrics by its population (`Poblacion_Origen`) to understand how many actual residents have good vs poor connections. This gives us the true impact on people, not just geographic coverage.

    2. **Distance efficiency**: We calculate average travel time per kilometer (Tiempo_Viaje_Total_Minutos / Distancia_Viaje_Total_Km) to identify where service is inefficient despite short distances.

    3. **POI accessibility score**: For each POI, we calculate the average travel time from all origin zones, weighted by the population of each zone. This shows which destinations are accessible to the most people.

    ### Important Notes:
    - Each row represents one possible origin-destination pair
    - **`Poblacion_Origen`** contains the actual population count for each origin zone
    - All percentages are **population-weighted** to reflect impact on actual residents
    - When we say "X% of population has good access", we mean X% of the total population across all origin zones
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

    # Filter by origin municipality
    unique_municipalities = sorted(
        df_filtered['Zona_Origen_nombre'].str.extract(r'\((.*?)\)')[0].dropna().unique().tolist())
    municipalities = ['All'] + unique_municipalities
    selected_municipality = st.sidebar.selectbox("Filter by Municipality", municipalities)

    if selected_municipality != 'All':
        df_filtered = df_filtered[df_filtered['Zona_Origen_nombre'].str.contains(selected_municipality, na=False)]

    # Calculate total population for weighting
    total_population = df_filtered.groupby('Zona_Origen')['Poblacion_Origen'].first().sum()

    st.sidebar.markdown(f"**Analyzing {len(df_filtered):,} origin-destination pairs**")
    st.sidebar.markdown(f"**Total population covered: {total_population:,.0f} residents**")

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
        # Calculate population-weighted percentage under 30 min
        df_under_30 = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] <= 30].groupby('Zona_Origen')[
            'Poblacion_Origen'].first().sum()
        under_30_pct = (df_under_30 / total_population * 100) if total_population > 0 else 0
        st.metric("Population < 30min", f"{under_30_pct:.1f}%")

    with col4:
        total_zones = df_filtered['Zona_Origen'].nunique()
        st.metric("Origin Zones", f"{total_zones}")

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

        # Accessibility categories
        df_filtered['Accessibility_Category'] = pd.cut(
            df_filtered['Tiempo_Viaje_Total_Minutos'],
            bins=[0, 30, 45, 60, float('inf')],
            labels=['🟢 Excellent (<30min)', '🟡 Good (30-45min)',
                    '🟠 Fair (45-60min)', '🔴 Poor (>60min)']
        )

        # Calculate POPULATION-WEIGHTED percentages
        pop_by_category = {}
        for category in ['🟢 Excellent (<30min)', '🟡 Good (30-45min)', '🟠 Fair (45-60min)', '🔴 Poor (>60min)']:
            category_data = df_filtered[df_filtered['Accessibility_Category'] == category]
            # Get unique zones in this category and sum their populations
            pop_in_category = category_data.groupby('Zona_Origen')['Poblacion_Origen'].first().sum()
            pop_by_category[category] = pop_in_category

        category_pcts = {k: (v / total_population * 100) if total_population > 0 else 0
                         for k, v in pop_by_category.items()}

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart with population-weighted percentages
            categories = list(pop_by_category.keys())
            populations = list(pop_by_category.values())
            percentages = [category_pcts[cat] for cat in categories]

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
                title="Distribution of Population by Accessibility",
                xaxis_title="Accessibility Category",
                yaxis_title="Population (residents)",
                showlegend=False,
                height=400
            )
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("### Summary")
            for cat in categories:
                pct = category_pcts[cat]
                pop = pop_by_category[cat]
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
            color_discrete_sequence=['#3498db']
        )
        fig2.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="30 min")
        fig2.add_vline(x=45, line_dash="dash", line_color="orange", annotation_text="45 min")
        fig2.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="60 min")
        fig2.update_layout(
            xaxis_title="Total Travel Time (minutes)",
            yaxis_title="Percentage of Connections (%)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Municipality comparison
        st.markdown("---")
        st.markdown("### Accessibility by Zone")
        st.markdown("*Average travel time and percentage of good connections by origin zone (sized by population)*")

        # Calculate zone-level statistics with population
        zone_stats = df_filtered.groupby('Zona_Origen_nombre').agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Zona_Origen': 'first',
            'Poblacion_Origen': 'first'
        }).reset_index()

        # Calculate % of good connections per zone
        good_connections_per_zone = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] <= 45].groupby(
            'Zona_Origen_nombre').size()
        total_connections_per_zone = df_filtered.groupby('Zona_Origen_nombre').size()
        zone_stats['Good_Connection_Pct'] = (good_connections_per_zone / total_connections_per_zone * 100).reindex(
            zone_stats['Zona_Origen_nombre'], fill_value=0).values

        zone_stats.columns = ['Zone', 'Avg Travel Time (min)', 'Zone_ID', 'Population', 'Good Connections (%)']
        zone_stats = zone_stats.sort_values('Avg Travel Time (min)')

        fig3 = px.scatter(
            zone_stats,
            x='Avg Travel Time (min)',
            y='Good Connections (%)',
            size='Population',
            text='Zone',
            color='Avg Travel Time (min)',
            color_continuous_scale='RdYlGn_r',
            size_max=60,
            hover_data={'Population': ':,.0f'}
        )
        fig3.update_traces(textposition='top center', textfont_size=8)
        fig3.update_layout(
            title="Zone Performance: Travel Time vs Quality (bubble size = population)",
            height=500,
            xaxis_title="Average Travel Time (minutes)",
            yaxis_title="% of Connections < 45 min"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # TAB 2: POI Analysis
    with tab2:
        st.subheader("Point of Interest (POI) Accessibility")
        st.markdown("*Which destinations are well-connected and which need improvement?*")

        # POI statistics with POPULATION weighting
        poi_accessibility = []
        for poi in df_filtered['Zona_Destino_nombre'].unique():
            poi_data = df_filtered[df_filtered['Zona_Destino_nombre'] == poi]

            # Get unique zones and their populations
            zone_pops = poi_data.groupby('Zona_Origen')['Poblacion_Origen'].first()
            total_pop_poi = zone_pops.sum()

            # Calculate population in each time category
            excellent_zones = poi_data[poi_data['Tiempo_Viaje_Total_Minutos'] <= 30]['Zona_Origen'].unique()
            good_zones = poi_data[(poi_data['Tiempo_Viaje_Total_Minutos'] > 30) &
                                  (poi_data['Tiempo_Viaje_Total_Minutos'] <= 45)]['Zona_Origen'].unique()
            poor_zones = poi_data[poi_data['Tiempo_Viaje_Total_Minutos'] > 60]['Zona_Origen'].unique()

            pop_excellent = zone_pops[zone_pops.index.isin(excellent_zones)].sum()
            pop_good = zone_pops[zone_pops.index.isin(good_zones)].sum()
            pop_poor = zone_pops[zone_pops.index.isin(poor_zones)].sum()

            # Calculate weighted average travel time
            poi_data_unique = poi_data.groupby('Zona_Origen').agg({
                'Tiempo_Viaje_Total_Minutos': 'mean',
                'Poblacion_Origen': 'first'
            })
            weighted_avg_time = (poi_data_unique['Tiempo_Viaje_Total_Minutos'] * poi_data_unique[
                'Poblacion_Origen']).sum() / total_pop_poi if total_pop_poi > 0 else 0

            poi_accessibility.append({
                'POI': poi,
                'Avg Time': weighted_avg_time,
                'Total Population': total_pop_poi,
                'Excellent (%)': (pop_excellent / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Good (%)': (pop_good / total_pop_poi * 100) if total_pop_poi > 0 else 0,
                'Poor (%)': (pop_poor / total_pop_poi * 100) if total_pop_poi > 0 else 0
            })

        poi_df = pd.DataFrame(poi_accessibility).sort_values('Avg Time')

        # Top and bottom POIs
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Best Connected POIs")
            best_pois = poi_df.head(5)
            fig4 = px.bar(
                best_pois,
                x='Avg Time',
                y='POI',
                orientation='h',
                color='Excellent (%)',
                color_continuous_scale='Greens',
                text='Avg Time'
            )
            fig4.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
            fig4.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.markdown("#### 🔴 Worst Connected POIs")
            worst_pois = poi_df.tail(5)
            fig5 = px.bar(
                worst_pois,
                x='Avg Time',
                y='POI',
                orientation='h',
                color='Poor (%)',
                color_continuous_scale='Reds',
                text='Avg Time'
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

        poi_df_sorted = poi_df.sort_values('Avg Time')

        fig6.add_trace(go.Bar(
            name='🟢 Excellent (<30min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Excellent (%)'],
            orientation='h',
            marker_color='#2ecc71',
            text=poi_df_sorted['Excellent (%)'].round(1),
            textposition='inside'
        ))

        fig6.add_trace(go.Bar(
            name='🟡 Good (30-45min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Good (%)'],
            orientation='h',
            marker_color='#f1c40f',
            text=poi_df_sorted['Good (%)'].round(1),
            textposition='inside'
        ))

        fig6.add_trace(go.Bar(
            name='🔴 Poor (>60min)',
            y=poi_df_sorted['POI'],
            x=poi_df_sorted['Poor (%)'],
            orientation='h',
            marker_color='#e74c3c',
            text=poi_df_sorted['Poor (%)'].round(1),
            textposition='inside'
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
        display_df = poi_df[['POI', 'Avg Time', 'Total Population', 'Excellent (%)', 'Good (%)', 'Poor (%)']].copy()
        st.dataframe(
            display_df.style.format({
                'Avg Time': '{:.1f} min',
                'Total Population': '{:,.0f}',
                'Excellent (%)': '{:.1f}%',
                'Good (%)': '{:.1f}%',
                'Poor (%)': '{:.1f}%'
            }).background_gradient(subset=['Avg Time'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    # TAB 3: Priority Areas
    with tab3:
        st.subheader("Priority Areas for Service Improvement")
        st.markdown("*Origin zones with poor accessibility affecting the most residents*")

        # Calculate zone-level statistics with POPULATION
        zone_stats = df_filtered.groupby(['Zona_Origen_nombre', 'Zona_Origen']).agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Zona_Destino': 'count',
            'Numero_Transbordos': 'mean',
            'Poblacion_Origen': 'first'
        }).reset_index()
        zone_stats.columns = ['Origin Zone', 'Zone_ID', 'Avg Travel Time', 'Destinations Served', 'Avg Transfers',
                              'Population']

        # Identify poor connections (>45min average) with high population
        poor_zones = zone_stats[zone_stats['Avg Travel Time'] > 45].sort_values('Population', ascending=False)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### 🔴 High-Population Zones with Worst Accessibility")
            st.markdown(
                f"**{len(poor_zones)} zones** with {poor_zones['Population'].sum():,.0f} residents have average travel times exceeding 45 minutes")

            top_poor = poor_zones.head(15)
            fig7 = px.bar(
                top_poor,
                x='Population',
                y='Origin Zone',
                orientation='h',
                color='Avg Travel Time',
                color_continuous_scale='Reds',
                hover_data=['Avg Transfers', 'Destinations Served'],
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

        with col2:
            st.markdown("#### Key Impact Metrics")

            avg_poor = poor_zones['Avg Travel Time'].mean()
            avg_transfers_poor = poor_zones['Avg Transfers'].mean()
            total_affected_pop = poor_zones['Population'].sum()
            pct_affected = (total_affected_pop / total_population * 100) if total_population > 0 else 0

            st.error(f"**Average travel time in poor zones:** {avg_poor:.1f} minutes")
            st.warning(f"**Average transfers needed:** {avg_transfers_poor:.2f}")
            st.info(f"**Population affected:** {total_affected_pop:,.0f} residents ({pct_affected:.1f}% of total)")

            # Top affected zones list
            st.markdown("#### Most Critical Zones")
            st.markdown("*(by population affected)*")
            for idx, row in poor_zones.head(5).iterrows():
                st.write(f"**{row['Origin Zone']}**")
                st.write(f"→ {row['Population']:,.0f} residents, {row['Avg Travel Time']:.1f} min avg")
                st.write("---")
                Issues
                ")

            avg_poor = poor_zones['Avg Travel Time'].mean()
            avg_transfers_poor = poor_zones['Avg Transfers'].mean()

            st.error(f"**Average travel time in poor zones:** {avg_poor:.1f} minutes")
            st.warning(f"**Average transfers needed:** {avg_transfers_poor:.2f}")
            st.info(
                f"**Zones affected:** {len(poor_zones)} out of {len(zone_stats)} ({len(poor_zones) / len(zone_stats) * 100:.1f}%)")

            # Municipality breakdown
            st.markdown("#### Affected Municipalities")
            poor_by_muni = poor_zones.groupby('Municipality').size().sort_values(ascending=False)

            for muni, count in poor_by_muni.head(5).items():
                total_in_muni = zone_stats[zone_stats['Municipality'] == muni].shape[0]
                pct = count / total_in_muni * 100
                st.write(f"**{muni}**: {count}/{total_in_muni} zones ({pct:.1f}%)")

        # Problem matrix: POI vs Origin zones with poor access (population-weighted)
        st.markdown("---")
        st.markdown("#### Problem Matrix: Which populations struggle to reach which POIs?")
        st.markdown("*Color intensity = population affected, Red = poor access (>60min)*")

        # Filter for problematic connections
        problem_connections = df_filtered[df_filtered['Tiempo_Viaje_Total_Minutos'] > 45].copy()

        if len(problem_connections) > 0:
            # Get zones with highest population impact
            zone_pop_impact = problem_connections.groupby('Zona_Origen_nombre').agg({
                'Poblacion_Origen': 'first'
            }).sort_values('Poblacion_Origen', ascending=False)
            top_problem_zones = zone_pop_impact.head(10).index

            # Get most problematic POIs
            top_problem_pois = problem_connections.groupby('Zona_Destino_nombre').agg({
                'Poblacion_Origen': 'sum'
            }).sort_values('Poblacion_Origen', ascending=False).head(8).index

            # Create pivot table with population weighting
            problem_matrix = problem_connections[
                (problem_connections['Zona_Origen_nombre'].isin(top_problem_zones)) &
                (problem_connections['Zona_Destino_nombre'].isin(top_problem_pois))
                ].pivot_table(
                values='Tiempo_Viaje_Total_Minutos',
                index='Zona_Origen_nombre',
                columns='Zona_Destino_nombre',
                aggfunc='mean'
            )

            fig8 = px.imshow(
                problem_matrix,
                color_continuous_scale='RdYlGn_r',
                aspect='auto',
                labels=dict(x="Destination POI", y="Origin Zone (by population)", color="Travel Time (min)")
            )
            fig8.update_layout(height=500)
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.success("No problematic connections found with current filters!")

    # TAB 4: Distance Efficiency
    with tab4:
        st.subheader("Distance vs Travel Time Efficiency")
        st.markdown("*Identifying where service is slow despite short distances*")

        # Calculate time per km
        df_filtered['Time_per_Km'] = df_filtered['Tiempo_Viaje_Total_Minutos'] / df_filtered['Distancia_Viaje_Total_Km']
        df_filtered['Time_per_Km'] = df_filtered['Time_per_Km'].replace([np.inf, -np.inf], np.nan)

        # Scatter plot: Distance vs Time
        fig9 = px.scatter(
            df_filtered,
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

        # Add reference line for ideal speed (e.g., 30 km/h average)
        max_dist = df_filtered['Distancia_Viaje_Total_Km'].max()
        fig9.add_trace(go.Scatter(
            x=[0, max_dist],
            y=[0, max_dist * 2],  # 30 km/h = 2 min/km
            mode='lines',
            name='Ideal (30 km/h avg)',
            line=dict(color='green', dash='dash')
        ))

        fig9.update_layout(
            title='Travel Time vs Distance (connections above the line are slower than ideal)',
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
            inefficient = (df_filtered['Time_per_Km'] > 3).sum() / len(df_filtered) * 100
            st.metric("Inefficient Connections", f"{inefficient:.1f}%",
                      help="Connections with >3 min/km (slower than 20 km/h)")

        # Municipality efficiency comparison
        st.markdown("---")
        st.markdown("#### Efficiency by Zone (Population-Weighted)")

        zone_efficiency = df_filtered.groupby(['Zona_Origen_nombre', 'Zona_Origen']).agg({
            'Time_per_Km': 'mean',
            'Distancia_Viaje_Total_Km': 'mean',
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Poblacion_Origen': 'first'
        }).reset_index()
        zone_efficiency.columns = ['Zone', 'Zone_ID', 'Min per Km', 'Avg Distance (km)', 'Avg Time (min)', 'Population']
        zone_efficiency['Effective Speed (km/h)'] = 60 / zone_efficiency['Min per Km']
        zone_efficiency = zone_efficiency.sort_values('Population', ascending=False).head(20)

        fig10 = px.bar(
            zone_efficiency.sort_values('Min per Km', ascending=False),
            x='Zone',
            y='Min per Km',
            color='Min per Km',
            color_continuous_scale='RdYlGn_r',
            hover_data={'Effective Speed (km/h)': ':.1f', 'Avg Distance (km)': ':.1f',
                        'Avg Time (min)': ':.1f', 'Population': ':,.0f'}
        )
        fig10.update_layout(
            title='Travel Time Efficiency by Zone (Top 20 by population, lower is better)',
            xaxis_title='Origin Zone',
            yaxis_title='Average Minutes per Kilometer',
            height=400
        )
        fig10.update_xaxes(tickangle=45)
        st.plotly_chart(fig10, use_container_width=True)

        # Data table
        st.markdown("#### Zone Efficiency Rankings (Top 20 by Population)")
        display_efficiency = zone_efficiency.sort_values('Min per Km')
        st.dataframe(
            display_efficiency[['Zone', 'Population', 'Min per Km', 'Avg Distance (km)',
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
       - What % of **population** has good/poor connections
       - Travel time distribution across the network
       - Zone-level performance comparison (weighted by population)

    2. **📍 POI Analysis**
       - Which destinations are well/poorly connected
       - % of **population** that can reach each POI easily
       - Accessibility quality breakdown by POI (population-weighted)

    3. **⚠️ Priority Areas**
       - Zones with worst accessibility and **highest population impact**
       - Which populations are most affected
       - Problem matrix showing which origin-destination pairs need improvement

    4. **📊 Distance Efficiency**
       - Are travel times reasonable given distances?
       - Which high-population areas have inefficient service
       - Zone efficiency rankings

    ### Key Metrics Used:
    - **Travel Time**: Tiempo_Viaje_Total_Minutos (door-to-door)
    - **Origin Zone**: Zona_Origen (zone ID), Zona_Origen_nombre (zone name)
    - **Population**: Poblacion_Origen (number of residents in each origin zone)
    - **Destination**: Zona_Destino_nombre (POI name)
    - **Distance**: Distancia_Viaje_Total_Km
    - **Transfers**: Numero_Transbordos

    ### Important:
    All percentages and statistics are **population-weighted**, meaning they reflect the actual number of residents affected, not just geographic coverage.
    """)