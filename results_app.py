import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Bizkaia Accessibility Analysis", layout="wide", page_icon="🚌")

# Title and introduction
st.title("🚌 Bizkaia Public Transport Accessibility Analysis")
st.markdown("""
**Comprehensive analysis of Bizkaibus service accessibility across zones and Points of Interest (POI)**  
*Evaluating door-to-door travel times, service coverage, and population access*
""")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)

    # Data cleaning
    df.columns = df.columns.str.strip()

    # Store original dataframe for calculations
    df_original = df.copy()

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Filter by public transport usage
    transport_filter = st.sidebar.radio(
        "Transport Mode",
        ["All trips", "Public transport only", "Without public transport"],
        index=1
    )

    if transport_filter == "Public transport only":
        df_filtered = df[df['Usa_Transporte_Publico'] == True].copy()
    elif transport_filter == "Without public transport":
        df_filtered = df[df['Usa_Transporte_Publico'] == False].copy()
    else:
        df_filtered = df.copy()

    # Filter by origin municipality - FIXED to show actual unique values
    unique_municipalities = sorted(df_filtered['Poblacion_Origen'].dropna().unique().tolist())
    municipalities = ['All'] + unique_municipalities
    selected_municipality = st.sidebar.selectbox("Origin Municipality", municipalities)

    if selected_municipality != 'All':
        df_filtered = df_filtered[df_filtered['Poblacion_Origen'] == selected_municipality]

    # Main metrics
    st.header("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_travel_time = df_filtered['Tiempo_Viaje_Total_Minutos'].mean()
        st.metric("Avg Travel Time", f"{avg_travel_time:.1f} min")

    with col2:
        avg_transfers = df_filtered['Numero_Transbordos'].mean()
        st.metric("Avg Transfers", f"{avg_transfers:.2f}")

    with col3:
        pt_usage = (df['Usa_Transporte_Publico'].sum() / len(df) * 100)
        st.metric("PT Usage Rate", f"{pt_usage:.1f}%")

    with col4:
        total_zones = df_filtered['Zona_Origen'].nunique()
        st.metric("Total Origin Zones", f"{total_zones}")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Zone Accessibility",
        "📍 POI Analysis",
        "⏱️ Travel Time Analysis",
        "🔄 Transfer Analysis",
        "👥 Population Coverage",
        "📈 Service Quality"
    ])

    # TAB 1: Zone Accessibility
    with tab1:
        st.subheader("Zone Accessibility Rankings")

        col1, col2 = st.columns(2)

        with col1:
            # Best connected zones (lowest average travel time)
            zone_stats = df_filtered.groupby('Zona_Origen_nombre').agg({
                'Tiempo_Viaje_Total_Minutos': 'mean',
                'Numero_Transbordos': 'mean',
                'Poblacion_Origen': 'first'
            }).reset_index()
            zone_stats = zone_stats.sort_values('Tiempo_Viaje_Total_Minutos')

            st.markdown("**🟢 Top 10 Best Connected Zones**")
            best_zones = zone_stats.head(10)
            fig1 = px.bar(best_zones,
                          x='Tiempo_Viaje_Total_Minutos',
                          y='Zona_Origen_nombre',
                          orientation='h',
                          color='Numero_Transbordos',
                          color_continuous_scale='Greens_r',
                          labels={'Tiempo_Viaje_Total_Minutos': 'Avg Travel Time (min)',
                                  'Zona_Origen_nombre': 'Origin Zone',
                                  'Numero_Transbordos': 'Avg Transfers'})
            fig1.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Worst connected zones (highest average travel time)
            st.markdown("**🔴 Top 10 Least Connected Zones**")
            worst_zones = zone_stats.tail(10)
            fig2 = px.bar(worst_zones,
                          x='Tiempo_Viaje_Total_Minutos',
                          y='Zona_Origen_nombre',
                          orientation='h',
                          color='Numero_Transbordos',
                          color_continuous_scale='Reds',
                          labels={'Tiempo_Viaje_Total_Minutos': 'Avg Travel Time (min)',
                                  'Zona_Origen_nombre': 'Origin Zone',
                                  'Numero_Transbordos': 'Avg Transfers'})
            fig2.update_layout(height=400, yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig2, use_container_width=True)

        # Municipality comparison
        st.markdown("**📊 Municipality Accessibility Comparison**")
        muni_stats = df_filtered.groupby('Poblacion_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Numero_Transbordos': 'mean',
            'Zona_Origen': 'nunique'
        }).reset_index()
        muni_stats.columns = ['Municipality', 'Avg Travel Time (min)', 'Avg Transfers', 'Number of Zones']

        fig3 = px.scatter(muni_stats,
                          x='Avg Travel Time (min)',
                          y='Avg Transfers',
                          size='Number of Zones',
                          text='Municipality',
                          color='Avg Travel Time (min)',
                          color_continuous_scale='RdYlGn_r')
        fig3.update_traces(textposition='top center')
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

    # TAB 2: POI Analysis
    with tab2:
        st.subheader("Point of Interest (POI) Accessibility")

        # POI accessibility ranking
        poi_stats = df_filtered.groupby('Zona_Destino_nombre').agg({
            'Tiempo_Viaje_Total_Minutos': ['mean', 'std', 'min', 'max'],
            'Numero_Transbordos': 'mean',
            'Zona_Origen': 'count'
        }).reset_index()
        poi_stats.columns = ['POI', 'Avg Time', 'Std Time', 'Min Time', 'Max Time', 'Avg Transfers', 'Connections']
        poi_stats = poi_stats.sort_values('Avg Time')

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**POI Accessibility Ranking (by average travel time)**")
            fig4 = px.bar(poi_stats,
                          x='Avg Time',
                          y='POI',
                          orientation='h',
                          color='Connections',
                          color_continuous_scale='Blues',
                          hover_data=['Avg Transfers', 'Min Time', 'Max Time'])
            fig4.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.markdown("**POI Statistics**")
            st.dataframe(poi_stats[['POI', 'Avg Time', 'Avg Transfers', 'Connections']]
                         .sort_values('Avg Time')
                         .style.format({'Avg Time': '{:.1f}', 'Avg Transfers': '{:.2f}'})
                         .background_gradient(subset=['Avg Time'], cmap='RdYlGn_r'),
                         height=600)

        # POI accessibility distribution
        st.markdown("**📊 Travel Time Distribution by POI**")
        fig5 = px.box(df_filtered,
                      x='Zona_Destino_nombre',
                      y='Tiempo_Viaje_Total_Minutos',
                      color='Zona_Destino_nombre',
                      labels={'Tiempo_Viaje_Total_Minutos': 'Travel Time (min)',
                              'Zona_Destino_nombre': 'POI'})
        fig5.update_layout(height=500, showlegend=False)
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)

    # TAB 3: Travel Time Analysis
    with tab3:
        st.subheader("Travel Time Components Analysis")

        # Calculate time components percentages
        df_filtered['Access_Pct'] = (
                    df_filtered['Tiempo_Acceso_Minutos'] / df_filtered['Tiempo_Viaje_Total_Minutos'] * 100)
        df_filtered['InVehicle_Pct'] = (
                    df_filtered['Tiempo_Trayecto_Minutos'] / df_filtered['Tiempo_Viaje_Total_Minutos'] * 100)
        df_filtered['Egress_Pct'] = (
                    df_filtered['Tiempo_Salida_Minutos'] / df_filtered['Tiempo_Viaje_Total_Minutos'] * 100)
        df_filtered['Wait_Pct'] = (
                    df_filtered['Tiempo_Espera_Transbordo_Minutos'] / df_filtered['Tiempo_Viaje_Total_Minutos'] * 100)

        col1, col2 = st.columns(2)

        with col1:
            # Average time components
            avg_components = {
                'Access (Walk to stop)': df_filtered['Tiempo_Acceso_Minutos'].mean(),
                'In-Vehicle': df_filtered['Tiempo_Trayecto_Minutos'].mean(),
                'Transfer Wait': df_filtered['Tiempo_Espera_Transbordo_Minutos'].mean(),
                'Egress (Stop to destination)': df_filtered['Tiempo_Salida_Minutos'].mean()
            }

            fig6 = go.Figure(data=[go.Pie(labels=list(avg_components.keys()),
                                          values=list(avg_components.values()),
                                          hole=.3)])
            fig6.update_layout(title="Average Travel Time Breakdown", height=400)
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            # Travel time distribution with percentages - FIXED
            st.markdown("**Travel Time Distribution**")
            hist_data = df_filtered['Tiempo_Viaje_Total_Minutos'].dropna()
            fig7 = px.histogram(df_filtered,
                                x='Tiempo_Viaje_Total_Minutos',
                                nbins=30,
                                color_discrete_sequence=['#1f77b4'],
                                histnorm='percent')
            fig7.update_layout(xaxis_title="Total Travel Time (min)",
                               yaxis_title="Percentage of Trips (%)",
                               height=400)
            st.plotly_chart(fig7, use_container_width=True)

        # Travel time vs distance
        st.markdown("**⚡ Travel Speed Analysis**")
        df_filtered['Avg_Speed_Kmh'] = (df_filtered['Distancia_Viaje_Total_Km'] /
                                        (df_filtered['Tiempo_Viaje_Total_Minutos'] / 60))

        fig8 = px.scatter(df_filtered,
                          x='Distancia_Viaje_Total_Km',
                          y='Tiempo_Viaje_Total_Minutos',
                          color='Numero_Transbordos',
                          hover_data=['Zona_Origen_nombre', 'Zona_Destino_nombre', 'Avg_Speed_Kmh'],
                          labels={'Distancia_Viaje_Total_Km': 'Distance (km)',
                                  'Tiempo_Viaje_Total_Minutos': 'Travel Time (min)',
                                  'Numero_Transbordos': 'Transfers'})
        fig8.update_layout(height=500)
        st.plotly_chart(fig8, use_container_width=True)

        # Accessibility categories with percentages - FIXED
        st.markdown("**🎯 Accessibility Categories**")
        df_filtered['Accessibility_Category'] = pd.cut(df_filtered['Tiempo_Viaje_Total_Minutos'],
                                                       bins=[0, 30, 45, 60, float('inf')],
                                                       labels=['Excellent (<30min)', 'Good (30-45min)',
                                                               'Fair (45-60min)', 'Poor (>60min)'])

        category_counts = df_filtered['Accessibility_Category'].value_counts()
        category_pcts = (category_counts / len(df_filtered) * 100).round(1)

        # Create labels with both count and percentage
        labels_with_pct = [f"{cat}<br>{count} trips ({category_pcts[cat]}%)"
                           for cat, count in category_counts.items()]

        fig9 = px.bar(x=category_counts.index,
                      y=category_counts.values,
                      color=category_counts.index,
                      color_discrete_map={'Excellent (<30min)': 'green',
                                          'Good (30-45min)': 'lightgreen',
                                          'Fair (45-60min)': 'orange',
                                          'Poor (>60min)': 'red'},
                      text=[f"{count}<br>({category_pcts[cat]}%)" for cat, count in category_counts.items()])
        fig9.update_layout(title="Distribution of Accessibility Categories",
                           xaxis_title="Category",
                           yaxis_title="Number of Connections",
                           showlegend=False,
                           height=400)
        fig9.update_traces(textposition='outside')
        st.plotly_chart(fig9, use_container_width=True)

    # TAB 4: Transfer Analysis
    with tab4:
        st.subheader("Transfer and Service Quality Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Transfer distribution with percentages - FIXED
            transfer_dist = df_filtered['Numero_Transbordos'].value_counts().sort_index()
            transfer_pcts = (transfer_dist / len(df_filtered) * 100).round(1)

            fig10 = px.bar(x=transfer_dist.index,
                           y=transfer_dist.values,
                           labels={'x': 'Number of Transfers', 'y': 'Number of Trips'},
                           color=transfer_dist.index,
                           color_continuous_scale='Reds',
                           text=[f"{count}<br>({transfer_pcts[idx]}%)" for idx, count in transfer_dist.items()])
            fig10.update_layout(title="Transfer Distribution", height=400, showlegend=False)
            fig10.update_traces(textposition='outside')
            st.plotly_chart(fig10, use_container_width=True)

        with col2:
            # Service frequency analysis with percentages - FIXED
            if df_filtered['Frecuencia_Servicio'].notna().any():
                fig11 = px.histogram(df_filtered,
                                     x='Frecuencia_Servicio',
                                     nbins=20,
                                     histnorm='percent',
                                     labels={'Frecuencia_Servicio': 'Service Headway (min)'})
                fig11.update_layout(title="Service Frequency Distribution",
                                    yaxis_title="Percentage of Trips (%)",
                                    height=400)
                st.plotly_chart(fig11, use_container_width=True)
            else:
                st.info("Service frequency data not available")

        # Impact of transfers on travel time
        st.markdown("**🔄 Impact of Transfers on Travel Time**")
        transfer_impact = df_filtered.groupby('Numero_Transbordos').agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Tiempo_Espera_Transbordo_Minutos': 'mean',
            'Zona_Origen': 'count'
        }).reset_index()
        transfer_impact.columns = ['Transfers', 'Avg Total Time', 'Avg Wait Time', 'Count']

        fig12 = go.Figure()
        fig12.add_trace(go.Bar(name='Average Total Travel Time',
                               x=transfer_impact['Transfers'],
                               y=transfer_impact['Avg Total Time']))
        fig12.add_trace(go.Bar(name='Average Transfer Wait Time',
                               x=transfer_impact['Transfers'],
                               y=transfer_impact['Avg Wait Time']))
        fig12.update_layout(barmode='group',
                            title="Travel Time by Number of Transfers",
                            xaxis_title="Number of Transfers",
                            yaxis_title="Time (minutes)",
                            height=400)
        st.plotly_chart(fig12, use_container_width=True)

        # Direct connections analysis
        st.markdown("**🎯 Direct vs. Transfer Connections**")
        direct_trips = df_filtered[df_filtered['Numero_Transbordos'] == 0]
        transfer_trips = df_filtered[df_filtered['Numero_Transbordos'] > 0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Direct Connections",
                      f"{len(direct_trips)} ({len(direct_trips) / len(df_filtered) * 100:.1f}%)")
        with col2:
            if len(direct_trips) > 0:
                st.metric("Avg Time - Direct",
                          f"{direct_trips['Tiempo_Viaje_Total_Minutos'].mean():.1f} min")
            else:
                st.metric("Avg Time - Direct", "N/A")
        with col3:
            if len(transfer_trips) > 0:
                st.metric("Avg Time - With Transfers",
                          f"{transfer_trips['Tiempo_Viaje_Total_Minutos'].mean():.1f} min")
            else:
                st.metric("Avg Time - With Transfers", "N/A")

    # TAB 5: Population Coverage
    with tab5:
        st.subheader("Population Accessibility Coverage")

        # Population by municipality with accessibility metrics
        pop_coverage = df_filtered.groupby('Poblacion_Origen').agg({
            'Zona_Origen': 'nunique',
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Numero_Transbordos': 'mean'
        }).reset_index()
        pop_coverage.columns = ['Municipality', 'Zones Served', 'Avg Travel Time', 'Avg Transfers']

        st.markdown("**🏘️ Municipality Service Coverage**")
        fig13 = px.bar(pop_coverage,
                       x='Municipality',
                       y='Zones Served',
                       color='Avg Travel Time',
                       color_continuous_scale='RdYlGn_r',
                       hover_data=['Avg Transfers'])
        fig13.update_layout(height=400)
        fig13.update_xaxes(tickangle=45)
        st.plotly_chart(fig13, use_container_width=True)

        # Accessibility by population (if we had population data)
        st.markdown("**📊 Accessibility Heatmap by Origin-Destination**")

        # Create pivot table for heatmap
        heatmap_data = df_filtered.pivot_table(
            values='Tiempo_Viaje_Total_Minutos',
            index='Poblacion_Origen',
            columns='Zona_Destino_nombre',
            aggfunc='mean'
        )

        fig14 = px.imshow(heatmap_data,
                          labels=dict(x="Destination POI", y="Origin Municipality",
                                      color="Travel Time (min)"),
                          color_continuous_scale='RdYlGn_r',
                          aspect="auto")
        fig14.update_layout(height=500)
        st.plotly_chart(fig14, use_container_width=True)

        # Best and worst connected population centers
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Best Connected Municipalities**")
            best_munis = pop_coverage.sort_values('Avg Travel Time').head(5)
            st.dataframe(best_munis.style.format({
                'Avg Travel Time': '{:.1f} min',
                'Avg Transfers': '{:.2f}'
            }), hide_index=True)

        with col2:
            st.markdown("**🔴 Least Connected Municipalities**")
            worst_munis = pop_coverage.sort_values('Avg Travel Time', ascending=False).head(5)
            st.dataframe(worst_munis.style.format({
                'Avg Travel Time': '{:.1f} min',
                'Avg Transfers': '{:.2f}'
            }), hide_index=True)

    # TAB 6: Service Quality - FIXED
    with tab6:
        st.subheader("Service Quality Indicators")

        # Overall service quality score (0-100) - FIXED calculations
        # Handle division by zero and NaN values
        max_travel = df_filtered['Tiempo_Viaje_Total_Minutos'].max()
        max_transfers = df_filtered['Numero_Transbordos'].max()
        max_walking = (df_filtered['Tiempo_Acceso_Minutos'] + df_filtered['Tiempo_Salida_Minutos']).max()

        # Avoid division by zero
        travel_score = 0 if max_travel == 0 else (df_filtered['Tiempo_Viaje_Total_Minutos'] / max_travel * 40)
        transfer_score = 0 if max_transfers == 0 else (df_filtered['Numero_Transbordos'] / max_transfers * 30)
        walking_score = 0 if max_walking == 0 else (
                    (df_filtered['Tiempo_Acceso_Minutos'] + df_filtered['Tiempo_Salida_Minutos']) / max_walking * 30)

        df_filtered['Quality_Score'] = 100 - travel_score - transfer_score - walking_score

        # Replace any NaN or infinite values
        df_filtered['Quality_Score'] = df_filtered['Quality_Score'].replace([np.inf, -np.inf], np.nan)
        df_filtered['Quality_Score'] = df_filtered['Quality_Score'].fillna(0)

        avg_quality = df_filtered['Quality_Score'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Service Quality Score", f"{avg_quality:.1f}/100")
        with col2:
            excellent_pct = (df_filtered['Quality_Score'] >= 70).sum() / len(df_filtered) * 100
            st.metric("Excellent Quality Connections", f"{excellent_pct:.1f}%")
        with col3:
            poor_pct = (df_filtered['Quality_Score'] < 40).sum() / len(df_filtered) * 100
            st.metric("Poor Quality Connections", f"{poor_pct:.1f}%")

        # Quality distribution with percentages - FIXED
        st.markdown("**📊 Service Quality Distribution**")
        fig15 = px.histogram(df_filtered,
                             x='Quality_Score',
                             nbins=20,
                             histnorm='percent',
                             color_discrete_sequence=['#2ca02c'])
        fig15.update_layout(title="Distribution of Service Quality Scores",
                            xaxis_title="Quality Score (0-100)",
                            yaxis_title="Percentage of Connections (%)",
                            height=400)
        st.plotly_chart(fig15, use_container_width=True)

        # Quality by municipality
        st.markdown("**🏆 Service Quality by Municipality**")
        quality_by_muni = df_filtered.groupby('Poblacion_Origen')['Quality_Score'].mean().sort_values(ascending=False)

        fig16 = px.bar(x=quality_by_muni.index,
                       y=quality_by_muni.values,
                       color=quality_by_muni.values,
                       color_continuous_scale='RdYlGn')
        fig16.update_layout(title="Average Quality Score by Origin Municipality",
                            xaxis_title="Municipality",
                            yaxis_title="Quality Score",
                            showlegend=False,
                            height=400)
        fig16.update_xaxes(tickangle=45)
        st.plotly_chart(fig16, use_container_width=True)

        # Walking distances analysis
        st.markdown("**🚶 Walking Distance Analysis (Access + Egress)**")
        df_filtered['Total_Walking_Distance'] = (df_filtered['Distancia_Acceso_Metros'] +
                                                 df_filtered['Distancia_Salida_Metros'])

        col1, col2 = st.columns(2)

        with col1:
            avg_walking = df_filtered['Total_Walking_Distance'].mean()
            st.metric("Average Walking Distance", f"{avg_walking:.0f} m")

            walking_categories = pd.cut(df_filtered['Total_Walking_Distance'],
                                        bins=[0, 500, 1000, 2000, float('inf')],
                                        labels=['<500m', '500-1000m', '1-2km', '>2km'])
            walking_dist = walking_categories.value_counts()
            walking_pcts = (walking_dist / len(df_filtered) * 100).round(1)

            # Add percentages to labels
            labels_walk = [f"{cat}: {walking_dist[cat]} ({walking_pcts[cat]}%)" for cat in walking_dist.index]

            fig17 = px.pie(values=walking_dist.values,
                           names=labels_walk,
                           title="Walking Distance Distribution")
            st.plotly_chart(fig17, use_container_width=True)

        with col2:
            # Walking time analysis
            df_filtered['Total_Walking_Time'] = (df_filtered['Tiempo_Acceso_Minutos'] +
                                                 df_filtered['Tiempo_Salida_Minutos'])
            avg_walking_time = df_filtered['Total_Walking_Time'].mean()
            st.metric("Average Walking Time", f"{avg_walking_time:.1f} min")

            fig18 = px.box(df_filtered,
                           y='Total_Walking_Time',
                           title="Walking Time Distribution")
            fig18.update_layout(yaxis_title="Total Walking Time (min)")
            st.plotly_chart(fig18, use_container_width=True)

    # Download section
    st.header("📥 Download Analysis Report")

    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Origin Zones',
            'Total POIs',
            'Average Travel Time (min)',
            'Average Transfers',
            'Average Walking Distance (m)',
            'Public Transport Usage (%)',
            'Direct Connections (%)',
            'Overall Quality Score'
        ],
        'Value': [
            df_filtered['Zona_Origen'].nunique(),
            df_filtered['Zona_Destino'].nunique(),
            f"{df_filtered['Tiempo_Viaje_Total_Minutos'].mean():.1f}",
            f"{df_filtered['Numero_Transbordos'].mean():.2f}",
            f"{df_filtered['Total_Walking_Distance'].mean():.0f}",
            f"{(df['Usa_Transporte_Publico'].sum() / len(df) * 100):.1f}",
            f"{(df_filtered['Numero_Transbordos'] == 0).sum() / len(df_filtered) * 100:.1f}",
            f"{avg_quality:.1f}"
        ]
    })

    st.dataframe(summary_stats, hide_index=True, use_container_width=True)

    # Convert to CSV for download
    csv = summary_stats.to_csv(index=False)
    st.download_button(
        label="Download Summary Statistics (CSV)",
        data=csv,
        file_name="bizkaia_accessibility_summary.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload an Excel file to begin the analysis")

    st.markdown("""
    ### Expected Data Structure
    The uploaded file should contain the following columns:
    - Origin and destination zone information
    - Public transport usage indicator
    - Travel time components (access, in-vehicle, transfer wait, egress)
    - Distance metrics
    - Number of transfers
    - Service frequency

    This analysis will provide insights into:
    - ✅ Best and worst connected zones
    - 📍 POI accessibility rankings
    - ⏱️ Travel time breakdowns
    - 🔄 Transfer patterns and impacts
    - 👥 Population coverage analysis
    - 📈 Overall service quality metrics
    """)