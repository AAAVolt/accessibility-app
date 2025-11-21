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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class AccessibilityConfig:
    """Configuration for accessibility analysis"""
    EXCELLENT_THRESHOLD: int = 30
    GOOD_THRESHOLD: int = 45
    FAIR_THRESHOLD: int = 60
    MODERATE_THRESHOLD: int = 75
    POOR_THRESHOLD: int = 90

    COLORS = {
        'excellent': '#1A9850',
        'good': '#A6D96A',
        'fair': '#FEE08B',
        'moderate': '#FDAE61',
        'poor': '#F46D43',
        'very_poor': '#D73027',
        'public_transport': '#1f77b4',
        'private_transport': '#ff7f0e',
        'difference': '#2ca02c'
    }

    CATEGORY_LABELS = {
        'excellent': '🟢 Excellent (<30min)',
        'good': '🟡 Good (30-45min)',
        'fair': '🟠 Fair (45-60min)',
        'moderate': '🔶 Moderate (60-75min)',
        'poor': '🔴 Poor (75-90min)',
        'very_poor': '⚫ Very Poor (>90min)'
    }


CONFIG = AccessibilityConfig()


# ============================================================================
# DATA PROCESSING CLASSES
# ============================================================================

class AccessibilityAnalyzer:
    """Main class for accessibility analysis"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Calculate total travel time in minutes
        self.df['Tiempo_Total_Minutos'] = self.df['Tiempo_Viaje']  # Using Tiempo_Trayecto as total time

        # Filter out zones with travel times > 999 minutes (unrealistic/invalid data)
        self.df = self.df[self.df['Tiempo_Total_Minutos'] <= 999]

        # Check for private transport columns
        self.has_private_transport = 'TT_Coche_NoPico' in self.df.columns and 'Distancia_km' in self.df.columns

        if self.has_private_transport:
            # Convert car travel time to minutes if needed
            self.df['TT_Coche_Minutos'] = self.df['TT_Coche_NoPico']  # Assuming already in minutes
            # Calculate time difference (positive means public transport is slower)
            self.df['Diferencia_Tiempo'] = self.df['Tiempo_Total_Minutos'] - self.df['TT_Coche_Minutos']

        self.zone_populations = self._calculate_zone_populations()
        self.total_population = self.zone_populations.sum()

    def _calculate_zone_populations(self) -> pd.Series:
        """Calculate population by zone"""
        return self.df.groupby('Zona_Origen')['Poblacion'].first()

    def categorize_accessibility(self, travel_times: pd.Series) -> pd.Series:
        """Categorize travel times into accessibility levels"""
        return pd.cut(
            travel_times,
            bins=[0, CONFIG.EXCELLENT_THRESHOLD, CONFIG.GOOD_THRESHOLD,
                  CONFIG.FAIR_THRESHOLD, CONFIG.MODERATE_THRESHOLD,
                  CONFIG.POOR_THRESHOLD, float('inf')],
            labels=list(CONFIG.CATEGORY_LABELS.values())
        )

    def get_zone_best_access(self) -> pd.DataFrame:
        """Get best accessibility for each zone"""
        zone_best = self.df.groupby('Zona_Origen').agg({
            'Tiempo_Total_Minutos': 'min',
            'Poblacion': 'first',
            'Nombre_Origen': 'first'
        }).reset_index()

        zone_best['Best_Category'] = self.categorize_accessibility(
            zone_best['Tiempo_Total_Minutos']
        )
        return zone_best

    def get_population_by_category(self) -> Tuple[pd.Series, pd.Series]:
        """Get population distribution by accessibility category"""
        zone_best = self.get_zone_best_access()
        pop_by_category = zone_best.groupby('Best_Category')['Poblacion'].sum()
        category_pcts = (pop_by_category / self.total_population * 100) if self.total_population > 0 else pd.Series()
        return pop_by_category, category_pcts

    def get_poi_accessibility(self) -> pd.DataFrame:
        """Calculate accessibility metrics for each POI"""
        poi_data = []

        for poi in self.df['Nombre_Destino'].unique():
            poi_subset = self.df[self.df['Nombre_Destino'] == poi]
            zone_data = poi_subset.groupby('Zona_Origen').agg({
                'Poblacion': 'first',
                'Tiempo_Total_Minutos': 'mean'
            })

            total_pop = zone_data['Poblacion'].sum()
            if total_pop == 0:
                continue

            # Calculate population in each category
            categories = {}
            categories['excellent'] = zone_data[zone_data['Tiempo_Total_Minutos'] <= CONFIG.EXCELLENT_THRESHOLD][
                'Poblacion'].sum()
            categories['good'] = zone_data[(zone_data['Tiempo_Total_Minutos'] > CONFIG.EXCELLENT_THRESHOLD) & (
                    zone_data['Tiempo_Total_Minutos'] <= CONFIG.GOOD_THRESHOLD)]['Poblacion'].sum()
            categories['fair'] = zone_data[(zone_data['Tiempo_Total_Minutos'] > CONFIG.GOOD_THRESHOLD) & (
                    zone_data['Tiempo_Total_Minutos'] <= CONFIG.FAIR_THRESHOLD)]['Poblacion'].sum()
            categories['moderate'] = zone_data[(zone_data['Tiempo_Total_Minutos'] > CONFIG.FAIR_THRESHOLD) & (
                    zone_data['Tiempo_Total_Minutos'] <= CONFIG.MODERATE_THRESHOLD)][
                'Poblacion'].sum()
            categories['poor'] = zone_data[(zone_data['Tiempo_Total_Minutos'] > CONFIG.MODERATE_THRESHOLD) & (
                    zone_data['Tiempo_Total_Minutos'] <= CONFIG.POOR_THRESHOLD)][
                'Poblacion'].sum()
            categories['very_poor'] = zone_data[zone_data['Tiempo_Total_Minutos'] > CONFIG.POOR_THRESHOLD][
                'Poblacion'].sum()

            weighted_avg_time = (zone_data['Tiempo_Total_Minutos'] * zone_data[
                'Poblacion']).sum() / total_pop

            poi_data.append({
                'POI': poi,
                'Avg Time (weighted)': weighted_avg_time,
                'Total Population': total_pop,
                'Excellent (%)': (categories['excellent'] / total_pop * 100),
                'Good (%)': (categories['good'] / total_pop * 100),
                'Fair (%)': (categories['fair'] / total_pop * 100),
                'Moderate (%)': (categories['moderate'] / total_pop * 100),
                'Poor (%)': (categories['poor'] / total_pop * 100),
                'Very Poor (%)': (categories['very_poor'] / total_pop * 100)
            })

        return pd.DataFrame(poi_data).sort_values('Avg Time (weighted)')

    def get_priority_zones(self, threshold: float = 45) -> pd.DataFrame:
        """Get zones with poor accessibility for prioritization"""
        zone_metrics = self.df.groupby('Zona_Origen').agg({
            'Tiempo_Total_Minutos': 'mean',
            'Nombre_Origen': 'first',
            'Destino_Óptimo': 'count',
            'Num_Transbordos': 'mean',
            'Poblacion': 'first'
        }).reset_index()

        zone_metrics.columns = ['Zone_ID', 'Avg Travel Time', 'Zone Name', 'POIs Served', 'Avg Transfers', 'Population']

        return zone_metrics[zone_metrics['Avg Travel Time'] > threshold].sort_values('Population', ascending=False)

    def get_transport_comparison(self) -> pd.DataFrame:
        """Get comparison between public and private transport"""
        if not self.has_private_transport:
            return pd.DataFrame()

        comparison_data = []

        for poi in self.df['Nombre_Destino'].unique():
            poi_subset = self.df[self.df['Nombre_Destino'] == poi]
            zone_data = poi_subset.groupby('Zona_Origen').agg({
                'Poblacion': 'first',
                'Tiempo_Total_Minutos': 'mean',
                'TT_Coche_Minutos': 'mean',
                'Distancia_km': 'mean',
                'Diferencia_Tiempo': 'mean'
            })

            total_pop = zone_data['Poblacion'].sum()
            if total_pop == 0:
                continue

            # Calculate weighted averages
            weighted_public = (zone_data['Tiempo_Total_Minutos'] * zone_data['Poblacion']).sum() / total_pop
            weighted_private = (zone_data['TT_Coche_Minutos'] * zone_data['Poblacion']).sum() / total_pop
            weighted_distance = (zone_data['Distancia_km'] * zone_data['Poblacion']).sum() / total_pop
            weighted_difference = (zone_data['Diferencia_Tiempo'] * zone_data['Poblacion']).sum() / total_pop

            comparison_data.append({
                'POI': poi,
                'Public Transport (min)': weighted_public,
                'Private Transport (min)': weighted_private,
                'Distance (km)': weighted_distance,
                'Time Difference (min)': weighted_difference,
                'Total Population': total_pop
            })

        return pd.DataFrame(comparison_data).sort_values('Time Difference (min)', ascending=False)

    def get_geographic_differences(self, geo_column: str) -> pd.DataFrame:
        """Calculate geographic differences between public and private transport"""
        if not self.has_private_transport:
            return pd.DataFrame()

        # Get zone-level data with geographic info
        zone_geo = self.df[['Zona_Origen', geo_column]].drop_duplicates()
        zone_transport = self.df.groupby('Zona_Origen').agg({
            'Poblacion': 'first',
            'Tiempo_Total_Minutos': 'mean',
            'TT_Coche_Minutos': 'mean',
            'Distancia_km': 'mean',
            'Diferencia_Tiempo': 'mean'
        }).reset_index()

        analysis_data = zone_transport.merge(zone_geo, on='Zona_Origen')

        # Group by geographic unit and calculate weighted averages
        geo_analysis = []
        for geo_unit in analysis_data[geo_column].unique():
            unit_data = analysis_data[analysis_data[geo_column] == geo_unit]
            total_pop = unit_data['Poblacion'].sum()

            if total_pop == 0:
                continue

            weighted_public = (unit_data['Tiempo_Total_Minutos'] * unit_data['Poblacion']).sum() / total_pop
            weighted_private = (unit_data['TT_Coche_Minutos'] * unit_data['Poblacion']).sum() / total_pop
            weighted_distance = (unit_data['Distancia_km'] * unit_data['Poblacion']).sum() / total_pop
            weighted_difference = (unit_data['Diferencia_Tiempo'] * unit_data['Poblacion']).sum() / total_pop

            geo_analysis.append({
                'Geographic Unit': geo_unit,
                'Public Transport (min)': weighted_public,
                'Private Transport (min)': weighted_private,
                'Distance (km)': weighted_distance,
                'Time Difference (min)': weighted_difference,
                'Total Population': total_pop,
                'Zones': len(unit_data)
            })

        return pd.DataFrame(geo_analysis).sort_values('Time Difference (min)', ascending=False)


# ============================================================================
# VISUALIZATION CLASSES
# ============================================================================

class ChartStyler:
    """Handles consistent chart styling"""

    @staticmethod
    def apply_standard_styling(fig, title: str, x_title: str, y_title: str,
                               height: int = 700, width: int = 1200):
        """Apply consistent styling to charts with minimum 16px font sizes"""
        fig.update_layout(
            title=dict(text=title, font=dict(size=24)),
            xaxis=dict(
                title=dict(text=x_title, font=dict(size=18)),
                tickfont=dict(size=16)
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=18)),
                tickfont=dict(size=16)
            ),
            legend=dict(font=dict(size=16)),
            height=height,
            width=width,
            margin=dict(l=200, r=100, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.4)'
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.4)'
        )

        return fig


class ExportManager:
    """Handles data export functionality"""

    @staticmethod
    def generate_detailed_report(analyzer: AccessibilityAnalyzer) -> BytesIO:
        """Generate comprehensive Excel report"""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Zone-level accessibility
            zone_best = analyzer.get_zone_best_access()
            zone_best.to_excel(writer, sheet_name='Zone_Accessibility', index=False)

            # POI analysis
            poi_analysis = analyzer.get_poi_accessibility()
            poi_analysis.to_excel(writer, sheet_name='POI_Analysis', index=False)

            # Priority zones
            priority_zones = analyzer.get_priority_zones()
            priority_zones.to_excel(writer, sheet_name='Priority_Zones', index=False)

            # Geographic analysis
            if 'Municipio' in analyzer.df.columns:
                muni_analysis = analyzer._calculate_geographic_analysis('Municipio', 'Municipality')
                muni_analysis.to_excel(writer, sheet_name='Municipality_Analysis', index=False)

            if 'Comarca' in analyzer.df.columns:
                comarca_analysis = analyzer._calculate_geographic_analysis('Comarca', 'Comarca')
                comarca_analysis.to_excel(writer, sheet_name='Comarca_Analysis', index=False)

            # Transport comparison if available
            if analyzer.has_private_transport:
                transport_comparison = analyzer.get_transport_comparison()
                transport_comparison.to_excel(writer, sheet_name='Transport_Comparison', index=False)

                if 'Municipio' in analyzer.df.columns:
                    muni_diff = analyzer.get_geographic_differences('Municipio')
                    muni_diff.to_excel(writer, sheet_name='Municipality_Transport_Diff', index=False)

                if 'Comarca' in analyzer.df.columns:
                    comarca_diff = analyzer.get_geographic_differences('Comarca')
                    comarca_diff.to_excel(writer, sheet_name='Comarca_Transport_Diff', index=False)

            # Summary statistics
            summary_stats = pd.DataFrame({
                'Metric': [
                    'Total Zones',
                    'Total Population',
                    'Average Travel Time (min)',
                    'Population with Excellent Access (<30min)',
                    'Population with Good Access (30-45min)',
                    'Population with Poor Access (>75min)'
                ],
                'Value': [
                    len(analyzer.zone_populations),
                    analyzer.total_population,
                    analyzer.df.groupby('Zona_Origen')['Tiempo_Total_Minutos'].min().mean(),
                    zone_best[zone_best['Tiempo_Total_Minutos'] <= 30]['Poblacion'].sum(),
                    zone_best[(zone_best['Tiempo_Total_Minutos'] > 30) &
                              (zone_best['Tiempo_Total_Minutos'] <= 45)]['Poblacion'].sum(),
                    zone_best[zone_best['Tiempo_Total_Minutos'] > 75]['Poblacion'].sum()
                ]
            })
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)

        output.seek(0)
        return output


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

class StreamlitApp:
    """Main Streamlit application class"""

    def __init__(self):
        self.styler = ChartStyler()
        self.exporter = ExportManager()

    def render_header(self):
        """Render application header"""
        st.set_page_config(
            page_title="Public Transport Accessibility Analysis",
            page_icon="🚌",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚌 Public Transport Accessibility Analysis")
        st.markdown("""
        **Upload your accessibility analysis data to generate comprehensive insights about public transport coverage and performance.**

        This tool analyzes:
        - **Population accessibility** across different travel time thresholds
        - **POI-specific performance** for different destinations
        - **Geographic patterns** by municipality and comarca
        - **Trip generation analysis** for service planning
        - **Transport mode comparison** (when private transport data is available)
        """)

    def process_uploaded_file(self, uploaded_file) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[AccessibilityAnalyzer]]:
        """Process uploaded Excel file"""
        try:
            # Read data
            df_all = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success(f"✅ File loaded successfully! {len(df_all):,} records found.")

            # Check required columns
            required_columns = ['Zona_Origen', 'Nombre_Destino', 'Tiempo_Viaje', 'Poblacion']
            missing_columns = [col for col in required_columns if col not in df_all.columns]

            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                return None, None, None

            # Show column info
            with st.expander("📋 Data Overview"):
                st.write("**Columns found:**")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Core columns:**")
                    core_cols = ['Zona_Origen', 'Nombre_Destino', 'Tiempo_Viaje', 'Poblacion']
                    for col in core_cols:
                        status = "✅" if col in df_all.columns else "❌"
                        st.write(f"{status} {col}")

                with col2:
                    st.write("**Optional columns:**")
                    optional_cols = ['Municipio', 'Comarca', 'TT_Coche_NoPico', 'Distancia_km']
                    for col in optional_cols:
                        status = "✅" if col in df_all.columns else "❌"
                        st.write(f"{status} {col}")

                st.dataframe(df_all.head(), use_container_width=True)

            # Filter valid data
            df = df_all[df_all['Tiempo_Viaje'] <= 999].copy()

            if len(df) < len(df_all):
                removed_count = len(df_all) - len(df)
                st.warning(f"⚠️ Removed {removed_count:,} records with travel times > 999 minutes")

            # Initialize analyzer
            analyzer = AccessibilityAnalyzer(df)

            return df, df_all, analyzer

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return None, None, None

    def render_key_metrics(self, analyzer: AccessibilityAnalyzer):
        """Render key performance metrics"""
        st.markdown("## 📈 Key Performance Metrics")

        # Calculate metrics
        zone_best = analyzer.get_zone_best_access()
        pop_by_category, category_pcts = analyzer.get_population_by_category()

        # Define metrics
        metrics = [
            ("🏘️ Total Zones", f"{len(analyzer.zone_populations):,}"),
            ("👥 Total Population", f"{analyzer.total_population:,}"),
            ("⏱️ Average Best Travel Time", f"{zone_best['Tiempo_Total_Minutos'].mean():.1f} min"),
            ("🟢 Excellent Access", f"{category_pcts.get('🟢 Excellent (<30min)', 0):.1f}%"),
            ("🔴 Poor Access",
             f"{(category_pcts.get('🔴 Poor (75-90min)', 0) + category_pcts.get('⚫ Very Poor (>90min)', 0)):.1f}%")
        ]

        # Display metrics
        cols = st.columns(len(metrics))
        for i, (label, value) in enumerate(metrics):
            with cols[i]:
                st.metric(label, value)

        # Transport comparison metrics if available
        if analyzer.has_private_transport:
            st.markdown("### 🚗 Transport Mode Comparison")
            transport_comparison = analyzer.get_transport_comparison()
            if not transport_comparison.empty:
                avg_public = transport_comparison['Public Transport (min)'].mean()
                avg_private = transport_comparison['Private Transport (min)'].mean()
                avg_difference = transport_comparison['Time Difference (min)'].mean()

                transport_metrics = [
                    ("🚌 Avg Public Transport", f"{avg_public:.1f} min"),
                    ("🚗 Avg Private Transport", f"{avg_private:.1f} min"),
                    ("📊 Average Difference", f"{avg_difference:.1f} min"),
                    ("📈 Public vs Private Ratio", f"{avg_public / avg_private:.1f}x")
                ]

                transport_cols = st.columns(len(transport_metrics))
                for i, (label, value) in enumerate(transport_metrics):
                    with transport_cols[i]:
                        st.metric(label, value)

    def render_accessibility_overview(self, analyzer: AccessibilityAnalyzer):
        """Render accessibility overview section"""
        st.markdown("## 🎯 Population Accessibility Overview")

        # Get data
        zone_best = analyzer.get_zone_best_access()
        pop_by_category, category_pcts = analyzer.get_population_by_category()

        # Population distribution pie chart
        if not category_pcts.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = px.pie(
                    values=pop_by_category.values,
                    names=pop_by_category.index,
                    title="Population by Accessibility Level",
                    color_discrete_map={
                        '🟢 Excellent (<30min)': CONFIG.COLORS['excellent'],
                        '🟡 Good (30-45min)': CONFIG.COLORS['good'],
                        '🟠 Fair (45-60min)': CONFIG.COLORS['fair'],
                        '🔶 Moderate (60-75min)': CONFIG.COLORS['moderate'],
                        '🔴 Poor (75-90min)': CONFIG.COLORS['poor'],
                        '⚫ Very Poor (>90min)': CONFIG.COLORS['very_poor']
                    }
                )

                fig_pie.update_layout(
                    title_font_size=20,
                    legend=dict(font=dict(size=14)),
                    height=500
                )

                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=12
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown("### 📊 Accessibility Breakdown")

                breakdown_data = []
                for category, population in pop_by_category.items():
                    percentage = category_pcts.get(category, 0)
                    breakdown_data.append({
                        'Category': category,
                        'Population': population,
                        'Percentage': percentage
                    })

                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(
                    breakdown_df.style.format({
                        'Population': '{:,.0f}',
                        'Percentage': '{:.1f}%'
                    }),
                    hide_index=True,
                    use_container_width=True
                )

        # Travel time distribution histogram
        st.markdown("### 📈 Travel Time Distribution")

        # Create weighted histogram data
        hist_data = []
        for _, row in zone_best.iterrows():
            hist_data.extend([row['Tiempo_Total_Minutos']] * int(row['Poblacion']))

        fig_hist = px.histogram(
            x=hist_data,
            nbins=30,
            title='Population Distribution by Best Travel Time',
            labels={'x': 'Travel Time (minutes)', 'y': 'Population'}
        )

        # Add threshold lines
        thresholds = [30, 45, 60, 75, 90]
        threshold_names = ['Excellent', 'Good', 'Fair', 'Moderate', 'Poor']

        for threshold, name in zip(thresholds, threshold_names):
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{name} ({threshold}min)",
                annotation_position="top"
            )

        fig_hist = self.styler.apply_standard_styling(
            fig_hist,
            'Population Distribution by Best Travel Time',
            'Travel Time (minutes)',
            'Population'
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    def render_travel_time_distributions(self, analyzer: AccessibilityAnalyzer):
        """Render travel time distribution analysis"""
        st.markdown("## ⏱️ Travel Time Distribution Analysis")

        # Box plot by POI
        st.markdown("### 📦 Travel Time Variations by POI")

        fig_box = px.box(
            analyzer.df,
            x='Nombre_Destino',
            y='Tiempo_Total_Minutos',
            title='Travel Time Distribution by POI'
        )

        fig_box.update_xaxes(tickangle=45)
        fig_box = self.styler.apply_standard_styling(
            fig_box,
            'Travel Time Distribution by Point of Interest',
            'Point of Interest',
            'Travel Time (minutes)',
            height=600
        )

        st.plotly_chart(fig_box, use_container_width=True)

        # Summary statistics table
        st.markdown("### 📋 Travel Time Summary Statistics by POI")

        poi_stats = analyzer.df.groupby('Nombre_Destino')['Tiempo_Total_Minutos'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        poi_stats.columns = ['Routes', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        poi_stats = poi_stats.reset_index()
        poi_stats.columns = ['POI', 'Routes', 'Mean (min)', 'Median (min)', 'Std Dev (min)', 'Min (min)', 'Max (min)']

        st.dataframe(poi_stats, use_container_width=True, hide_index=True)

    def render_poi_analysis(self, analyzer: AccessibilityAnalyzer):
        """Render POI-specific analysis"""
        st.markdown("## 🎯 Point of Interest Analysis")

        poi_analysis = analyzer.get_poi_accessibility()

        if not poi_analysis.empty:
            # POI performance ranking
            st.markdown("### 🏆 POI Performance Ranking")

            fig_poi = go.Figure()

            categories = ['Excellent (%)', 'Good (%)', 'Fair (%)', 'Moderate (%)', 'Poor (%)', 'Very Poor (%)']
            colors = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                      CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]
            labels = ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔶 Moderate', '🔴 Poor', '⚫ Very Poor']

            for cat, color, label in zip(categories, colors, labels):
                fig_poi.add_trace(go.Bar(
                    name=label,
                    y=poi_analysis['POI'],
                    x=poi_analysis[cat],
                    orientation='h',
                    marker_color=color,
                    text=poi_analysis[cat].round(1),
                    textposition='inside',
                    texttemplate='%{text:.0f}%'
                ))

            fig_poi = self.styler.apply_standard_styling(
                fig_poi,
                'POI Accessibility Performance',
                '% of Population',
                'Point of Interest',
                height=max(600, len(poi_analysis) * 40)
            )
            fig_poi.update_layout(
                barmode='stack',
                yaxis=dict(categoryorder='total ascending'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig_poi, use_container_width=True)

            # Detailed POI table
            st.markdown("### 📊 Detailed POI Analysis")
            display_columns = ['POI', 'Avg Time (weighted)', 'Total Population',
                               'Excellent (%)', 'Good (%)', 'Fair (%)', 'Moderate (%)', 'Poor (%)', 'Very Poor (%)']

            format_dict = {
                'Avg Time (weighted)': '{:.1f} min',
                'Total Population': '{:,.0f}',
                'Excellent (%)': '{:.1f}%',
                'Good (%)': '{:.1f}%',
                'Fair (%)': '{:.1f}%',
                'Moderate (%)': '{:.1f}%',
                'Poor (%)': '{:.1f}%',
                'Very Poor (%)': '{:.1f}%'
            }

            st.dataframe(
                poi_analysis[display_columns].style.format(format_dict),
                hide_index=True,
                use_container_width=True,
                height=400
            )

    def render_trip_need_analysis(self, analyzer: AccessibilityAnalyzer):
        """Render trip need and priority analysis"""
        st.markdown("## 🔄 Trip Need & Priority Zone Analysis")

        # Priority zones analysis
        priority_zones = analyzer.get_priority_zones(threshold=45)

        if not priority_zones.empty:
            st.markdown("### ⚠️ Priority Zones (>45min average travel time)")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Priority zones chart
                fig_priority = px.scatter(
                    priority_zones.head(20),
                    x='Avg Travel Time',
                    y='Population',
                    size='Population',
                    hover_data=['Zone Name', 'POIs Served', 'Avg Transfers'],
                    title='Top 20 Priority Zones by Population',
                    labels={'Avg Travel Time': 'Average Travel Time (minutes)', 'Population': 'Population'}
                )

                fig_priority = self.styler.apply_standard_styling(
                    fig_priority,
                    'Priority Zones - High Travel Time & Population',
                    'Average Travel Time (minutes)',
                    'Population'
                )

                st.plotly_chart(fig_priority, use_container_width=True)

            with col2:
                st.markdown("#### 📈 Priority Zone Summary")
                st.metric("Zones Needing Attention", len(priority_zones))
                st.metric("Population Affected", f"{priority_zones['Population'].sum():,}")
                st.metric("Average Travel Time", f"{priority_zones['Avg Travel Time'].mean():.1f} min")

            # Priority zones table
            st.markdown("#### 📋 Detailed Priority Zones")
            priority_display = priority_zones[
                ['Zone Name', 'Avg Travel Time', 'Population', 'POIs Served', 'Avg Transfers']].head(15)

            st.dataframe(
                priority_display.style.format({
                    'Avg Travel Time': '{:.1f} min',
                    'Population': '{:,.0f}',
                    'POIs Served': '{:.0f}',
                    'Avg Transfers': '{:.2f}'
                }),
                hide_index=True,
                use_container_width=True,
                height=400
            )

        # Service frequency analysis
        st.markdown("### 🚌 Service Coverage Analysis")

        coverage_analysis = analyzer.df.groupby('Zona_Origen').agg({
            'Nombre_Destino': 'nunique',
            'Poblacion': 'first',
            'Tiempo_Total_Minutos': 'mean'
        }).reset_index()

        coverage_analysis.columns = ['Zone_ID', 'POIs_Accessible', 'Population', 'Avg_Travel_Time']

        # Coverage distribution
        hist_coverage_data = []
        for _, row in coverage_analysis.iterrows():
            hist_coverage_data.extend([row['POIs_Accessible']] * int(row['Population']))

        fig_coverage = px.histogram(
            x=hist_coverage_data,
            title='Population Distribution by Number of Accessible POIs',
            labels={'x': 'Number of Accessible POIs', 'y': 'Population'}
        )

        fig_coverage = self.styler.apply_standard_styling(
            fig_coverage,
            'Population Distribution by POI Accessibility',
            'Number of Accessible POIs',
            'Population'
        )

        st.plotly_chart(fig_coverage, use_container_width=True)

    def render_transport_comparison(self, analyzer: AccessibilityAnalyzer):
        """Render transport mode comparison analysis"""
        if not analyzer.has_private_transport:
            st.warning(
                "⚠️ Private transport data not available. Please ensure your data includes 'TT_Coche_NoPico' and 'Distancia_km' columns.")
            return

        st.markdown("## 🚗🚌 Public vs Private Transport Comparison")

        # Get comparison data
        transport_comparison = analyzer.get_transport_comparison()

        if transport_comparison.empty:
            st.error("No transport comparison data available.")
            return

        # Overall comparison metrics
        col1, col2, col3 = st.columns(3)

        avg_public = transport_comparison['Public Transport (min)'].mean()
        avg_private = transport_comparison['Private Transport (min)'].mean()
        avg_difference = transport_comparison['Time Difference (min)'].mean()

        with col1:
            st.metric("Average Public Transport", f"{avg_public:.1f} min")
        with col2:
            st.metric("Average Private Transport", f"{avg_private:.1f} min")
        with col3:
            delta_color = "normal" if avg_difference > 0 else "inverse"
            st.metric("Average Time Difference", f"{avg_difference:.1f} min",
                      delta=f"{avg_difference:.1f} min slower" if avg_difference > 0 else f"{abs(avg_difference):.1f} min faster",
                      delta_color=delta_color)

        # Transport comparison by POI
        st.markdown("### 🎯 Transport Comparison by POI")

        fig_transport = go.Figure()

        # Add public transport bars
        fig_transport.add_trace(go.Bar(
            name='🚌 Public Transport',
            x=transport_comparison['POI'],
            y=transport_comparison['Public Transport (min)'],
            marker_color=CONFIG.COLORS['public_transport'],
            text=transport_comparison['Public Transport (min)'].round(1),
            textposition='outside'
        ))

        # Add private transport bars
        fig_transport.add_trace(go.Bar(
            name='🚗 Private Transport',
            x=transport_comparison['POI'],
            y=transport_comparison['Private Transport (min)'],
            marker_color=CONFIG.COLORS['private_transport'],
            text=transport_comparison['Private Transport (min)'].round(1),
            textposition='outside'
        ))

        fig_transport.update_layout(
            title='Travel Time Comparison by POI',
            xaxis_title='Point of Interest',
            yaxis_title='Travel Time (minutes)',
            barmode='group',
            height=600
        )

        fig_transport.update_xaxes(tickangle=45)
        st.plotly_chart(fig_transport, use_container_width=True)

        # Time difference analysis
        st.markdown("### 📊 Time Difference Analysis")

        fig_diff = px.bar(
            transport_comparison.sort_values('Time Difference (min)', ascending=True),
            x='Time Difference (min)',
            y='POI',
            orientation='h',
            title='Public Transport Time Penalty/Advantage by POI',
            color='Time Difference (min)',
            color_continuous_scale=['green', 'yellow', 'red'],
            text='Time Difference (min)'
        )

        fig_diff.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
        fig_diff.update_layout(
            height=max(400, len(transport_comparison) * 30),
            xaxis_title='Time Difference (minutes - positive means public transport is slower)',
            yaxis_title='Point of Interest'
        )

        st.plotly_chart(fig_diff, use_container_width=True)

        # Distance vs time analysis
        st.markdown("### 🛣️ Distance vs Travel Time Analysis")

        fig_scatter = px.scatter(
            transport_comparison,
            x='Distance (km)',
            y='Public Transport (min)',
            size='Total Population',
            hover_data=['POI', 'Private Transport (min)', 'Time Difference (min)'],
            title='Public Transport Time vs Distance',
            color='Time Difference (min)',
            color_continuous_scale=['green', 'yellow', 'red']
        )

        fig_scatter.update_layout(
            height=600,
            xaxis_title='Distance (km)',
            yaxis_title='Public Transport Time (minutes)'
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Detailed comparison table
        st.markdown("### 📋 Detailed Transport Comparison")

        format_dict = {
            'Public Transport (min)': '{:.1f}',
            'Private Transport (min)': '{:.1f}',
            'Distance (km)': '{:.1f}',
            'Time Difference (min)': '{:.1f}',
            'Total Population': '{:,.0f}'
        }

        st.dataframe(
            transport_comparison.style.format(format_dict),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    def render_geographic_analysis(self, analyzer: AccessibilityAnalyzer):
        """Render geographic analysis by municipality and comarca"""
        st.markdown("## 🗺️ Geographic Analysis by Municipality & Comarca")

        # Municipality analysis
        if 'Municipio' in analyzer.df.columns:
            st.markdown("### Analysis by Municipality")
            muni_analysis = self._calculate_geographic_analysis(analyzer, 'Municipio', 'Municipality')
            self._render_geographic_chart(muni_analysis, 'Municipality', 10)
            self._render_geographic_table(muni_analysis)

            # Municipality transport differences if available
            if analyzer.has_private_transport:
                st.markdown("### 🚗🚌 Municipality Transport Mode Differences")
                muni_diff = analyzer.get_geographic_differences('Municipio')
                if not muni_diff.empty:
                    self._render_transport_difference_chart(muni_diff, 'Municipality')
                    self._render_transport_difference_table(muni_diff)

        # Comarca analysis
        if 'Comarca' in analyzer.df.columns:
            st.markdown("### Analysis by Comarca")
            comarca_analysis = self._calculate_geographic_analysis(analyzer, 'Comarca', 'Comarca')
            self._render_geographic_chart(comarca_analysis, 'Comarca', 10)
            self._render_geographic_table(comarca_analysis)

            # Comarca transport differences if available
            if analyzer.has_private_transport:
                st.markdown("### 🚗🚌 Comarca Transport Mode Differences")
                comarca_diff = analyzer.get_geographic_differences('Comarca')
                if not comarca_diff.empty:
                    self._render_transport_difference_chart(comarca_diff, 'Comarca')
                    self._render_transport_difference_table(comarca_diff)

    def _calculate_geographic_analysis(self, analyzer: AccessibilityAnalyzer,
                                       geo_column: str, unit_type: str) -> pd.DataFrame:
        """Calculate geographic analysis for a given geographic unit"""
        # Get zone-level best access
        zone_best = analyzer.get_zone_best_access()

        # Merge with geographic data
        zone_geo = analyzer.df[['Zona_Origen', geo_column, 'Num_Transbordos']].drop_duplicates()
        analysis_data = zone_best.merge(zone_geo, on='Zona_Origen')

        # Group by geographic unit
        geo_analysis = analysis_data.groupby(geo_column).agg({
            'Poblacion': 'sum',
            'Tiempo_Total_Minutos': lambda x: np.average(x, weights=analysis_data.loc[x.index, 'Poblacion']),
            'Num_Transbordos': lambda x: np.average(x, weights=analysis_data.loc[x.index, 'Poblacion']),
            'Zona_Origen': 'count'
        }).reset_index()

        geo_analysis.columns = ['Geographic Unit', 'Total Population', 'Avg Travel Time', 'Avg Transfers', 'Zones']

        # Calculate accessibility percentages
        for category, threshold_info in [
            ('Excellent', (0, CONFIG.EXCELLENT_THRESHOLD)),
            ('Good', (CONFIG.EXCELLENT_THRESHOLD, CONFIG.GOOD_THRESHOLD)),
            ('Fair', (CONFIG.GOOD_THRESHOLD, CONFIG.FAIR_THRESHOLD)),
            ('Moderate', (CONFIG.FAIR_THRESHOLD, CONFIG.MODERATE_THRESHOLD)),
            ('Poor', (CONFIG.MODERATE_THRESHOLD, CONFIG.POOR_THRESHOLD)),
            ('Very Poor', (CONFIG.POOR_THRESHOLD, float('inf')))
        ]:
            min_time, max_time = threshold_info
            category_pop = []

            for geo_unit in geo_analysis['Geographic Unit']:
                unit_data = analysis_data[analysis_data[geo_column] == geo_unit]
                if max_time == float('inf'):
                    cat_zones = unit_data[unit_data['Tiempo_Total_Minutos'] > min_time]
                else:
                    cat_zones = unit_data[
                        (unit_data['Tiempo_Total_Minutos'] > min_time) &
                        (unit_data['Tiempo_Total_Minutos'] <= max_time)
                        ]

                cat_pop = cat_zones['Poblacion'].sum()
                total_pop = unit_data['Poblacion'].sum()
                category_pop.append((cat_pop / total_pop * 100) if total_pop > 0 else 0)

            geo_analysis[f'{category} (%)'] = category_pop

        return geo_analysis.sort_values('Total Population', ascending=False)

    def _render_geographic_chart(self, analysis: pd.DataFrame, unit_type: str, top_n: int):
        """Render geographic analysis chart"""
        unit_plural = f"{unit_type}s" if unit_type != "Comarca" else "Comarcas"

        chart_data = analysis.head(top_n) if top_n < len(analysis) else analysis

        fig_access = go.Figure()

        categories = ['Excellent (%)', 'Good (%)', 'Fair (%)', 'Moderate (%)', 'Poor (%)', 'Very Poor (%)']
        colors = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                  CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]
        labels = ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔶 Moderate', '🔴 Poor', '⚫ Very Poor']

        for cat, color, label in zip(categories, colors, labels):
            fig_access.add_trace(go.Bar(
                name=label,
                y=chart_data['Geographic Unit'],
                x=chart_data[cat],
                orientation='h',
                marker_color=color,
                text=chart_data[cat].round(1),
                textposition='inside',
                texttemplate='%{text:.0f}%'
            ))

        title = f'Top {top_n} {unit_plural} by Population - Accessibility Distribution' if top_n < len(
            analysis) else f'{unit_plural} by Population - Accessibility Distribution'

        fig_access = self.styler.apply_standard_styling(
            fig_access,
            title,
            '% of Population',
            unit_type,
            height=max(600, len(chart_data) * 30)
        )
        fig_access.update_layout(
            barmode='stack',
            yaxis=dict(categoryorder='total ascending'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_access, use_container_width=True)

    def _render_geographic_table(self, analysis: pd.DataFrame):
        """Render geographic analysis table"""
        columns = ['Geographic Unit', 'Total Population', 'Avg Travel Time', 'Avg Transfers', 'Zones',
                   'Excellent (%)', 'Good (%)', 'Fair (%)', 'Moderate (%)', 'Poor (%)', 'Very Poor (%)']
        display_df = analysis[columns]

        format_dict = {
            'Total Population': '{:,.0f}',
            'Avg Travel Time': '{:.1f} min',
            'Avg Transfers': '{:.2f}',
            'Zones': '{:.0f}',
            'Excellent (%)': '{:.1f}%',
            'Good (%)': '{:.1f}%',
            'Fair (%)': '{:.1f}%',
            'Moderate (%)': '{:.1f}%',
            'Poor (%)': '{:.1f}%',
            'Very Poor (%)': '{:.1f}%'
        }

        st.dataframe(
            display_df.style.format(format_dict),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    def _render_transport_difference_chart(self, analysis: pd.DataFrame, unit_type: str):
        """Render transport mode difference chart"""
        unit_plural = f"{unit_type}s" if unit_type != "Comarca" else "Comarcas"

        fig_diff = go.Figure()

        # Add bars for time difference
        fig_diff.add_trace(go.Bar(
            name='Time Difference (Public - Private)',
            y=analysis['Geographic Unit'],
            x=analysis['Time Difference (min)'],
            orientation='h',
            marker_color=CONFIG.COLORS['difference'],
            text=analysis['Time Difference (min)'].round(1),
            textposition='outside',
            texttemplate='%{text:.1f} min'
        ))

        title = f'{unit_plural} - Public vs Private Transport Time Difference'

        fig_diff = self.styler.apply_standard_styling(
            fig_diff,
            title,
            'Time Difference (minutes)',
            unit_type,
            height=max(600, len(analysis) * 30)
        )

        # Add vertical line at 0
        fig_diff.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Equal Travel Time")

        fig_diff.update_layout(yaxis=dict(categoryorder='total ascending'))
        st.plotly_chart(fig_diff, use_container_width=True)

    def _render_transport_difference_table(self, analysis: pd.DataFrame):
        """Render transport mode difference table"""
        columns = ['Geographic Unit', 'Public Transport (min)', 'Private Transport (min)',
                   'Distance (km)', 'Time Difference (min)', 'Total Population', 'Zones']
        display_df = analysis[columns]

        format_dict = {
            'Public Transport (min)': '{:.1f}',
            'Private Transport (min)': '{:.1f}',
            'Distance (km)': '{:.1f}',
            'Time Difference (min)': '{:.1f}',
            'Total Population': '{:,.0f}',
            'Zones': '{:.0f}'
        }

        st.dataframe(
            display_df.style.format(format_dict),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    def run(self):
        """Main application runner"""
        self.render_header()

        # File uploader
        uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])

        if uploaded_file is not None:
            # Process data
            df, df_all, analyzer = self.process_uploaded_file(uploaded_file)

            if analyzer is not None:
                # Render key metrics
                self.render_key_metrics(analyzer)

                # Create tabs for different sections
                tab_list = [
                    "📊 Population Accessibility",
                    "⏱️ Travel Time Distributions",
                    "🎯 POI Breakdown",
                    "🔄 Trip Need Analysis",
                    "🗺️ Geographic Analysis"
                ]

                # Add transport comparison tab if data is available
                if analyzer.has_private_transport:
                    tab_list.insert(4, "🚗🚌 Transport Comparison")

                tabs = st.tabs(tab_list)

                with tabs[0]:
                    self.render_accessibility_overview(analyzer)

                with tabs[1]:
                    self.render_travel_time_distributions(analyzer)

                with tabs[2]:
                    self.render_poi_analysis(analyzer)

                with tabs[3]:
                    self.render_trip_need_analysis(analyzer)

                # Handle transport comparison tab if available
                if analyzer.has_private_transport:
                    with tabs[4]:
                        self.render_transport_comparison(analyzer)
                    with tabs[5]:
                        self.render_geographic_analysis(analyzer)
                else:
                    with tabs[4]:
                        self.render_geographic_analysis(analyzer)

                # Export functionality
                st.markdown("## 📥 Export Data")

                if st.button("📊 Generate Comprehensive Excel Report"):
                    excel_data = self.exporter.generate_detailed_report(analyzer)

                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_data,
                        file_name=f"accessibility_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
    