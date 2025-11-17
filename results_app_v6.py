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
        'very_poor': '#D73027'
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
        self.df['Tiempo_Total_Minutos'] = self.df['Tiempo_Trayecto']  # Using Tiempo_Trayecto as total time

        # Filter out zones with travel times > 999 minutes (unrealistic/invalid data)
        self.df = self.df[self.df['Tiempo_Total_Minutos'] <= 999]

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
            font=dict(size=16),
            annotations=[dict(font=dict(size=16))] if fig.layout.annotations else []
        )

        fig.update_traces(
            textfont=dict(size=16),
            hoverlabel=dict(font=dict(size=16))
        )

        return fig


class MetricsCalculator:
    """Calculate key accessibility metrics"""

    def __init__(self, analyzer: AccessibilityAnalyzer):
        self.analyzer = analyzer

    def calculate_weighted_avg_time(self) -> float:
        """Calculate population-weighted average travel time"""
        zone_best = self.analyzer.get_zone_best_access()
        if self.analyzer.total_population == 0:
            return 0
        return (zone_best['Tiempo_Total_Minutos'] * zone_best['Poblacion']).sum() / self.analyzer.total_population

    def calculate_access_percentages(self) -> Dict[str, float]:
        """Calculate percentage of population in each accessibility category"""
        _, percentages = self.analyzer.get_population_by_category()
        return percentages.to_dict() if not percentages.empty else {}

    def get_top_metrics(self) -> Dict:
        """Get comprehensive top-level metrics"""
        return {
            'total_population': self.analyzer.total_population,
            'avg_travel_time': self.calculate_weighted_avg_time(),
            'access_percentages': self.calculate_access_percentages(),
            'total_zones': len(self.analyzer.zone_populations)
        }


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class StreamlitApp:
    """Main Streamlit application"""

    def __init__(self):
        self.styler = ChartStyler()
        st.set_page_config(page_title="Accessibility Analysis - Simplified", layout="wide")

    def render_header(self):
        """Render application header"""
        st.title("🚌 Transportation Accessibility Analysis - Simplified Version")
        st.info(
            "ℹ️ Note: Zones with travel times > 999 minutes are excluded from analysis as unrealistic/invalid data.")
        st.markdown("---")

    def process_uploaded_file(self, uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame, AccessibilityAnalyzer]:
        """Process the uploaded file and return necessary dataframes"""
        try:
            df = pd.read_excel(uploaded_file)

            # Verify required columns exist
            required_columns = [
                'Zona_Origen', 'Nombre_Origen', 'Poblacion', 'Municipio', 'Comarca',
                'Destino_Óptimo', 'Nombre_Destino', 'Necesita_Viaje', 'Tiempo_Trayecto'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return None, None, None

            # Show data filtering info
            original_count = len(df)
            filtered_df = df[df['Tiempo_Trayecto'] <= 999]
            filtered_count = len(filtered_df)
            excluded_count = original_count - filtered_count

            if excluded_count > 0:
                st.warning(
                    f"⚠️ Excluded {excluded_count:,} records with travel times > 999 minutes from {original_count:,} total records.")
            else:
                st.success(f"✅ Processing {original_count:,} records (no filtering needed).")

            # Create analyzer (which will apply the same filter internally)
            analyzer = AccessibilityAnalyzer(df)

            return df, df, analyzer

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None, None, None

    def render_key_metrics(self, analyzer: AccessibilityAnalyzer):
        """Render key metrics section"""
        st.markdown("## 🎯 Key Metrics")

        metrics_calc = MetricsCalculator(analyzer)
        metrics = metrics_calc.get_top_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Population",
                value=f"{metrics['total_population']:,.0f}"
            )

        with col2:
            st.metric(
                label="Avg Travel Time",
                value=f"{metrics['avg_travel_time']:.1f} min"
            )

        with col3:
            excellent_pct = metrics['access_percentages'].get('🟢 Excellent (<30min)', 0)
            st.metric(
                label="Excellent Access",
                value=f"{excellent_pct:.1f}%"
            )

        with col4:
            st.metric(
                label="Total Zones",
                value=f"{metrics['total_zones']:,}"
            )

    def render_accessibility_overview(self, analyzer: AccessibilityAnalyzer):
        """Render accessibility overview section"""
        st.markdown("## 📊 Population Accessibility Distribution")

        pop_by_category, category_pcts = analyzer.get_population_by_category()

        if not pop_by_category.empty:
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=pop_by_category.index,
                values=pop_by_category.values,
                hole=.3,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(colors=[
                                       CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                                       CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']
                                   ][:len(pop_by_category)])
            )])

            fig = self.styler.apply_standard_styling(
                fig,
                "Population Distribution by Accessibility Level",
                "", ""
            )

            st.plotly_chart(fig, use_container_width=True)

            # Create bar chart
            fig_bar = go.Figure(data=[go.Bar(
                x=pop_by_category.index,
                y=pop_by_category.values,
                marker_color=[
                                 CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                                 CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']
                             ][:len(pop_by_category)],
                text=[f"{val:,.0f}" for val in pop_by_category.values],
                textposition='outside'
            )])

            fig_bar = self.styler.apply_standard_styling(
                fig_bar,
                "Population by Accessibility Category",
                "Accessibility Category",
                "Population"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

    def render_travel_time_distributions(self, analyzer: AccessibilityAnalyzer):
        """Render travel time distribution analysis"""
        st.markdown("## ⏱️ Travel Time Distribution Analysis")

        st.markdown("### Population-Weighted Distribution")
        fig_weighted = self._create_travel_time_distribution(analyzer)
        st.plotly_chart(fig_weighted, use_container_width=True)

        st.markdown("---")

        st.markdown("### Distribution by Zones")
        fig_zones = self._create_travel_time_by_zones_chart(analyzer)
        st.plotly_chart(fig_zones, use_container_width=True)

    def _create_travel_time_distribution(self, analyzer: AccessibilityAnalyzer) -> go.Figure:
        """Create travel time distribution with KDE overlay"""
        df = analyzer.df
        zone_populations = analyzer.zone_populations
        total_population = analyzer.total_population

        # Create bins
        bins = np.linspace(df['Tiempo_Total_Minutos'].min(),
                           df['Tiempo_Total_Minutos'].max(), 41)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate population-weighted histogram
        bin_populations = []
        for i in range(len(bins) - 1):
            mask = ((df['Tiempo_Total_Minutos'] >= bins[i]) &
                    (df['Tiempo_Total_Minutos'] < bins[i + 1]))
            zones_in_bin = df[mask]['Zona_Origen'].unique()
            pop_in_bin = zone_populations[zone_populations.index.isin(zones_in_bin)].sum()
            bin_populations.append(pop_in_bin / total_population * 100)

        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=bin_populations,
            marker_color='#EBEBEB',
            width=(bins[1] - bins[0]) * 0.9,
            showlegend=False
        ))

        # Add KDE curve
        self._add_kde_curve(fig, df, bin_populations)

        # Add reference lines
        self._add_reference_lines(fig, analyzer)

        return self.styler.apply_standard_styling(
            fig, "Travel Time Distribution (Population-Weighted)",
            "Total Travel Time (minutes)", "Percentage of Population (%)"
        )

    def _create_travel_time_by_zones_chart(self, analyzer: AccessibilityAnalyzer) -> go.Figure:
        """Create travel time distribution by zones chart"""
        zone_best = analyzer.get_zone_best_access()

        fig = px.histogram(
            zone_best,
            x='Tiempo_Total_Minutos',
            nbins=40,
            histnorm='percent',
            color_discrete_sequence=['#EBEBEB']
        )

        # Add KDE curve for zones
        travel_times_zones = zone_best['Tiempo_Total_Minutos'].dropna()
        if len(travel_times_zones) > 0:
            kde = stats.gaussian_kde(travel_times_zones)
            x_range = np.linspace(travel_times_zones.min(), travel_times_zones.max(), 200)
            y_kde = kde(x_range)
            y_kde_normalized = y_kde * 100 * (travel_times_zones.max() - travel_times_zones.min()) / 40

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_normalized,
                mode='lines',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

            # Add reference lines with zone percentages
            total_zones = len(zone_best)
            zones_under_30_pct = (zone_best['Tiempo_Total_Minutos'] <= 30).sum() / total_zones * 100
            zones_under_45_pct = (zone_best['Tiempo_Total_Minutos'] <= 45).sum() / total_zones * 100
            zones_under_60_pct = (zone_best['Tiempo_Total_Minutos'] <= 60).sum() / total_zones * 100
            zones_under_75_pct = (zone_best['Tiempo_Total_Minutos'] <= 75).sum() / total_zones * 100
            zones_under_90_pct = (zone_best['Tiempo_Total_Minutos'] <= 90).sum() / total_zones * 100

            fig.add_vline(x=30, line_dash="dash", line_color="green",
                          annotation_text=f"30 min<br>({zones_under_30_pct:.1f}% zones)",
                          annotation_position="top")
            fig.add_vline(x=45, line_dash="dash", line_color="orange",
                          annotation_text=f"45 min<br>({zones_under_45_pct:.1f}% zones)",
                          annotation_position="top")
            fig.add_vline(x=60, line_dash="dash", line_color="red",
                          annotation_text=f"60 min<br>({zones_under_60_pct:.1f}% zones)",
                          annotation_position="top")
            fig.add_vline(x=75, line_dash="dash", line_color="darkred",
                          annotation_text=f"75 min<br>({zones_under_75_pct:.1f}% zones)",
                          annotation_position="top")
            fig.add_vline(x=90, line_dash="dash", line_color="maroon",
                          annotation_text=f"90 min<br>({zones_under_90_pct:.1f}% zones)",
                          annotation_position="top")

        return self.styler.apply_standard_styling(
            fig, "Travel Time Distribution by Zones",
            "Total Travel Time (minutes)", "Percentage of Zones (%)"
        )

    def _add_kde_curve(self, fig: go.Figure, df: pd.DataFrame, bin_populations: List[float]):
        """Add KDE curve to histogram"""
        travel_times_list = []
        for _, row in df.iterrows():
            travel_times_list.extend([row['Tiempo_Total_Minutos']] * int(row['Poblacion'] / 100))

        if len(travel_times_list) > 0:
            travel_times_weighted = np.array(travel_times_list)
            kde = stats.gaussian_kde(travel_times_weighted)
            x_range = np.linspace(df['Tiempo_Total_Minutos'].min(),
                                  df['Tiempo_Total_Minutos'].max(), 200)
            y_kde = kde(x_range)
            y_kde_scaled = y_kde * max(bin_populations) / y_kde.max() * 0.8

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

    def _add_reference_lines(self, fig: go.Figure, analyzer: AccessibilityAnalyzer):
        """Add reference lines for accessibility thresholds"""
        pop_by_category, category_pcts = analyzer.get_population_by_category()

        excellent_pct = category_pcts.get(CONFIG.CATEGORY_LABELS['excellent'], 0)
        good_pct = excellent_pct + category_pcts.get(CONFIG.CATEGORY_LABELS['good'], 0)
        fair_pct = good_pct + category_pcts.get(CONFIG.CATEGORY_LABELS['fair'], 0)
        moderate_pct = fair_pct + category_pcts.get(CONFIG.CATEGORY_LABELS['moderate'], 0)
        poor_pct = moderate_pct + category_pcts.get(CONFIG.CATEGORY_LABELS['poor'], 0)

        fig.add_vline(x=30, line_dash="dash", line_color="green",
                      annotation_text=f"30 min<br>({excellent_pct:.1f}%)",
                      annotation_position="top")
        fig.add_vline(x=45, line_dash="dash", line_color="orange",
                      annotation_text=f"45 min<br>({good_pct:.1f}%)",
                      annotation_position="top")
        fig.add_vline(x=60, line_dash="dash", line_color="red",
                      annotation_text=f"60 min<br>({fair_pct:.1f}%)",
                      annotation_position="top")
        fig.add_vline(x=75, line_dash="dash", line_color="darkred",
                      annotation_text=f"75 min<br>({moderate_pct:.1f}%)",
                      annotation_position="top")
        fig.add_vline(x=90, line_dash="dash", line_color="maroon",
                      annotation_text=f"90 min<br>({poor_pct:.1f}%)",
                      annotation_position="top")

    def render_poi_analysis(self, analyzer: AccessibilityAnalyzer):
        """Render POI accessibility breakdown"""
        st.markdown("## 🎯 Accessibility Breakdown by POI")

        poi_analysis = analyzer.get_poi_accessibility()

        if not poi_analysis.empty:
            # Show top 10 POIs
            top_pois = poi_analysis.head(10)

            fig = go.Figure()

            categories = ['Excellent (%)', 'Good (%)', 'Fair (%)', 'Moderate (%)', 'Poor (%)', 'Very Poor (%)']
            colors = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                      CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]
            labels = ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔶 Moderate', '🔴 Poor', '⚫ Very Poor']

            for cat, color, label in zip(categories, colors, labels):
                fig.add_trace(go.Bar(
                    name=label,
                    y=top_pois['POI'],
                    x=top_pois[cat],
                    orientation='h',
                    marker_color=color,
                    text=top_pois[cat].round(1),
                    textposition='inside',
                    texttemplate='%{text:.0f}%'
                ))

            fig = self.styler.apply_standard_styling(
                fig,
                "Top 10 POIs - Accessibility Distribution",
                "% of Population",
                "Point of Interest"
            )
            fig.update_layout(
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

            st.plotly_chart(fig, use_container_width=True)

            # Show detailed table
            st.markdown("### Detailed POI Analysis")
            st.dataframe(
                poi_analysis.style.format({
                    'Avg Time (weighted)': '{:.1f} min',
                    'Total Population': '{:,.0f}',
                    'Excellent (%)': '{:.1f}%',
                    'Good (%)': '{:.1f}%',
                    'Fair (%)': '{:.1f}%',
                    'Moderate (%)': '{:.1f}%',
                    'Poor (%)': '{:.1f}%',
                    'Very Poor (%)': '{:.1f}%'
                }),
                use_container_width=True
            )

    def render_trip_need_analysis(self, analyzer: AccessibilityAnalyzer):
        """Render trip need analysis"""
        st.markdown("## 🔄 Trip Need Analysis")

        # Calculate trip need statistics
        trip_need_stats = analyzer.df.groupby('Zona_Origen').agg({
            'Necesita_Viaje': ['count', 'sum'],
            'Poblacion': 'first',
            'Nombre_Origen': 'first'
        }).reset_index()

        trip_need_stats.columns = ['Zone_ID', 'Total_Trips', 'Trips_Needed', 'Population', 'Zone_Name']
        trip_need_stats['Need_Rate'] = (trip_need_stats['Trips_Needed'] / trip_need_stats['Total_Trips'] * 100)

        # Overall statistics
        col1, col2, col3 = st.columns(3)

        total_trips = trip_need_stats['Total_Trips'].sum()
        total_needed = trip_need_stats['Trips_Needed'].sum()
        overall_rate = (total_needed / total_trips * 100) if total_trips > 0 else 0

        with col1:
            st.metric("Total Trip Connections", f"{total_trips:,}")

        with col2:
            st.metric("Trips Requiring Travel", f"{total_needed:,}")

        with col3:
            st.metric("Overall Need Rate", f"{overall_rate:.1f}%")

        # Trip need distribution
        need_rates = trip_need_stats['Need_Rate'].values
        populations = trip_need_stats['Population'].values

        # Create population-weighted histogram using numpy
        hist, bins = np.histogram(need_rates, bins=20, weights=populations)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        fig = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=hist,
            width=bins[1] - bins[0],
            name="Population",
            text=[f"{val:,.0f}" for val in hist],
            textposition='outside'
        )])

        fig = self.styler.apply_standard_styling(
            fig,
            "Distribution of Trip Need Rates (Population-Weighted)",
            "Trip Need Rate (%)",
            "Population"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Top zones by trip need
        st.markdown("### Zones with Highest Trip Need Rates")
        top_need_zones = trip_need_stats.nlargest(10, 'Need_Rate')

        fig_bar = go.Figure(data=[go.Bar(
            x=top_need_zones['Need_Rate'],
            y=top_need_zones['Zone_Name'],
            orientation='h',
            text=top_need_zones['Need_Rate'].round(1),
            textposition='outside',
            texttemplate='%{text:.1f}%'
        )])

        fig_bar = self.styler.apply_standard_styling(
            fig_bar,
            "Top 10 Zones by Trip Need Rate",
            "Trip Need Rate (%)",
            "Zone"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    def render_geographic_analysis(self, analyzer: AccessibilityAnalyzer):
        """Render geographic analysis by municipality and comarca"""
        st.markdown("## 🗺️ Geographic Analysis by Municipality & Comarca")

        # Municipality analysis
        st.markdown("### Analysis by Municipality")
        muni_analysis = self._calculate_geographic_analysis(analyzer, 'Municipio', 'Municipality')
        self._render_geographic_chart(muni_analysis, 'Municipality', 10)
        self._render_geographic_table(muni_analysis)

        # Comarca analysis
        st.markdown("### Analysis by Comarca")
        comarca_analysis = self._calculate_geographic_analysis(analyzer, 'Comarca', 'Comarca')
        self._render_geographic_chart(comarca_analysis, 'Comarca', 10)
        self._render_geographic_table(comarca_analysis)

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
                tabs = st.tabs([
                    "📊 Population Accessibility",
                    "⏱️ Travel Time Distributions",
                    "🎯 POI Breakdown",
                    "🔄 Trip Need Analysis",
                    "🗺️ Geographic Analysis"
                ])

                with tabs[0]:
                    self.render_accessibility_overview(analyzer)

                with tabs[1]:
                    self.render_travel_time_distributions(analyzer)

                with tabs[2]:
                    self.render_poi_analysis(analyzer)

                with tabs[3]:
                    self.render_trip_need_analysis(analyzer)

                with tabs[4]:
                    self.render_geographic_analysis(analyzer)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()