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

    COLORS = {
        'excellent': '#2ecc71',
        'good': '#f1c40f',
        'fair': '#e67e22',
        'poor': '#e74c3c'
    }

    CATEGORY_LABELS = {
        'excellent': '🟢 Excellent (<30min)',
        'good': '🟡 Good (30-45min)',
        'fair': '🟠 Fair (45-60min)',
        'poor': '🔴 Poor (>60min)'
    }


CONFIG = AccessibilityConfig()


# ============================================================================
# DATA PROCESSING CLASSES
# ============================================================================

class AccessibilityAnalyzer:
    """Main class for accessibility analysis"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.zone_populations = self._calculate_zone_populations()
        self.total_population = self.zone_populations.sum()

    def _calculate_zone_populations(self) -> pd.Series:
        """Calculate population by zone"""
        return self.df.groupby('Zona_Origen')['Poblacion_Origen'].first()

    def categorize_accessibility(self, travel_times: pd.Series) -> pd.Series:
        """Categorize travel times into accessibility levels"""
        return pd.cut(
            travel_times,
            bins=[0, CONFIG.EXCELLENT_THRESHOLD, CONFIG.GOOD_THRESHOLD,
                  CONFIG.FAIR_THRESHOLD, float('inf')],
            labels=list(CONFIG.CATEGORY_LABELS.values())
        )

    def get_zone_best_access(self) -> pd.DataFrame:
        """Get best accessibility for each zone"""
        zone_best = self.df.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'min',
            'Poblacion_Origen': 'first'
        }).reset_index()

        zone_best['Best_Category'] = self.categorize_accessibility(
            zone_best['Tiempo_Viaje_Total_Minutos']
        )
        return zone_best

    def get_population_by_category(self) -> Tuple[pd.Series, pd.Series]:
        """Get population distribution by accessibility category"""
        zone_best = self.get_zone_best_access()
        pop_by_category = zone_best.groupby('Best_Category')['Poblacion_Origen'].sum()
        category_pcts = (pop_by_category / self.total_population * 100) if self.total_population > 0 else pd.Series()
        return pop_by_category, category_pcts

    def get_poi_accessibility(self) -> pd.DataFrame:
        """Calculate accessibility metrics for each POI"""
        poi_data = []

        for poi in self.df['Zona_Destino_nombre'].unique():
            poi_subset = self.df[self.df['Zona_Destino_nombre'] == poi]
            zone_data = poi_subset.groupby('Zona_Origen').agg({
                'Poblacion_Origen': 'first',
                'Tiempo_Viaje_Total_Minutos': 'mean'
            })

            total_pop = zone_data['Poblacion_Origen'].sum()
            if total_pop == 0:
                continue

            # Calculate population in each category
            categories = {}
            categories['excellent'] = zone_data[zone_data['Tiempo_Viaje_Total_Minutos'] <= CONFIG.EXCELLENT_THRESHOLD][
                'Poblacion_Origen'].sum()
            categories['good'] = zone_data[(zone_data['Tiempo_Viaje_Total_Minutos'] > CONFIG.EXCELLENT_THRESHOLD) &
                                           (zone_data['Tiempo_Viaje_Total_Minutos'] <= CONFIG.GOOD_THRESHOLD)][
                'Poblacion_Origen'].sum()
            categories['fair'] = zone_data[(zone_data['Tiempo_Viaje_Total_Minutos'] > CONFIG.GOOD_THRESHOLD) &
                                           (zone_data['Tiempo_Viaje_Total_Minutos'] <= CONFIG.FAIR_THRESHOLD)][
                'Poblacion_Origen'].sum()
            categories['poor'] = zone_data[zone_data['Tiempo_Viaje_Total_Minutos'] > CONFIG.FAIR_THRESHOLD][
                'Poblacion_Origen'].sum()

            weighted_avg_time = (zone_data['Tiempo_Viaje_Total_Minutos'] * zone_data[
                'Poblacion_Origen']).sum() / total_pop

            poi_data.append({
                'POI': poi,
                'Avg Time (weighted)': weighted_avg_time,
                'Total Population': total_pop,
                'Excellent (%)': (categories['excellent'] / total_pop * 100),
                'Good (%)': (categories['good'] / total_pop * 100),
                'Fair (%)': (categories['fair'] / total_pop * 100),
                'Poor (%)': (categories['poor'] / total_pop * 100)
            })

        return pd.DataFrame(poi_data).sort_values('Avg Time (weighted)')

    def get_priority_zones(self, threshold: float = 45) -> pd.DataFrame:
        """Get zones with poor accessibility for prioritization"""
        zone_metrics = self.df.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Zona_Origen_nombre': 'first',
            'Zona_Destino': 'count',
            'Numero_Transbordos': 'mean',
            'Poblacion_Origen': 'first'
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
                tickfont=dict(size=16)  # Axis tick labels
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=18)),
                tickfont=dict(size=16)  # Axis tick labels
            ),
            legend=dict(font=dict(size=16)),
            height=height,
            width=width,
            margin=dict(l=200, r=100, t=100, b=80),
            font=dict(size=16),  # Default font size for all text
            # Ensure annotation text is also sized appropriately
            annotations=[dict(font=dict(size=16))] if fig.layout.annotations else []
        )

        # Update trace text font sizes
        fig.update_traces(
            textfont=dict(size=16),  # Text on bars/points
            hoverlabel=dict(font=dict(size=16))  # Hover text
        )

        # Update colorbar font size if present
        if hasattr(fig.layout, 'coloraxis') and fig.layout.coloraxis:
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(font=dict(size=16)),
                    tickfont=dict(size=16)
                )
            )

        return fig

    @staticmethod
    def apply_pie_chart_styling(fig, title: str, height: int = 400):
        """Apply styling specifically for pie charts"""
        fig.update_layout(
            title=dict(text=title, font=dict(size=24)),
            font=dict(size=16),
            height=height,
            legend=dict(font=dict(size=16))
        )

        fig.update_traces(
            textfont=dict(size=16),
            hoverlabel=dict(font=dict(size=16))
        )

        return fig

    @staticmethod
    def apply_subplot_styling(fig, title: str, height: int = 600):
        """Apply styling for subplots and complex charts"""
        fig.update_layout(
            title=dict(text=title, font=dict(size=24)),
            font=dict(size=16),
            height=height,
            legend=dict(font=dict(size=16))
        )

        # Update all xaxis and yaxis
        fig.update_xaxes(
            titlefont=dict(size=18),
            tickfont=dict(size=16)
        )
        fig.update_yaxes(
            titlefont=dict(size=18),
            tickfont=dict(size=16)
        )

        fig.update_traces(
            textfont=dict(size=16),
            hoverlabel=dict(font=dict(size=16))
        )

        return fig

class ChartGenerator:
    """Generates all visualization charts"""

    def __init__(self, analyzer: AccessibilityAnalyzer):
        self.analyzer = analyzer
        self.styler = ChartStyler()

    def create_population_accessibility_chart(self) -> go.Figure:
        """Create population distribution by accessibility chart"""
        pop_by_category, category_pcts = self.analyzer.get_population_by_category()

        categories = list(CONFIG.CATEGORY_LABELS.values())
        populations = [pop_by_category.get(cat, 0) for cat in categories]
        percentages = [category_pcts.get(cat, 0) for cat in categories]

        fig = px.bar(
            x=categories,
            y=populations,
            color=categories,
            color_discrete_map={
                CONFIG.CATEGORY_LABELS['excellent']: CONFIG.COLORS['excellent'],
                CONFIG.CATEGORY_LABELS['good']: CONFIG.COLORS['good'],
                CONFIG.CATEGORY_LABELS['fair']: CONFIG.COLORS['fair'],
                CONFIG.CATEGORY_LABELS['poor']: CONFIG.COLORS['poor']
            },
            text=[f"{int(pop):,}<br><b>{pct:.1f}%</b>" for pop, pct in zip(populations, percentages)]
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)

        return self.styler.apply_standard_styling(
            fig, "Distribution of Population by Best Accessibility to Any POI",
            "Accessibility Category", "Population (residents)"
        )

    def create_travel_time_distribution(self) -> go.Figure:
        """Create travel time distribution with KDE overlay"""
        df = self.analyzer.df
        zone_populations = self.analyzer.zone_populations
        total_population = self.analyzer.total_population

        # Create bins
        bins = np.linspace(df['Tiempo_Viaje_Total_Minutos'].min(),
                           df['Tiempo_Viaje_Total_Minutos'].max(), 41)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate population-weighted histogram
        bin_populations = []
        for i in range(len(bins) - 1):
            mask = ((df['Tiempo_Viaje_Total_Minutos'] >= bins[i]) &
                    (df['Tiempo_Viaje_Total_Minutos'] < bins[i + 1]))
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
        self._add_reference_lines(fig)

        return self.styler.apply_standard_styling(
            fig, "Travel Time Distribution (Population-Weighted)",
            "Total Travel Time (minutes)", "Percentage of Population (%)"
        )

    def create_travel_time_by_zones_chart(self) -> go.Figure:
        """Create travel time distribution by zones chart"""
        zone_best = self.analyzer.get_zone_best_access()

        fig = px.histogram(
            zone_best,
            x='Tiempo_Viaje_Total_Minutos',
            nbins=40,
            histnorm='percent',
            color_discrete_sequence=['#EBEBEB']
        )

        # Add KDE curve for zones
        travel_times_zones = zone_best['Tiempo_Viaje_Total_Minutos'].dropna()
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
            zones_under_30_pct = (zone_best['Tiempo_Viaje_Total_Minutos'] <= 30).sum() / total_zones * 100
            zones_under_45_pct = (zone_best['Tiempo_Viaje_Total_Minutos'] <= 45).sum() / total_zones * 100
            zones_under_60_pct = (zone_best['Tiempo_Viaje_Total_Minutos'] <= 60).sum() / total_zones * 100

            fig.add_vline(x=30, line_dash="dash", line_color="green",
                          annotation_text=f"30 min<br>({zones_under_30_pct:.1f}% zones)")
            fig.add_vline(x=45, line_dash="dash", line_color="orange",
                          annotation_text=f"45 min<br>({zones_under_45_pct:.1f}% zones)")
            fig.add_vline(x=60, line_dash="dash", line_color="red",
                          annotation_text=f"60 min<br>({zones_under_60_pct:.1f}% zones)")

        return self.styler.apply_standard_styling(
            fig, "Travel Time Distribution by Zones",
            "Total Travel Time (minutes)", "Percentage of Zones (%)"
        )

    def _add_kde_curve(self, fig: go.Figure, df: pd.DataFrame, bin_populations: List[float]):
        """Add KDE curve to histogram"""
        travel_times_list = []
        for _, row in df.iterrows():
            travel_times_list.extend([row['Tiempo_Viaje_Total_Minutos']] * int(row['Poblacion_Origen'] / 100))

        if len(travel_times_list) > 0:
            travel_times_weighted = np.array(travel_times_list)
            kde = stats.gaussian_kde(travel_times_weighted)
            x_range = np.linspace(df['Tiempo_Viaje_Total_Minutos'].min(),
                                  df['Tiempo_Viaje_Total_Minutos'].max(), 200)
            y_kde = kde(x_range)
            y_kde_scaled = y_kde * max(bin_populations) / y_kde.max() * 0.8

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

    def _add_reference_lines(self, fig: go.Figure):
        """Add reference lines for accessibility thresholds"""
        pop_by_category, category_pcts = self.analyzer.get_population_by_category()

        excellent_pct = category_pcts.get(CONFIG.CATEGORY_LABELS['excellent'], 0)
        good_pct = excellent_pct + category_pcts.get(CONFIG.CATEGORY_LABELS['good'], 0)
        fair_pct = good_pct + category_pcts.get(CONFIG.CATEGORY_LABELS['fair'], 0)

        fig.add_vline(x=30, line_dash="dash", line_color="green",
                      annotation_text=f"30 min ({excellent_pct:.1f}%)")
        fig.add_vline(x=45, line_dash="dash", line_color="orange",
                      annotation_text=f"45 min ({good_pct:.1f}%)")
        fig.add_vline(x=60, line_dash="dash", line_color="red",
                      annotation_text=f"60 min ({fair_pct:.1f}%)")

    def create_poi_charts(self) -> Tuple[go.Figure, go.Figure, go.Figure]:
        """Create POI accessibility charts"""
        poi_df = self.analyzer.get_poi_accessibility()

        # Best connected POIs
        best_fig = px.bar(
            poi_df.head(5),
            x='Avg Time (weighted)',
            y='POI',
            orientation='h',
            color='Excellent (%)',
            color_continuous_scale='Greens',
            text='Avg Time (weighted)'
        )
        best_fig.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
        # ADD THIS:
        best_fig = self.styler.apply_standard_styling(
            best_fig, "Best Connected POIs", "Average Travel Time (min)", "POI", height=300
        )
        best_fig.update_layout(yaxis={'categoryorder': 'total ascending'})

        # Worst connected POIs
        worst_fig = px.bar(
            poi_df.tail(5),
            x='Avg Time (weighted)',
            y='POI',
            orientation='h',
            color='Poor (%)',
            color_continuous_scale='Reds',
            text='Avg Time (weighted)'
        )
        worst_fig.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
        # ADD THIS:
        worst_fig = self.styler.apply_standard_styling(
            worst_fig, "Worst Connected POIs", "Average Travel Time (min)", "POI", height=300
        )
        worst_fig.update_layout(yaxis={'categoryorder': 'total descending'})

        # Stacked accessibility breakdown
        breakdown_fig = self._create_poi_breakdown_chart(poi_df)

        return best_fig, worst_fig, breakdown_fig

    def _create_poi_breakdown_chart(self, poi_df: pd.DataFrame) -> go.Figure:
        """Create stacked bar chart for POI accessibility breakdown"""
        poi_sorted = poi_df.sort_values('Avg Time (weighted)')

        fig = go.Figure()

        categories = ['excellent', 'good', 'fair', 'poor']
        category_labels = [CONFIG.CATEGORY_LABELS[cat] for cat in categories]
        colors = [CONFIG.COLORS[cat] for cat in categories]

        for cat, label, color in zip(categories, category_labels, colors):
            col_name = f'{cat.title()} (%)'
            fig.add_trace(go.Bar(
                name=label,
                y=poi_sorted['POI'],
                x=poi_sorted[col_name],
                orientation='h',
                marker_color=color,
                text=poi_sorted[col_name].round(1),
                textposition='inside',
                texttemplate='%{text:.0f}%'
            ))

        fig = self.styler.apply_standard_styling(
            fig,
            'POI Accessibility Quality Distribution (Population-Weighted)',
            'Percentage of Population (%)',
            'Destination POI',
            height=max(400, len(poi_df) * 25)
        )
        fig.update_layout(barmode='stack')

        return fig


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Handles Excel and chart report generation"""

    def __init__(self, analyzer: AccessibilityAnalyzer):
        self.analyzer = analyzer
        self.chart_generator = ChartGenerator(analyzer)

    def create_excel_report(self, poi_df: pd.DataFrame, poor_zones: pd.DataFrame,
                            df_all: Optional[pd.DataFrame] = None,
                            zone_need: Optional[pd.DataFrame] = None) -> BytesIO:
        """Generate comprehensive Excel report"""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Key Metrics Sheet
            self._create_key_metrics_sheet(writer, poi_df, poor_zones, df_all, zone_need)

            # POI Statistics Sheet
            poi_df.to_excel(writer, sheet_name='POI Statistics', index=False)

            # Critical Zones Sheet
            if len(poor_zones) > 0:
                critical_zones = poor_zones[['Zone Name', 'Population', 'Avg Travel Time']].head(15).copy()
                critical_zones.columns = ['Zone Name', 'Residents', 'Avg Travel Time (min)']
                critical_zones.to_excel(writer, sheet_name='Most Critical Zones', index=False)

        output.seek(0)
        return output

    def _create_key_metrics_sheet(self, writer, poi_df: pd.DataFrame, poor_zones: pd.DataFrame,
                                  df_all: Optional[pd.DataFrame], zone_need: Optional[pd.DataFrame]):
        """Create the key metrics sheet"""
        df = self.analyzer.df
        total_population = self.analyzer.total_population

        # Calculate basic metrics
        pop_by_category, category_pcts = self.analyzer.get_population_by_category()
        under_30_pct = category_pcts.get(CONFIG.CATEGORY_LABELS['excellent'], 0)

        # Build metrics data
        metrics_data = {
            'Metric': [
                'Average Travel Time (min)',
                'Average Transfers',
                'Population with <30min access (%)',
                'Total POIs',
                '',
                'ACCESSIBILITY SUMMARY'
            ],
            'Value': [
                f"{df['Tiempo_Viaje_Total_Minutos'].mean():.1f}",
                f"{df['Numero_Transbordos'].mean():.2f}",
                f"{under_30_pct:.1f}",
                df['Zona_Destino'].nunique(),
                '',
                ''
            ]
        }

        # Add accessibility breakdown
        for category in ['excellent', 'good', 'fair', 'poor']:
            label = CONFIG.CATEGORY_LABELS[category]
            pop = pop_by_category.get(label, 0)
            pct = category_pcts.get(label, 0)

            metrics_data['Metric'].extend([
                f'{label} - Population',
                f'{label} - Percentage'
            ])
            metrics_data['Value'].extend([
                f"{pop:,.0f}",
                f"{pct:.1f}%"
            ])

        # Add trip need metrics if available
        if df_all is not None and 'Necesita_viaje' in df_all.columns and zone_need is not None:
            self._add_trip_need_metrics(metrics_data, df_all, zone_need)

        pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Key Metrics', index=False)

    def _add_trip_need_metrics(self, metrics_data: Dict, df_all: pd.DataFrame, zone_need: pd.DataFrame):
        """Add trip necessity metrics to the report"""
        zones_with_need = zone_need['Necesita_viaje'].sum()
        total_zones = len(zone_need)
        zones_need_pct = (zones_with_need / total_zones * 100) if total_zones > 0 else 0

        pop_with_need = zone_need[zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
        total_pop = zone_need['Poblacion_Origen'].sum()
        pop_with_need_pct = (pop_with_need / total_pop * 100) if total_pop > 0 else 0

        pois_with_need = df_all[df_all['Necesita_viaje'] == 1]['Zona_Destino'].nunique()
        total_pois_all = df_all['Zona_Destino'].nunique()
        pois_need_pct = (pois_with_need / total_pois_all * 100) if total_pois_all > 0 else 0

        metrics_data['Metric'].extend([
            '', 'TRIP NEED STATISTICS',
            'Zones with Need of Travel with BizkaiBus (%)',
            'Population with Need to travel with Bizkaibus (%)',
            'POIs with need (%)'
        ])
        metrics_data['Value'].extend([
            '', '',
            f"{zones_need_pct:.1f}",
            f"{pop_with_need_pct:.1f}",
            f"{pois_need_pct:.1f}"
        ])

    def create_all_charts_images(self, poi_df: pd.DataFrame, poor_zones: pd.DataFrame) -> Dict[str, bytes]:
        """Generate all chart images"""
        charts = {}

        # Population accessibility chart
        fig1 = self.chart_generator.create_population_accessibility_chart()
        charts['01_population_accessibility_bar.png'] = pio.to_image(fig1, format='png', width=1200, height=500)

        # Travel time distribution
        fig2 = self.chart_generator.create_travel_time_distribution()
        charts['02_travel_time_population_weighted.png'] = pio.to_image(fig2, format='png', width=1200, height=400)

        # POI charts
        best_fig, worst_fig, breakdown_fig = self.chart_generator.create_poi_charts()
        charts['03_best_connected_pois.png'] = pio.to_image(best_fig, format='png', width=1200, height=300)
        charts['04_worst_connected_pois.png'] = pio.to_image(worst_fig, format='png', width=1200, height=300)
        charts['05_poi_accessibility_breakdown.png'] = pio.to_image(breakdown_fig, format='png', width=1200,
                                                                    height=max(400, len(poi_df) * 25))

        # Priority zones chart
        if len(poor_zones) > 0:
            priority_fig = self._create_priority_zones_chart(poor_zones)
            charts['06_priority_zones.png'] = pio.to_image(priority_fig, format='png', width=1200, height=500)

        return charts

    def _create_priority_zones_chart(self, poor_zones: pd.DataFrame) -> go.Figure:
        """Create priority zones chart"""
        top_poor = poor_zones.head(15)
        fig = px.bar(
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
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        styler = ChartStyler()
        fig = styler.apply_standard_styling(
            fig,
            "High-Population Zones with Worst Accessibility",
            "Population",
            "Zone Name",
            height=500
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

class StreamlitApp:
    """Main Streamlit application"""

    def __init__(self):
        self.setup_page_config()

    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Bizkaia Accessibility Analysis",
            layout="wide",
            page_icon="🚌"
        )

    def render_header(self):
        """Render application header"""
        st.title("🚌 Bizkaia Public Transport Accessibility Analysis")
        st.markdown("""
        **Policy-focused analysis of Bizkaibus service accessibility**  
        *Identifying priority areas for service improvement based on population and travel time*
        """)

    def render_methodology(self):
        """Render methodology expander"""
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

    def process_uploaded_file(self, uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame, AccessibilityAnalyzer]:
        """Process uploaded Excel file and create analyzer"""
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()

        # Handle trip necessity
        df_all = df.copy()
        if 'Necesita_viaje' in df.columns:
            df = df[df['Necesita_viaje'] == 1].copy()

        return df, df_all, AccessibilityAnalyzer(df)

    def render_sidebar(self, df: pd.DataFrame, analyzer: AccessibilityAnalyzer, df_all: pd.DataFrame):
        """Render sidebar with filters and metrics"""
        st.sidebar.header("🔍 Filters")

        # Transport filter
        transport_filter = st.sidebar.radio(
            "Transport Mode",
            ["Public transport only", "All trips"],
            index=0
        )

        if transport_filter == "Public transport only" and 'Usa_Transporte_Publico' in df.columns:
            filtered_df = df[df['Usa_Transporte_Publico'] == True].copy()
            analyzer = AccessibilityAnalyzer(filtered_df)

        # Display metrics
        st.sidebar.markdown(f"**Analyzing {len(df):,} origin-destination pairs**")
        st.sidebar.markdown(f"**Total population: {analyzer.total_population:,.0f} residents**")
        st.sidebar.markdown(f"**Origin zones: {df['Zona_Origen'].nunique()}**")

        # Trip necessity metrics
        if 'Necesita_viaje' in df_all.columns:
            self._render_trip_necessity_metrics(st.sidebar, df_all)

        return analyzer

    def _render_trip_necessity_metrics(self, sidebar, df_all: pd.DataFrame):
        """Render trip necessity metrics in sidebar"""
        total_trips = len(df_all)
        necessary_trips = (df_all['Necesita_viaje'] == 1).sum()
        unnecessary_trips = total_trips - necessary_trips
        unnecessary_pct = (unnecessary_trips / total_trips * 100) if total_trips > 0 else 0

        sidebar.markdown("---")
        sidebar.markdown("**Trip Necessity:**")
        sidebar.markdown(f"✅ Necessary trips: {necessary_trips:,} ({100 - unnecessary_pct:.1f}%)")
        sidebar.markdown(f"⌫ Unnecessary trips: {unnecessary_trips:,} ({unnecessary_pct:.1f}%)")

    def render_key_metrics(self, analyzer: AccessibilityAnalyzer):
        """Render key performance indicators"""
        st.header("📊 Key Metrics")

        df = analyzer.df
        pop_by_category, category_pcts = analyzer.get_population_by_category()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_travel_time = df['Tiempo_Viaje_Total_Minutos'].mean()
            st.metric("Avg Travel Time", f"{avg_travel_time:.1f} min")

        with col2:
            avg_transfers = df['Numero_Transbordos'].mean()
            st.metric("Avg Transfers", f"{avg_transfers:.2f}")

        with col3:
            under_30_pct = category_pcts.get(CONFIG.CATEGORY_LABELS['excellent'], 0)
            st.metric("Population with <30min access", f"{under_30_pct:.1f}%")

        with col4:
            total_pois = df['Zona_Destino'].nunique()
            st.metric("Total POIs", f"{total_pois}")

    def render_accessibility_overview_tab(self, analyzer: AccessibilityAnalyzer):
        """Render accessibility overview tab content"""
        st.subheader("Population Accessibility Distribution")
        st.markdown("*What percentage of the population has good vs poor connections to POIs?*")

        # Population accessibility chart
        chart_gen = ChartGenerator(analyzer)
        fig1 = chart_gen.create_population_accessibility_chart()
        st.plotly_chart(fig1, use_container_width=True)

        # Summary cards
        pop_by_category, category_pcts = analyzer.get_population_by_category()
        self._render_accessibility_summary_cards(pop_by_category, category_pcts)

        # Travel time distributions
        st.markdown("---")
        st.markdown("### Travel Time Distribution (Population-Weighted)")

        fig2 = chart_gen.create_travel_time_distribution()
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### Travel Time Distribution by Zones")

        fig3 = chart_gen.create_travel_time_by_zones_chart()
        st.plotly_chart(fig3, use_container_width=True)

    def _render_accessibility_summary_cards(self, pop_by_category: pd.Series, category_pcts: pd.Series):
        """Render accessibility summary cards"""
        st.markdown("### Summary")

        categories = ['excellent', 'good', 'fair', 'poor']
        for category in categories:
            label = CONFIG.CATEGORY_LABELS[category]
            pop = pop_by_category.get(label, 0)
            pct = category_pcts.get(label, 0)

            if category in ['excellent', 'good']:
                st.success(f"**{label}**\n\n{pop:,.0f} residents ({pct:.1f}%)")
            elif category == 'fair':
                st.warning(f"**{label}**\n\n{pop:,.0f} residents ({pct:.1f}%)")
            else:
                st.error(f"**{label}**\n\n{pop:,.0f} residents ({pct:.1f}%)")

    def render_poi_analysis_tab(self, analyzer: AccessibilityAnalyzer):
        """Render POI analysis tab content"""
        st.subheader("Point of Interest (POI) Accessibility")
        st.markdown("*Which destinations are well-connected and which need improvement?*")

        poi_df = analyzer.get_poi_accessibility()
        chart_gen = ChartGenerator(analyzer)

        # POI charts
        best_fig, worst_fig, breakdown_fig = chart_gen.create_poi_charts()

        st.markdown("#### 🟢 Best Connected POIs")
        st.plotly_chart(best_fig, use_container_width=True)

        st.markdown("#### 🔴 Worst Connected POIs")
        st.plotly_chart(worst_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Accessibility Breakdown by POI")
        st.plotly_chart(breakdown_fig, use_container_width=True)

        # Detailed POI table
        st.markdown("#### Detailed POI Statistics")
        self._render_poi_table(poi_df)

    def _render_poi_table(self, poi_df: pd.DataFrame):
        """Render POI statistics table"""
        display_df = poi_df[
            ['POI', 'Avg Time (weighted)', 'Total Population',
             'Excellent (%)', 'Good (%)', 'Fair (%)', 'Poor (%)']
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

    def render_priority_areas_tab(self, analyzer: AccessibilityAnalyzer):
        """Render priority areas tab content"""
        st.subheader("Priority Areas for Service Improvement")
        st.markdown("*Origin zones with poor accessibility affecting the most residents*")

        poor_zones = analyzer.get_priority_zones()

        if len(poor_zones) > 0:
            self._render_priority_zones_content(poor_zones, analyzer.total_population)
        else:
            st.success("✅ No zones with poor average accessibility found!")

    def _render_priority_zones_content(self, poor_zones: pd.DataFrame, total_population: float):
        """Render priority zones content"""
        st.markdown("#### 🔴 High-Population Zones with Worst Accessibility")
        st.markdown(
            f"**{len(poor_zones)} zones** with **{poor_zones['Population'].sum():,.0f} residents** "
            f"have average travel times exceeding 45 minutes"
        )

        # Priority zones chart
        priority_fig = self._create_priority_zones_chart(poor_zones)
        st.plotly_chart(priority_fig, use_container_width=True)

        # Impact metrics
        self._render_impact_metrics(poor_zones, total_population)

        # Critical zones list
        self._render_critical_zones_list(poor_zones)

    def _create_priority_zones_chart(self, poor_zones: pd.DataFrame) -> go.Figure:
        """Create priority zones chart"""
        top_poor = poor_zones.head(15)
        fig = px.bar(
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
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        return fig

    def _render_impact_metrics(self, poor_zones: pd.DataFrame, total_population: float):
        """Render impact metrics for priority zones"""
        st.markdown("#### Key Impact Metrics")

        avg_poor = poor_zones['Avg Travel Time'].mean()
        avg_transfers_poor = poor_zones['Avg Transfers'].mean()
        total_affected_pop = poor_zones['Population'].sum()
        pct_affected = (total_affected_pop / total_population * 100) if total_population > 0 else 0

        st.error(f"**Avg travel time in poor zones:** {avg_poor:.1f} minutes")
        st.warning(f"**Avg transfers needed:** {avg_transfers_poor:.2f}")
        st.info(f"**Population affected:** {total_affected_pop:,.0f} residents ({pct_affected:.1f}% of total)")

    def _render_critical_zones_list(self, poor_zones: pd.DataFrame):
        """Render list of most critical zones"""
        st.markdown("#### Most Critical Zones")
        st.markdown("*(by population affected)*")

        for idx, row in poor_zones.head(5).iterrows():
            st.write(f"**{row['Zone Name']}**")
            st.write(f"→ {row['Population']:,.0f} residents")
            st.write(f"→ {row['Avg Travel Time']:.1f} min avg")
            st.markdown("---")

    def render_distance_efficiency_tab(self, analyzer: AccessibilityAnalyzer):
        """Render distance efficiency tab content"""
        st.subheader("Distance vs Travel Time Efficiency")
        st.markdown("*Identifying where service is slow despite short distances*")

        df = analyzer.df.copy()

        # Calculate efficiency metrics
        df['Time_per_Km'] = df['Tiempo_Viaje_Total_Minutos'] / df['Distancia_Viaje_Total_Km']
        df['Time_per_Km'] = df['Time_per_Km'].replace([np.inf, -np.inf], np.nan)

        # Scatter plot
        self._render_efficiency_scatter_plot(df)

        # Efficiency metrics
        self._render_efficiency_metrics(df)

        # Zone efficiency analysis
        self._render_zone_efficiency_analysis(df)

    def _render_efficiency_scatter_plot(self, df: pd.DataFrame):
        """Render efficiency scatter plot"""
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size) if len(df) > sample_size else df

        fig = px.scatter(
            sample_df,
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

        # Add ideal speed reference line
        max_dist = df['Distancia_Viaje_Total_Km'].max()
        fig.add_trace(go.Scatter(
            x=[0, max_dist],
            y=[0, max_dist * 3],
            mode='lines',
            name='Ideal (20 km/h avg)',
            line=dict(color='green', dash='dash', width=2)
        ))

        styler = ChartStyler()
        fig = styler.apply_standard_styling(
            fig,
            'Travel Time vs Distance (points above green line are slower than 20 km/h average)',
            'Distance (km)',
            'Travel Time (min)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_efficiency_metrics(self, df: pd.DataFrame):
        """Render efficiency metrics"""
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_time_per_km = df['Time_per_Km'].mean()
            st.metric("Avg Time per Km", f"{avg_time_per_km:.1f} min/km")

        with col2:
            avg_speed = 60 / avg_time_per_km if avg_time_per_km > 0 else 0
            st.metric("Avg Effective Speed", f"{avg_speed:.1f} km/h")

        with col3:
            inefficient_pct = (df['Time_per_Km'] > 3).sum() / len(df) * 100
            st.metric("Slow Connections", f"{inefficient_pct:.1f}%",
                      help="Connections slower than 20 km/h (>3 min/km)")

    def _render_zone_efficiency_analysis(self, df: pd.DataFrame):
        """Render zone efficiency analysis"""
        st.markdown("---")
        st.markdown("#### Travel Efficiency by Zone")
        st.markdown("*All zones by population, sorted by efficiency*")

        zone_efficiency = df.groupby('Zona_Origen').agg({
            'Time_per_Km': 'mean',
            'Distancia_Viaje_Total_Km': 'mean',
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Poblacion_Origen': 'first',
            'Zona_Origen_nombre': 'first'
        }).reset_index()

        zone_efficiency.columns = ['Zone_ID', 'Min per Km', 'Avg Distance (km)',
                                   'Avg Time (min)', 'Population', 'Zone Name']
        zone_efficiency['Effective Speed (km/h)'] = 60 / zone_efficiency['Min per Km']

        # Chart
        top_zones = zone_efficiency.sort_values('Min per Km', ascending=False)
        worse_zones = zone_efficiency.sort_values('Min per Km', ascending=False).head(25)

        fig = px.bar(
            worse_zones.sort_values('Min per Km', ascending=False),
            x='Zone Name',
            y='Min per Km',
            color='Min per Km',
            color_continuous_scale='Reds',
            hover_data={
                'Effective Speed (km/h)': ':.1f',
                'Avg Distance (km)': ':.1f',
                'Avg Time (min)': ':.1f',
                'Population': ':,.0f'
            },
            title='Travel Time Efficiency by Zone (lower = better)'
        )
        styler = ChartStyler()
        fig = styler.apply_standard_styling(
            fig,
            'Travel Time Efficiency by Zone (lower = better)',
            'Zone Name',
            'Min per Km',
            height=400
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        # Data table
        st.markdown("#### Zone Efficiency Rankings")
        self._render_efficiency_table(top_zones)

    def _render_efficiency_table(self, zone_efficiency: pd.DataFrame):
        """Render zone efficiency table"""
        display_df = zone_efficiency.sort_values('Min per Km')[
            ['Zone Name', 'Population', 'Min per Km', 'Avg Distance (km)',
             'Avg Time (min)', 'Effective Speed (km/h)']
        ]

        st.dataframe(
            display_df.style.format({
                'Population': '{:,.0f}',
                'Min per Km': '{:.2f}',
                'Avg Distance (km)': '{:.1f}',
                'Avg Time (min)': '{:.1f}',
                'Effective Speed (km/h)': '{:.1f}'
            }).background_gradient(subset=['Min per Km'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True
        )

    def render_trip_need_tab(self, analyzer: AccessibilityAnalyzer, df_all: pd.DataFrame):
        """Render trip need analysis tab"""
        st.subheader("Trip Need Analysis")
        st.markdown("*Understanding which origin-destination pairs have actual travel need*")

        if 'Necesita_viaje' not in df_all.columns:
            st.warning("⚠️ 'Necesita_viaje' column not found in the dataset")
            return

        # Calculate need statistics
        zone_need = self._calculate_zone_need_stats(df_all)

        # Overall statistics
        self._render_need_overview_metrics(zone_need, df_all)

        # Population breakdown
        self._render_population_need_breakdown(zone_need)

        # Zones with no need
        self._render_zones_no_need(zone_need)

        # POI need analysis
        self._render_poi_need_analysis(df_all, zone_need, analyzer.total_population)

    def _calculate_zone_need_stats(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Calculate zone-level need statistics"""
        zone_need = df_all.groupby('Zona_Origen').agg({
            'Necesita_viaje': 'sum',
            'Poblacion_Origen': 'first',
            'Zona_Origen_nombre': 'first'
        }).reset_index()
        zone_need['Necesita_viaje'] = zone_need['Necesita_viaje'] > 0
        return zone_need

    def _render_need_overview_metrics(self, zone_need: pd.DataFrame, df_all: pd.DataFrame):
        """Render overview metrics for trip need"""
        st.markdown("### Overall Need Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            zones_with_need = zone_need['Necesita_viaje'].sum()
            total_zones = len(zone_need)
            zones_need_pct = (zones_with_need / total_zones * 100) if total_zones > 0 else 0
            st.metric("Zones with Need of Travel with BizkaiBus", f"{zones_need_pct:.1f}%",
                      help=f"{zones_with_need} out of {total_zones} zones need to reach at least one POI")

        with col2:
            pop_with_need = zone_need[zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
            total_pop = zone_need['Poblacion_Origen'].sum()
            pop_with_need_pct = (pop_with_need / total_pop * 100) if total_pop > 0 else 0
            st.metric("Population with Need to travel with Bizkaibus", f"{pop_with_need_pct:.1f}%",
                      help=f"{pop_with_need:,.0f} out of {total_pop:,.0f} residents live in zones that need trips")

        with col3:
            pois_with_need = df_all[df_all['Necesita_viaje'] == 1]['Zona_Destino'].nunique()
            total_pois_all = df_all['Zona_Destino'].nunique()
            pois_need_pct = (pois_with_need / total_pois_all * 100) if total_pois_all > 0 else 0
            st.metric("POIs with need", f"{pois_need_pct:.1f}%",
                      help=f"{pois_with_need} out of {total_pois_all} POIs have at least one origin zone that needs them")

    def _render_population_need_breakdown(self, zone_need: pd.DataFrame):
        """Render population breakdown by trip necessity"""
        st.markdown("---")
        st.markdown("### Population Distribution by Trip Necessity")

        pop_with_need = zone_need[zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
        pop_without_need = zone_need[~zone_need['Necesita_viaje']]['Poblacion_Origen'].sum()
        total_pop = zone_need['Poblacion_Origen'].sum()
        pop_with_need_pct = (pop_with_need / total_pop * 100) if total_pop > 0 else 0

        pop_data = pd.DataFrame({
            'Category': ['Residents needing trips', 'Residents not needing trips'],
            'Population': [pop_with_need, pop_without_need],
            'Percentage': [pop_with_need_pct, 100 - pop_with_need_pct]
        })

        fig = px.pie(
            pop_data,
            values='Population',
            names='Category',
            color='Category',
            color_discrete_map={
                'Residents needing trips': '#3498db',
                'Residents not needing trips': '#95a5a6'
            },
            hole=0.4,
            title='Population by Trip Necessity'
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            texttemplate='<b>%{label}</b><br>%{value:,.0f} residents<br>(%{percent})'
        )
        styler = ChartStyler()
        fig = styler.apply_pie_chart_styling(
            fig,
            'Population by Trip Necessity',
            height=400
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_zones_no_need(self, zone_need: pd.DataFrame):
        """Render zones with no trip need"""
        zones_no_need = zone_need[~zone_need['Necesita_viaje']].sort_values('Poblacion_Origen', ascending=False)

        if len(zones_no_need) > 0:
            pop_without_need = zones_no_need['Poblacion_Origen'].sum()
            st.markdown("#### Zones with No Trip Need")
            st.markdown(
                f"*{len(zones_no_need)} zones with {pop_without_need:,.0f} residents don't need to reach any POI*")

            display_zones = zones_no_need.head(10) if len(zones_no_need) > 10 else zones_no_need
            if len(zones_no_need) > 10:
                st.markdown("**Top 10 by population:**")

            for idx, row in display_zones.iterrows():
                st.write(f"**{row['Zona_Origen_nombre']}**: {row['Poblacion_Origen']:,.0f} residents")
        else:
            st.success("All zones have trip need to at least one POI!")

    def _render_poi_need_analysis(self, df_all: pd.DataFrame, zone_need: pd.DataFrame, total_pop: float):
        """Render POI need analysis"""
        st.markdown("---")
        st.markdown("### Trip Need by Destination POI")
        st.markdown("*Showing both trip-based and population-based need*")

        # Calculate POI need metrics
        poi_need_data = []
        for poi in df_all['Zona_Destino_nombre'].unique():
            poi_data = df_all[df_all['Zona_Destino_nombre'] == poi]

            trips_needed = poi_data['Necesita_viaje'].sum()
            total_possible = len(poi_data)
            trip_need_rate = (trips_needed / total_possible * 100) if total_possible > 0 else 0

            zones_needing = poi_data[poi_data['Necesita_viaje'] == 1]['Zona_Origen'].unique()
            pop_needing = zone_need[zone_need['Zona_Origen'].isin(zones_needing)]['Poblacion_Origen'].sum()
            pop_need_rate = (pop_needing / total_pop * 100) if total_pop > 0 else 0

            poi_need_data.append({
                'POI': poi,
                'Trips_Needed': trips_needed,
                'Total_Possible_Trips': total_possible,
                'Trip_Need_Rate_%': trip_need_rate,
                'Population_Needing': pop_needing,
                'Population_Need_Rate_%': pop_need_rate
            })

        need_by_poi = pd.DataFrame(poi_need_data).sort_values('Population_Need_Rate_%', ascending=False)

        # Create dual bar chart
        self._render_poi_need_chart(need_by_poi)

        # Detailed table
        self._render_poi_need_table(need_by_poi)

    def _render_poi_need_chart(self, need_by_poi: pd.DataFrame):
        """Render POI need chart"""
        top_15 = need_by_poi.head(15)

        fig = go.Figure()

        fig.add_trace(go.Bar(
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

        fig.add_trace(go.Bar(
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

        styler = ChartStyler()
        fig = styler.apply_standard_styling(
            fig,
            'Top 15 POIs by Need Rate (Trip-based vs Population-based)',
            'Need Rate (%)',
            'Destination POI',
            height=600
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_poi_need_table(self, need_by_poi: pd.DataFrame):
        """Render POI need detailed table"""
        st.markdown("#### Detailed Need Statistics")

        display_df = need_by_poi[
            ['POI', 'Trips_Needed', 'Total_Possible_Trips', 'Trip_Need_Rate_%',
             'Population_Needing', 'Population_Need_Rate_%']
        ].copy()

        st.dataframe(
            display_df.style.format({
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

    def render_geographic_analysis_tab(self, analyzer: AccessibilityAnalyzer):
        """Render geographic analysis tab with filtering capabilities"""
        st.subheader("Geographic Analysis by Municipality & Comarca")
        st.markdown("*Accessibility patterns across municipalities and comarcas*")

        df = analyzer.df

        if 'Municipio' not in df.columns or 'Comarca' not in df.columns:
            st.warning("⚠️ 'Municipio' or 'Comarca' columns not found in the dataset")
            return

        # Create filter columns
        col1, col2 = st.columns(2)

        with col1:
            # Municipality filter
            municipalities = ['All'] + sorted(df['Municipio'].unique().tolist())
            selected_municipality = st.selectbox(
                "Filter by Municipality:",
                municipalities,
                index=0
            )

        with col2:
            # Comarca filter
            comarcas = ['All'] + sorted(df['Comarca'].unique().tolist())
            selected_comarca = st.selectbox(
                "Filter by Comarca:",
                comarcas,
                index=0
            )

        # Apply filters
        filtered_df = df.copy()

        if selected_municipality != 'All':
            filtered_df = filtered_df[filtered_df['Municipio'] == selected_municipality]

        if selected_comarca != 'All':
            filtered_df = filtered_df[filtered_df['Comarca'] == selected_comarca]

        # Check if we have data after filtering
        if len(filtered_df) == 0:
            st.warning("No data available for the selected filters.")
            return

        # Display filter status
        if selected_municipality != 'All' or selected_comarca != 'All':
            filter_text = []
            if selected_municipality != 'All':
                filter_text.append(f"Municipality: **{selected_municipality}**")
            if selected_comarca != 'All':
                filter_text.append(f"Comarca: **{selected_comarca}**")

            st.info(f"Showing data for: {' | '.join(filter_text)}")

            # Show summary stats for filtered data
            total_zones = filtered_df['Zona_Origen'].nunique()
            total_population = filtered_df.groupby('Zona_Origen')['Poblacion_Origen'].first().sum()
            avg_travel_time = filtered_df['Tiempo_Viaje_Total_Minutos'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Zones in Selection", f"{total_zones}")
            with col2:
                st.metric("Population in Selection", f"{total_population:,.0f}")
            with col3:
                st.metric("Avg Travel Time", f"{avg_travel_time:.1f} min")

        # Municipality analysis
        self._render_municipality_analysis_filtered(filtered_df)

        # Comarca analysis
        st.markdown("---")
        self._render_comarca_analysis_filtered(filtered_df)

    def _render_municipality_analysis_filtered(self, df: pd.DataFrame):
        """Render municipality analysis with filtered data"""
        st.markdown("### Analysis by Municipality")

        if df['Municipio'].nunique() == 1:
            # Single municipality selected - show zone-level analysis
            municipality_name = df['Municipio'].iloc[0]
            st.markdown(f"#### Zone-level Analysis for {municipality_name}")
            self._render_zone_level_analysis(df, municipality_name)
        else:
            # Multiple municipalities - show municipality comparison
            muni_analysis = self._calculate_geographic_metrics(df, 'Municipio', 'Municipality')
            self._render_geographic_charts(muni_analysis, 'Municipality', 'Municipalities', 15)

            st.markdown("#### Detailed Municipality Statistics")
            self._render_geographic_table(muni_analysis)

    def _render_comarca_analysis_filtered(self, df: pd.DataFrame):
        """Render comarca analysis with filtered data"""
        st.markdown("### Analysis by Comarca")

        if df['Comarca'].nunique() == 1:
            # Single comarca selected - show municipality breakdown within comarca
            comarca_name = df['Comarca'].iloc[0]
            st.markdown(f"#### Municipality Breakdown for {comarca_name}")

            if df['Municipio'].nunique() > 1:
                # Multiple municipalities in the comarca
                muni_in_comarca = self._calculate_geographic_metrics(df, 'Municipio', 'Municipality')
                self._render_geographic_charts(muni_in_comarca, 'Municipality', 'Municipalities in Comarca',
                                               len(muni_in_comarca))

                st.markdown("#### Municipality Statistics within Comarca")
                self._render_geographic_table(muni_in_comarca)
            else:
                # Single municipality in comarca - show zone analysis
                municipality_name = df['Municipio'].iloc[0]
                st.markdown(f"#### Zone-level Analysis for {municipality_name} (in {comarca_name})")
                self._render_zone_level_analysis(df, f"{municipality_name} ({comarca_name})")
        else:
            # Multiple comarcas - show comarca comparison
            comarca_analysis = df.groupby('Comarca').agg({
                'Poblacion_Origen': 'sum',
                'Tiempo_Viaje_Total_Minutos': 'mean',
                'Numero_Transbordos': 'mean',
                'Zona_Origen': 'nunique',
                'Municipio': 'nunique'
            }).reset_index()
            comarca_analysis.columns = ['Geographic Unit', 'Total Population', 'Avg Travel Time',
                                        'Avg Transfers', 'Zones', 'Municipalities']

            comarca_analysis = self._add_accessibility_percentages(df, comarca_analysis, 'Comarca')
            self._render_geographic_charts(comarca_analysis, 'Comarca', 'Comarcas', len(comarca_analysis))

            st.markdown("#### Detailed Comarca Statistics")
            self._render_geographic_table(comarca_analysis, include_municipalities=True)

    def _render_zone_level_analysis(self, df: pd.DataFrame, area_name: str):
        """Render zone-level analysis for a specific area"""
        # Calculate zone-level metrics
        zone_analysis = df.groupby(['Zona_Origen', 'Zona_Origen_nombre']).agg({
            'Poblacion_Origen': 'first',
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Numero_Transbordos': 'mean',
            'Zona_Destino': 'nunique'
        }).reset_index()
        zone_analysis.columns = ['Zone_ID', 'Zone_Name', 'Population', 'Avg_Travel_Time', 'Avg_Transfers',
                                 'POIs_Served']

        # Calculate accessibility categories for each zone
        zone_best = df.groupby('Zona_Origen').agg({
            'Tiempo_Viaje_Total_Minutos': 'min',
            'Poblacion_Origen': 'first',
            'Zona_Origen_nombre': 'first'
        }).reset_index()

        # Create accessibility analyzer for this subset
        subset_analyzer = AccessibilityAnalyzer(df)
        zone_best['Best_Category'] = subset_analyzer.categorize_accessibility(zone_best['Tiempo_Viaje_Total_Minutos'])

        # Merge with zone analysis
        zone_analysis = zone_analysis.merge(
            zone_best[['Zona_Origen', 'Best_Category']],
            left_on='Zone_ID',
            right_on='Zona_Origen',
            how='left'
        ).drop('Zona_Origen', axis=1)

        # Sort by population (highest first)
        zone_analysis = zone_analysis.sort_values('Population', ascending=False)

        # Create zone-level accessibility chart
        fig = px.bar(
            zone_analysis.head(20),  # Show top 20 zones by population
            x='Population',
            y='Zone_Name',
            orientation='h',
            color='Avg_Travel_Time',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Avg_Transfers', 'POIs_Served'],
            text='Population',
            title=f"Top 20 Zones by Population in {area_name}"
        )

        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')

        styler = ChartStyler()
        fig = styler.apply_standard_styling(
            fig,
            f"Top 20 Zones by Population in {area_name}",
            "Population",
            "Zone Name",
            height=600
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Accessibility distribution as single horizontal stacked bar (compact view)
        pop_by_category = zone_best.groupby('Best_Category')['Poblacion_Origen'].sum()
        total_pop = zone_best['Poblacion_Origen'].sum()

        if total_pop > 0:
            fig_access = go.Figure()

            categories = ['excellent', 'good', 'fair', 'poor']
            category_labels = [CONFIG.CATEGORY_LABELS[cat] for cat in categories]
            colors = [CONFIG.COLORS[cat] for cat in categories]

            for cat, label, color in zip(categories, category_labels, colors):
                pop = pop_by_category.get(label, 0)
                pct = (pop / total_pop * 100) if total_pop > 0 else 0

                fig_access.add_trace(go.Bar(
                    name=label,
                    y=[area_name],
                    x=[pct],
                    orientation='h',
                    marker_color=color,
                    text=f'{pct:.0f}%',
                    textposition='inside',
                    texttemplate='%{text}',
                    hovertemplate=f'<b>{label}</b><br>Population: {pop:,.0f}<br>Percentage: {pct:.1f}%<extra></extra>'
                ))

            styler = ChartStyler()
            fig_access = styler.apply_standard_styling(
                fig_access,
                f'Accessibility Distribution in {area_name}',
                'Percentage of Population (%)',
                '',
                height=210

            )
            fig_access.update_layout(
                barmode='stack',
                yaxis=dict(showticklabels=False),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=16)
                )
            )
            st.plotly_chart(fig_access, use_container_width=True)

        # Zone statistics table
        st.markdown(f"#### Zone Statistics for {area_name}")
        display_df = zone_analysis[
            ['Zone_Name', 'Population', 'Avg_Travel_Time', 'Avg_Transfers', 'POIs_Served', 'Best_Category']]

        st.dataframe(
            display_df.style.format({
                'Population': '{:,.0f}',
                'Avg_Travel_Time': '{:.1f} min',
                'Avg_Transfers': '{:.2f}',
                'POIs_Served': '{:.0f}'
            }).background_gradient(subset=['Avg_Travel_Time'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    def _calculate_geographic_metrics(self, df: pd.DataFrame, group_col: str, unit_name: str) -> pd.DataFrame:
        """Calculate geographic analysis metrics"""
        analysis = df.groupby(group_col).agg({
            'Poblacion_Origen': 'sum',
            'Tiempo_Viaje_Total_Minutos': 'mean',
            'Numero_Transbordos': 'mean',
            'Zona_Origen': 'nunique'
        }).reset_index()
        analysis.columns = ['Geographic Unit', 'Total Population', 'Avg Travel Time', 'Avg Transfers', 'Zones']

        # Add accessibility category percentages
        analysis = self._add_accessibility_percentages(df, analysis, group_col)

        return analysis.sort_values('Avg Travel Time', ascending=False)

    def _add_accessibility_percentages(self, df: pd.DataFrame, analysis: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Add accessibility category percentages to geographic analysis"""
        for unit in df[group_col].unique():
            unit_data = df[df[group_col] == unit]
            zone_best = unit_data.groupby('Zona_Origen').agg({
                'Tiempo_Viaje_Total_Minutos': 'min',
                'Poblacion_Origen': 'first'
            }).reset_index()

            total_pop_unit = zone_best['Poblacion_Origen'].sum()

            if total_pop_unit > 0:
                excellent = zone_best[zone_best['Tiempo_Viaje_Total_Minutos'] <= 30][
                                'Poblacion_Origen'].sum() / total_pop_unit * 100
                good = zone_best[(zone_best['Tiempo_Viaje_Total_Minutos'] > 30) &
                                 (zone_best['Tiempo_Viaje_Total_Minutos'] <= 45)][
                           'Poblacion_Origen'].sum() / total_pop_unit * 100
                fair = zone_best[(zone_best['Tiempo_Viaje_Total_Minutos'] > 45) &
                                 (zone_best['Tiempo_Viaje_Total_Minutos'] <= 60)][
                           'Poblacion_Origen'].sum() / total_pop_unit * 100
                poor = zone_best[zone_best['Tiempo_Viaje_Total_Minutos'] > 60][
                           'Poblacion_Origen'].sum() / total_pop_unit * 100
            else:
                excellent = good = fair = poor = 0

            mask = analysis['Geographic Unit'] == unit
            analysis.loc[mask, 'Excellent (%)'] = excellent
            analysis.loc[mask, 'Good (%)'] = good
            analysis.loc[mask, 'Fair (%)'] = fair
            analysis.loc[mask, 'Poor (%)'] = poor

        return analysis

    def _render_geographic_charts(self, analysis: pd.DataFrame, unit_type: str, unit_plural: str, top_n: int):
        """Render geographic analysis charts"""
        # Travel time chart
        chart_data = analysis.head(top_n) if top_n < len(analysis) else analysis

        fig_time = px.bar(
            chart_data,
            x='Avg Travel Time',
            y='Geographic Unit',
            orientation='h',
            color='Total Population',
            color_continuous_scale='Blues' if 'Municip' in unit_type else 'Reds',
            text='Avg Travel Time',
            title=f"Top {top_n} {unit_plural} by Average Travel Time" if top_n < len(
                analysis) else f"{unit_plural} by Average Travel Time"
        )
        fig_time.update_traces(texttemplate='%{text:.1f} min', textposition='outside')

        # FIX: Add the missing y_title argument
        styler = ChartStyler()
        fig_time = styler.apply_standard_styling(
            fig_time,
            f"Top {top_n} {unit_plural} by Average Travel Time" if top_n < len(
                analysis) else f"{unit_plural} by Average Travel Time",
            "Average Travel Time (min)",  # x_title
            unit_type,  # y_title - THIS WAS MISSING
            height=800
        )
        fig_time.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            coloraxis_colorbar=dict(
                title=dict(font=dict(size=16)),
                tickfont=dict(size=16)
            )
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Accessibility breakdown chart
        pop_sorted = analysis.sort_values('Total Population', ascending=False)
        chart_data = pop_sorted.head(top_n) if top_n < len(pop_sorted) else pop_sorted

        fig_access = go.Figure()

        categories = ['Excellent (%)', 'Good (%)', 'Fair (%)', 'Poor (%)']
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        labels = ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔴 Poor']

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

        # Apply ChartStyler to the stacked bar chart too
        fig_access = styler.apply_standard_styling(
            fig_access,
            title,
            '% of Population',  # x_title
            unit_type,  # y_title
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
                x=0.5,
                font=dict(size=16)
            )
        )
        st.plotly_chart(fig_access, use_container_width=True)

    def _render_geographic_table(self, analysis: pd.DataFrame, include_municipalities: bool = False):
        """Render geographic analysis table"""
        columns = ['Geographic Unit', 'Total Population', 'Avg Travel Time', 'Avg Transfers', 'Zones']
        if include_municipalities:
            columns.append('Municipalities')
        columns.extend(['Excellent (%)', 'Good (%)', 'Fair (%)', 'Poor (%)'])

        display_df = analysis[columns]

        format_dict = {
            'Total Population': '{:,.0f}',
            'Avg Travel Time': '{:.1f} min',
            'Avg Transfers': '{:.2f}',
            'Zones': '{:.0f}',
            'Excellent (%)': '{:.1f}%',
            'Good (%)': '{:.1f}%',
            'Fair (%)': '{:.1f}%',
            'Poor (%)': '{:.1f}%'
        }

        if include_municipalities:
            format_dict['Municipalities'] = '{:.0f}'

        st.dataframe(
            display_df.style.format(format_dict)
            .background_gradient(subset=['Avg Travel Time'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    def render_download_section(self, analyzer: AccessibilityAnalyzer, poi_df: pd.DataFrame,
                                poor_zones: pd.DataFrame, df_all: pd.DataFrame):
        """Render download section in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📥 Download Report")

        # Prepare zone_need data if needed
        zone_need_for_report = None
        if 'Necesita_viaje' in df_all.columns:
            zone_need_for_report = self._calculate_zone_need_stats(df_all)

        # Generate complete report package
        report_package = self._create_complete_report_package(
            analyzer, poi_df, poor_zones, df_all, zone_need_for_report
        )

        st.sidebar.download_button(
            label="📦 Download Complete Report",
            data=report_package,
            file_name=f"bizkaia_accessibility_report_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            help="Includes Excel report + all chart images (PNG)"
        )

        st.sidebar.markdown("*ZIP includes:*")
        st.sidebar.markdown("- 📊 Excel report (3 sheets)")
        st.sidebar.markdown("- 📈 High-resolution charts (PNG)")

    def _create_complete_report_package(self, analyzer: AccessibilityAnalyzer, poi_df: pd.DataFrame,
                                        poor_zones: pd.DataFrame, df_all: pd.DataFrame,
                                        zone_need: Optional[pd.DataFrame]) -> BytesIO:
        """Create complete report package as ZIP"""
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add Excel report
            report_gen = ReportGenerator(analyzer)
            excel_report = report_gen.create_excel_report(poi_df, poor_zones, df_all, zone_need)
            zip_file.writestr(f"bizkaia_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                              excel_report.getvalue())

            # Add all chart images
            charts = report_gen.create_all_charts_images(poi_df, poor_zones)
            for filename, image_data in charts.items():
                zip_file.writestr(filename, image_data)

        zip_buffer.seek(0)
        return zip_buffer

    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_methodology()

        # File uploader
        uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])

        if uploaded_file is not None:
            # Process data
            df, df_all, analyzer = self.process_uploaded_file(uploaded_file)

            # Render sidebar
            analyzer = self.render_sidebar(df, analyzer, df_all)

            # Render key metrics
            self.render_key_metrics(analyzer)

            # Calculate data for tabs
            poi_df = analyzer.get_poi_accessibility()
            poor_zones = analyzer.get_priority_zones()

            # Create tabs
            tabs = st.tabs([
                "🎯 Accessibility Overview",
                "🔍 POI Analysis",
                "⚠️ Priority Areas",
                "📊 Distance Efficiency",
                "🔄 Trip need Patterns",
                "🗺️ Geographic Analysis"
            ])

            # Render tab content
            with tabs[0]:
                self.render_accessibility_overview_tab(analyzer)


            with tabs[1]:
                self.render_poi_analysis_tab(analyzer)

            with tabs[2]:
                self.render_priority_areas_tab(analyzer)

            with tabs[3]:
                self.render_distance_efficiency_tab(analyzer)

            with tabs[4]:
                self.render_trip_need_tab(analyzer, df_all)

            with tabs[5]:
                self.render_geographic_analysis_tab(analyzer)

            # Render download section
            self.render_download_section(analyzer, poi_df, poor_zones, df_all)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()