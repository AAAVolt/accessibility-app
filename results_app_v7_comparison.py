import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class AccessibilityConfig:
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

    # Scenario label colors (for overlaid traces)
    SCENARIO_COLORS = ['#2563EB', '#DC2626']  # blue=A, red=B
    DELTA_POS = '#16A34A'   # green  → improvement (time went down)
    DELTA_NEG = '#DC2626'   # red    → degradation (time went up)
    DELTA_NEU = '#6B7280'   # grey   → negligible


CONFIG = AccessibilityConfig()
CATEGORY_SHORT = ['Excellent', 'Good', 'Fair', 'Moderate', 'Poor', 'Very Poor']
CATEGORY_COLS  = [f'{c} (%)' for c in CATEGORY_SHORT]


# ============================================================================
# DATA PROCESSING
# ============================================================================

class AccessibilityAnalyzer:
    def __init__(self, df: pd.DataFrame, time_metric: str = 'JRT'):
        self.df = df.copy()
        self.time_metric = time_metric

        if time_metric == 'PJT' and 'Tiempo_Viaje_Percibido' in df.columns:
            self.df['Tiempo_Total_Minutos'] = self.df['Tiempo_Viaje_Percibido']
            self.active_metric_label = 'Perceived Journey Time (PJT)'
        else:
            self.df['Tiempo_Total_Minutos'] = self.df['Tiempo_Trayecto']
            self.active_metric_label = 'Journey Time (JRT)'

        self.df = self.df[self.df['Tiempo_Total_Minutos'] <= 999]
        self.df_original = df.copy()
        self.df = self.df[self.df['Necesita_Viaje'] == 1]

        self.zone_populations = self._calculate_zone_populations()
        self.total_population = self.zone_populations.sum()

    def _calculate_zone_populations(self) -> pd.Series:
        return self.df.groupby('Zona_Origen')['Poblacion'].first()

    def categorize_accessibility(self, travel_times: pd.Series) -> pd.Series:
        return pd.cut(
            travel_times,
            bins=[0, CONFIG.EXCELLENT_THRESHOLD, CONFIG.GOOD_THRESHOLD,
                  CONFIG.FAIR_THRESHOLD, CONFIG.MODERATE_THRESHOLD,
                  CONFIG.POOR_THRESHOLD, float('inf')],
            labels=list(CONFIG.CATEGORY_LABELS.values())
        )

    def get_zone_best_access(self) -> pd.DataFrame:
        zone_best = self.df.groupby('Zona_Origen').agg({
            'Tiempo_Total_Minutos': 'min',
            'Poblacion': 'first',
            'Nombre_Origen': 'first'
        }).reset_index()
        zone_best['Best_Category'] = self.categorize_accessibility(zone_best['Tiempo_Total_Minutos'])
        return zone_best

    def get_population_by_category(self) -> Tuple[pd.Series, pd.Series]:
        zone_best = self.get_zone_best_access()
        pop_by_category = zone_best.groupby('Best_Category')['Poblacion'].sum()
        category_pcts = (pop_by_category / self.total_population * 100) if self.total_population > 0 else pd.Series()
        return pop_by_category, category_pcts

    def get_average_travel_times(self) -> Dict:
        zone_best = self.get_zone_best_access()
        avg_time_per_zone = zone_best['Tiempo_Total_Minutos'].mean()
        total_population = zone_best['Poblacion'].sum()
        weighted_avg_time = (
            (zone_best['Tiempo_Total_Minutos'] * zone_best['Poblacion']).sum() / total_population
            if total_population > 0 else 0
        )
        return {'avg_per_zone': avg_time_per_zone, 'weighted_avg_per_pop': weighted_avg_time}

    def get_geographic_analysis(self, geo_column: str) -> pd.DataFrame:
        zone_best = self.get_zone_best_access()
        zone_geo = self.df[['Zona_Origen', geo_column, 'Num_Transbordos']].drop_duplicates()
        data = zone_best.merge(zone_geo, on='Zona_Origen')

        geo_analysis = data.groupby(geo_column).apply(
            lambda g: pd.Series({
                'Total Population': g['Poblacion'].sum(),
                'Avg Travel Time': np.average(g['Tiempo_Total_Minutos'], weights=g['Poblacion']),
                'Avg Transfers': np.average(g['Num_Transbordos'], weights=g['Poblacion']),
                'Zones': len(g)
            })
        ).reset_index()
        geo_analysis.columns = ['Geographic Unit', 'Total Population', 'Avg Travel Time', 'Avg Transfers', 'Zones']

        for category, (min_t, max_t) in [
            ('Excellent', (0, CONFIG.EXCELLENT_THRESHOLD)),
            ('Good', (CONFIG.EXCELLENT_THRESHOLD, CONFIG.GOOD_THRESHOLD)),
            ('Fair', (CONFIG.GOOD_THRESHOLD, CONFIG.FAIR_THRESHOLD)),
            ('Moderate', (CONFIG.FAIR_THRESHOLD, CONFIG.MODERATE_THRESHOLD)),
            ('Poor', (CONFIG.MODERATE_THRESHOLD, CONFIG.POOR_THRESHOLD)),
            ('Very Poor', (CONFIG.POOR_THRESHOLD, float('inf')))
        ]:
            pct_list = []
            for geo_unit in geo_analysis['Geographic Unit']:
                unit_data = data[data[geo_column] == geo_unit]
                if max_t == float('inf'):
                    cat_zones = unit_data[unit_data['Tiempo_Total_Minutos'] > min_t]
                elif min_t == 0:
                    cat_zones = unit_data[unit_data['Tiempo_Total_Minutos'] <= max_t]
                else:
                    cat_zones = unit_data[
                        (unit_data['Tiempo_Total_Minutos'] > min_t) &
                        (unit_data['Tiempo_Total_Minutos'] <= max_t)
                    ]
                total_pop = unit_data['Poblacion'].sum()
                pct_list.append((cat_zones['Poblacion'].sum() / total_pop * 100) if total_pop > 0 else 0)
            geo_analysis[f'{category} (%)'] = pct_list

        return geo_analysis.sort_values('Total Population', ascending=False)


# ============================================================================
# CHART HELPERS
# ============================================================================

class ChartStyler:
    @staticmethod
    def apply_standard_styling(fig, title: str, x_title: str, y_title: str,
                                height: int = 700, width: int = 1200):
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(title=dict(text=x_title, font=dict(size=16)), tickfont=dict(size=14)),
            yaxis=dict(title=dict(text=y_title, font=dict(size=16)), tickfont=dict(size=14)),
            legend=dict(font=dict(size=14)),
            height=height, width=width,
            margin=dict(l=200, r=100, t=100, b=80),
            font=dict(size=14),
        )
        fig.update_traces(textfont=dict(size=13), hoverlabel=dict(font=dict(size=14)))
        return fig


STYLER = ChartStyler()


# ============================================================================
# DATA LOADING
# ============================================================================

REQUIRED_COLUMNS = [
    'Zona_Origen', 'Nombre_Origen', 'Poblacion', 'Municipio', 'Comarca',
    'Destino_Óptimo', 'Nombre_Destino', 'Necesita_Viaje', 'Tiempo_Trayecto'
]


def load_file(uploaded_file, time_metric: str = 'JRT') -> Optional[AccessibilityAnalyzer]:
    try:
        df = pd.read_excel(uploaded_file)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns in **{uploaded_file.name}**: `{missing}`")
            return None
        return AccessibilityAnalyzer(df, time_metric=time_metric)
    except Exception as e:
        st.error(f"Error reading **{uploaded_file.name}**: {e}")
        return None


def metric_selector(df_preview: pd.DataFrame, key: str) -> str:
    has_pjt = 'Tiempo_Viaje_Percibido' in df_preview.columns
    if has_pjt:
        choice = st.sidebar.radio(
            f"Travel time metric ({key})",
            ["JRT – Journey Time", "PJT – Perceived Journey Time"],
            key=f"metric_{key}"
        )
        return "PJT" if choice.startswith("PJT") else "JRT"
    return "JRT"


# ============================================================================
# COMPARISON RENDERING UTILITIES
# ============================================================================

def delta_color(val: float, better_direction: str = 'lower') -> str:
    """Return CSS color string for a delta value."""
    if abs(val) < 0.5:
        return CONFIG.DELTA_NEU
    if better_direction == 'lower':
        return CONFIG.DELTA_POS if val < 0 else CONFIG.DELTA_NEG
    else:
        return CONFIG.DELTA_POS if val > 0 else CONFIG.DELTA_NEG


def render_kpi_row(label_a: str, label_b: str,
                   az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer):
    """Top-level KPI strip comparing two scenarios."""
    a_times = az.get_average_travel_times()
    b_times = bz.get_average_travel_times()

    cols = st.columns(6)
    kpis = [
        ("Total Population", f"{az.total_population:,.0f}", f"{bz.total_population:,.0f}",
         bz.total_population - az.total_population, 'higher'),
        ("Total Zones", f"{len(az.zone_populations):,}", f"{len(bz.zone_populations):,}",
         len(bz.zone_populations) - len(az.zone_populations), 'higher'),
        ("Avg Time / Zone", f"{a_times['avg_per_zone']:.1f} min", f"{b_times['avg_per_zone']:.1f} min",
         b_times['avg_per_zone'] - a_times['avg_per_zone'], 'lower'),
        ("Pop-Weighted Avg", f"{a_times['weighted_avg_per_pop']:.1f} min", f"{b_times['weighted_avg_per_pop']:.1f} min",
         b_times['weighted_avg_per_pop'] - a_times['weighted_avg_per_pop'], 'lower'),
    ]
    for i, (lbl, va, vb, delta, direction) in enumerate(kpis):
        with cols[i]:
            color = delta_color(delta, direction)
            sign = "+" if delta > 0 else ""
            st.markdown(f"""
<div style="background:#F8FAFC;border-radius:8px;padding:12px 10px;text-align:center;border:1px solid #E2E8F0">
  <div style="font-size:12px;color:#64748B;margin-bottom:4px">{lbl}</div>
  <div style="font-size:14px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[0]}">{label_a}: {va}</div>
  <div style="font-size:14px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[1]}">{label_b}: {vb}</div>
  <div style="font-size:13px;font-weight:700;color:{color};margin-top:4px">Δ {sign}{delta:,.1f}</div>
</div>""", unsafe_allow_html=True)

    # % excellent+good side-by-side
    _, a_pcts = az.get_population_by_category()
    _, b_pcts = bz.get_population_by_category()
    a_good = a_pcts.get(CONFIG.CATEGORY_LABELS['excellent'], 0) + a_pcts.get(CONFIG.CATEGORY_LABELS['good'], 0)
    b_good = b_pcts.get(CONFIG.CATEGORY_LABELS['excellent'], 0) + b_pcts.get(CONFIG.CATEGORY_LABELS['good'], 0)
    delta_good = b_good - a_good

    with cols[4]:
        color = delta_color(delta_good, 'higher')
        sign = "+" if delta_good > 0 else ""
        st.markdown(f"""
<div style="background:#F8FAFC;border-radius:8px;padding:12px 10px;text-align:center;border:1px solid #E2E8F0">
  <div style="font-size:12px;color:#64748B;margin-bottom:4px">% Pop ≤45 min</div>
  <div style="font-size:14px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[0]}">{label_a}: {a_good:.1f}%</div>
  <div style="font-size:14px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[1]}">{label_b}: {b_good:.1f}%</div>
  <div style="font-size:13px;font-weight:700;color:{color};margin-top:4px">Δ {sign}{delta_good:.1f}pp</div>
</div>""", unsafe_allow_html=True)

    a_poor = a_pcts.get(CONFIG.CATEGORY_LABELS['poor'], 0) + a_pcts.get(CONFIG.CATEGORY_LABELS['very_poor'], 0)
    b_poor = b_pcts.get(CONFIG.CATEGORY_LABELS['poor'], 0) + b_pcts.get(CONFIG.CATEGORY_LABELS['very_poor'], 0)
    delta_poor = b_poor - a_poor

    with cols[5]:
        color = delta_color(delta_poor, 'lower')
        sign = "+" if delta_poor > 0 else ""
        st.markdown(f"""
<div style="background:#F8FAFC;border-radius:8px;padding:12px 10px;text-align:center;border:1px solid #E2E8F0">
  <div style="font-size:12px;color:#64748B;margin-bottom:4px">% Pop >75 min</div>
  <div style="font-size:14px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[0]}">{label_a}: {a_poor:.1f}%</div>
  <div style="font-size:14px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[1]}">{label_b}: {b_poor:.1f}%</div>
  <div style="font-size:13px;font-weight:700;color:{color};margin-top:4px">Δ {sign}{delta_poor:.1f}pp</div>
</div>""", unsafe_allow_html=True)

    st.markdown("")


# ============================================================================
# TAB 1 — POPULATION ACCESSIBILITY COMPARISON
# ============================================================================

def render_population_accessibility(
        az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer,
        label_a: str, label_b: str):

    st.markdown("## 📊 Population Accessibility Comparison")

    a_pop, a_pcts = az.get_population_by_category()
    b_pop, b_pcts = bz.get_population_by_category()

    # --- Grouped bar: population count ---
    st.markdown("### Population by Accessibility Category")
    _render_grouped_category_bar(a_pop, b_pop, label_a, label_b,
                                 y_title="Population", title="Population Count by Accessibility Category")

    st.markdown("---")

    # --- Grouped bar: percentage ---
    st.markdown("### Share of Population by Accessibility Category (%)")
    _render_grouped_category_bar(a_pcts, b_pcts, label_a, label_b,
                                 y_title="% of Total Population",
                                 title="Population Share by Accessibility Category")

    st.markdown("---")

    # --- Delta bar ---
    st.markdown("### Delta: Change in Population Share (B − A, percentage points)")
    _render_delta_category_bar(a_pcts, b_pcts, label_a, label_b)

    st.markdown("---")

    # --- Side-by-side pie ---
    st.markdown("### Breakdown Pie Charts")
    col1, col2 = st.columns(2)
    with col1:
        _render_pie(a_pop, label_a)
    with col2:
        _render_pie(b_pop, label_b)


def _render_grouped_category_bar(a_series: pd.Series, b_series: pd.Series,
                                  label_a: str, label_b: str,
                                  y_title: str, title: str):
    all_cats = list(CONFIG.CATEGORY_LABELS.values())
    a_vals = [a_series.get(c, 0) for c in all_cats]
    b_vals = [b_series.get(c, 0) for c in all_cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=label_a, x=all_cats, y=a_vals,
        marker_color=CONFIG.SCENARIO_COLORS[0],
        text=[f"{v:,.0f}" if y_title == "Population" else f"{v:.1f}%" for v in a_vals],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name=label_b, x=all_cats, y=b_vals,
        marker_color=CONFIG.SCENARIO_COLORS[1],
        text=[f"{v:,.0f}" if y_title == "Population" else f"{v:.1f}%" for v in b_vals],
        textposition='outside'
    ))
    fig = STYLER.apply_standard_styling(fig, title, "Accessibility Category", y_title, height=500)
    fig.update_layout(barmode='group',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)


def _render_delta_category_bar(a_pcts: pd.Series, b_pcts: pd.Series,
                                label_a: str, label_b: str):
    all_cats = list(CONFIG.CATEGORY_LABELS.values())
    deltas = [b_pcts.get(c, 0) - a_pcts.get(c, 0) for c in all_cats]
    colors = [CONFIG.DELTA_POS if d < 0 else (CONFIG.DELTA_NEG if d > 0 else CONFIG.DELTA_NEU)
              for d in deltas]
    # For "better" categories (excellent/good) flip the color logic
    better_idx = [0, 1]
    for i in better_idx:
        if abs(deltas[i]) >= 0.5:
            colors[i] = CONFIG.DELTA_POS if deltas[i] > 0 else CONFIG.DELTA_NEG

    fig = go.Figure(go.Bar(
        x=all_cats,
        y=deltas,
        marker_color=colors,
        text=[f"{'+' if d > 0 else ''}{d:.1f}pp" for d in deltas],
        textposition='outside'
    ))
    fig = STYLER.apply_standard_styling(
        fig, f"Population Share Change ({label_b} vs {label_a})",
        "Accessibility Category", "Δ Percentage Points", height=450)
    fig.add_hline(y=0, line_color='#94A3B8', line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)

    # Summary note
    improved_cats = [all_cats[i] for i in better_idx if deltas[i] > 0.5]
    degraded_cats  = [all_cats[i] for i in better_idx if deltas[i] < -0.5]
    if improved_cats:
        st.success(f"✅ Scenario B improves population share in: **{', '.join(improved_cats)}**")
    if degraded_cats:
        st.warning(f"⚠️ Scenario B reduces population share in: **{', '.join(degraded_cats)}**")


def _render_pie(pop_series: pd.Series, label: str):
    colors = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
              CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]
    fig = go.Figure(go.Pie(
        labels=pop_series.index,
        values=pop_series.values,
        hole=.35,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=colors[:len(pop_series)])
    ))
    fig.update_layout(title=dict(text=label, font=dict(size=16)), height=450,
                      margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2 — TRAVEL TIME DISTRIBUTIONS COMPARISON
# ============================================================================

def render_travel_time_distributions(
        az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer,
        label_a: str, label_b: str):

    st.markdown(f"## ⏱️ Travel Time Distribution Comparison")

    st.markdown("### Population-Weighted Distribution Overlay")
    fig = _create_overlay_distribution(az, bz, label_a, label_b, weighted=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Zone-Level Distribution Overlay")
    fig2 = _create_overlay_distribution(az, bz, label_a, label_b, weighted=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown("### Cumulative Distribution (Population-Weighted)")
    fig3 = _create_cumulative_distribution(az, bz, label_a, label_b)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.markdown("### Summary Statistics")
    _render_distribution_stats_table(az, bz, label_a, label_b)


def _weighted_histogram(analyzer: AccessibilityAnalyzer, bins: np.ndarray) -> np.ndarray:
    df = analyzer.df
    zone_populations = analyzer.zone_populations
    total_population = analyzer.total_population
    bin_pops = []
    for i in range(len(bins) - 1):
        mask = (df['Tiempo_Total_Minutos'] >= bins[i]) & (df['Tiempo_Total_Minutos'] < bins[i + 1])
        zones_in_bin = df[mask]['Zona_Origen'].unique()
        pop_in_bin = zone_populations[zone_populations.index.isin(zones_in_bin)].sum()
        bin_pops.append(pop_in_bin / total_population * 100 if total_population > 0 else 0)
    return np.array(bin_pops)


def _weighted_kde(analyzer: AccessibilityAnalyzer) -> Tuple[np.ndarray, np.ndarray]:
    df = analyzer.df
    travel_times_list = []
    for _, row in df.iterrows():
        travel_times_list.extend([row['Tiempo_Total_Minutos']] * max(1, int(row['Poblacion'] / 100)))
    travel_times_weighted = np.array(travel_times_list)
    kde = stats.gaussian_kde(travel_times_weighted)
    x_range = np.linspace(df['Tiempo_Total_Minutos'].min(), df['Tiempo_Total_Minutos'].max(), 300)
    return x_range, kde(x_range)


def _create_overlay_distribution(az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer,
                                  label_a: str, label_b: str, weighted: bool) -> go.Figure:
    fig = go.Figure()

    for analyzer, label, color, opacity in [
        (az, label_a, CONFIG.SCENARIO_COLORS[0], 0.35),
        (bz, label_b, CONFIG.SCENARIO_COLORS[1], 0.35)
    ]:
        if weighted:
            all_times = analyzer.df['Tiempo_Total_Minutos']
        else:
            all_times = analyzer.get_zone_best_access()['Tiempo_Total_Minutos']

        bins = np.linspace(all_times.min(), all_times.max(), 41)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if weighted:
            y_vals = _weighted_histogram(analyzer, bins)
            y_label = "% Population"
        else:
            counts, _ = np.histogram(all_times, bins=bins)
            y_vals = counts / len(all_times) * 100
            y_label = "% Zones"

        fig.add_trace(go.Bar(
            x=bin_centers, y=y_vals,
            name=f"{label} (histogram)",
            marker_color=color, opacity=opacity,
            width=(bins[1] - bins[0]) * 0.9,
            showlegend=True
        ))

        # KDE overlay
        if weighted:
            x_kde, y_kde = _weighted_kde(analyzer)
            y_kde_scaled = y_kde * max(y_vals) / y_kde.max() * 0.85 if y_kde.max() > 0 else y_kde
        else:
            zone_times = analyzer.get_zone_best_access()['Tiempo_Total_Minutos'].dropna()
            kde = stats.gaussian_kde(zone_times)
            x_kde = np.linspace(zone_times.min(), zone_times.max(), 300)
            y_kde_raw = kde(x_kde)
            y_kde_scaled = y_kde_raw * max(y_vals) / y_kde_raw.max() * 0.85 if y_kde_raw.max() > 0 else y_kde_raw

        fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde_scaled,
            mode='lines', name=f"{label} (KDE)",
            line=dict(color=color, width=2.5),
            showlegend=True
        ))

    # Reference vlines
    for thresh, dash, col_ref in [
        (30, 'dash', 'green'), (45, 'dash', 'orange'),
        (60, 'dot', 'red'), (75, 'dot', 'darkred'), (90, 'dot', 'maroon')
    ]:
        fig.add_vline(x=thresh, line_dash=dash, line_color=col_ref,
                      annotation_text=f"{thresh}m", annotation_position="top",
                      annotation_font_size=11)

    title = (f"Population-Weighted Travel Time Distribution — {az.active_metric_label}"
             if weighted else f"Zone-Level Travel Time Distribution — {az.active_metric_label}")
    y_label = "% Population" if weighted else "% Zones"
    fig = STYLER.apply_standard_styling(fig, title, "Travel Time (minutes)", y_label, height=550)
    fig.update_layout(
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def _create_cumulative_distribution(az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer,
                                     label_a: str, label_b: str) -> go.Figure:
    fig = go.Figure()
    x_max = max(az.df['Tiempo_Total_Minutos'].max(), bz.df['Tiempo_Total_Minutos'].max())
    x_range = np.linspace(0, min(x_max, 200), 400)

    for analyzer, label, color in [
        (az, label_a, CONFIG.SCENARIO_COLORS[0]),
        (bz, label_b, CONFIG.SCENARIO_COLORS[1])
    ]:
        zone_best = analyzer.get_zone_best_access()
        cum_pop = []
        for t in x_range:
            pop_under = zone_best[zone_best['Tiempo_Total_Minutos'] <= t]['Poblacion'].sum()
            cum_pop.append(pop_under / analyzer.total_population * 100 if analyzer.total_population > 0 else 0)
        fig.add_trace(go.Scatter(
            x=x_range, y=cum_pop,
            mode='lines', name=label,
            line=dict(color=color, width=2.5)
        ))

    for thresh, col_ref in [(30, 'green'), (45, 'orange'), (60, 'red'), (75, 'darkred')]:
        fig.add_vline(x=thresh, line_dash='dash', line_color=col_ref,
                      annotation_text=f"{thresh}m", annotation_position="top right",
                      annotation_font_size=11)
    fig.add_hline(y=50, line_dash='dot', line_color='#94A3B8',
                  annotation_text="50%", annotation_position="right")

    fig = STYLER.apply_standard_styling(
        fig, "Cumulative Population Distribution by Travel Time",
        "Travel Time (minutes)", "Cumulative % Population", height=520)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig


def _render_distribution_stats_table(az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer,
                                      label_a: str, label_b: str):
    def stats_for(analyzer: AccessibilityAnalyzer) -> Dict:
        zone_best = analyzer.get_zone_best_access()
        t = zone_best['Tiempo_Total_Minutos']
        w = zone_best['Poblacion']
        wavg = np.average(t, weights=w)
        return {
            'Min': t.min(),
            'P10': np.percentile(t, 10),
            'Median (zones)': np.median(t),
            'P90': np.percentile(t, 90),
            'Max': t.max(),
            'Pop-Weighted Mean': wavg,
            'Std Dev': t.std(),
        }

    sa = stats_for(az)
    sb = stats_for(bz)

    rows = []
    for k in sa:
        delta = sb[k] - sa[k]
        sign = "+" if delta > 0 else ""
        rows.append({
            'Metric': k,
            label_a: f"{sa[k]:.1f} min",
            label_b: f"{sb[k]:.1f} min",
            'Delta (B−A)': f"{sign}{delta:.1f} min",
            '_delta_raw': delta,
            '_better': 'lower'
        })

    df_stats = pd.DataFrame(rows)

    def highlight_delta(row):
        val = row['_delta_raw']
        color = delta_color(val, row['_better'])
        return [''] * (len(row) - 2) + [f'color: {color}; font-weight: bold'] + ['']

    display_cols = ['Metric', label_a, label_b, 'Delta (B−A)']
    st.dataframe(
        df_stats[display_cols + ['_delta_raw', '_better']].style.apply(highlight_delta, axis=1)
            .hide(axis='columns', subset=['_delta_raw', '_better']),
        use_container_width=True, hide_index=True
    )


# ============================================================================
# TAB 3 — GEOGRAPHIC ANALYSIS COMPARISON
# ============================================================================

def render_geographic_analysis(
        az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer,
        label_a: str, label_b: str):

    st.markdown("## 🗺️ Geographic Analysis Comparison")

    for geo_col, unit_label in [('Municipio', 'Municipality'), ('Comarca', 'Comarca')]:
        st.markdown(f"### {unit_label} Level")

        geo_a = az.get_geographic_analysis(geo_col)
        geo_b = bz.get_geographic_analysis(geo_col)

        subtab1, subtab2, subtab3 = st.tabs([
            f"📊 Side-by-Side ({unit_label})",
            f"📉 Delta Chart ({unit_label})",
            f"📋 Delta Table ({unit_label})"
        ])

        with subtab1:
            _render_geo_side_by_side(geo_a, geo_b, label_a, label_b, unit_label)

        with subtab2:
            _render_geo_delta_chart(geo_a, geo_b, label_a, label_b, unit_label)

        with subtab3:
            _render_geo_delta_table(geo_a, geo_b, label_a, label_b)

        st.markdown("---")


def _render_geo_side_by_side(geo_a: pd.DataFrame, geo_b: pd.DataFrame,
                              label_a: str, label_b: str, unit_type: str):
    """Grouped stacked bar for each scenario (top 10 by population)."""
    col1, col2 = st.columns(2)

    for col, geo, label, color_base in [
        (col1, geo_a, label_a, 0),
        (col2, geo_b, label_b, 0)
    ]:
        with col:
            chart_data = geo.head(10)
            fig = go.Figure()
            cats = CATEGORY_COLS
            cat_colors = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                          CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]
            labels = ['🟢 Excellent', '🟡 Good', '🟠 Fair', '🔶 Moderate', '🔴 Poor', '⚫ Very Poor']

            for cat, clr, lbl in zip(cats, cat_colors, labels):
                if cat in chart_data.columns:
                    fig.add_trace(go.Bar(
                        name=lbl, y=chart_data['Geographic Unit'], x=chart_data[cat],
                        orientation='h', marker_color=clr,
                        text=chart_data[cat].round(0),
                        textposition='inside', texttemplate='%{text:.0f}%'
                    ))

            fig = STYLER.apply_standard_styling(
                fig, f"{label} – {unit_type} Accessibility",
                '% Population', unit_type, height=max(500, len(chart_data) * 35))
            fig.update_layout(
                barmode='stack',
                yaxis=dict(categoryorder='total ascending'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                showlegend=(col == col2)
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_geo_delta_chart(geo_a: pd.DataFrame, geo_b: pd.DataFrame,
                             label_a: str, label_b: str, unit_type: str):
    """Avg Travel Time delta per geographic unit, horizontal bar."""
    merged = geo_a[['Geographic Unit', 'Avg Travel Time', 'Total Population']].merge(
        geo_b[['Geographic Unit', 'Avg Travel Time']],
        on='Geographic Unit', suffixes=('_a', '_b')
    )
    merged['Delta'] = merged['Avg Travel Time_b'] - merged['Avg Travel Time_a']
    merged = merged.sort_values('Delta')

    colors = [delta_color(d, 'lower') for d in merged['Delta']]
    texts = [f"{'+'if d>0 else ''}{d:.1f} min" for d in merged['Delta']]

    fig = go.Figure(go.Bar(
        x=merged['Delta'],
        y=merged['Geographic Unit'],
        orientation='h',
        marker_color=colors,
        text=texts,
        textposition='outside',
        customdata=merged[['Avg Travel Time_a', 'Avg Travel Time_b', 'Total Population']].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{label_a}: %{{customdata[0]:.1f}} min<br>"
            f"{label_b}: %{{customdata[1]:.1f}} min<br>"
            "Delta: %{x:.1f} min<br>"
            "Population: %{customdata[2]:,.0f}<extra></extra>"
        )
    ))

    fig = STYLER.apply_standard_styling(
        fig, f"Avg Travel Time Change ({label_b} − {label_a}) by {unit_type}",
        "Δ Minutes", unit_type,
        height=max(500, len(merged) * 30)
    )
    fig.add_vline(x=0, line_color='#94A3B8', line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)

    # Excellent% delta chart
    if 'Excellent (%)' in geo_a.columns and 'Excellent (%)' in geo_b.columns:
        merged2 = geo_a[['Geographic Unit', 'Excellent (%)']].merge(
            geo_b[['Geographic Unit', 'Excellent (%)']],
            on='Geographic Unit', suffixes=('_a', '_b')
        )
        merged2['Delta_Exc'] = merged2['Excellent (%)_b'] - merged2['Excellent (%)_a']
        merged2 = merged2.sort_values('Delta_Exc')
        colors2 = [CONFIG.DELTA_POS if d > 0.5 else (CONFIG.DELTA_NEG if d < -0.5 else CONFIG.DELTA_NEU)
                   for d in merged2['Delta_Exc']]

        fig2 = go.Figure(go.Bar(
            x=merged2['Delta_Exc'],
            y=merged2['Geographic Unit'],
            orientation='h',
            marker_color=colors2,
            text=[f"{'+'if d>0 else ''}{d:.1f}pp" for d in merged2['Delta_Exc']],
            textposition='outside'
        ))
        fig2 = STYLER.apply_standard_styling(
            fig2, f"Excellent Access Change ({label_b} − {label_a}) by {unit_type}",
            "Δ Percentage Points", unit_type,
            height=max(500, len(merged2) * 30)
        )
        fig2.add_vline(x=0, line_color='#94A3B8', line_width=1.5)
        st.plotly_chart(fig2, use_container_width=True)


def _render_geo_delta_table(geo_a: pd.DataFrame, geo_b: pd.DataFrame,
                             label_a: str, label_b: str):
    """Full delta table merging both geographies."""
    key_cols = ['Geographic Unit', 'Total Population', 'Avg Travel Time'] + CATEGORY_COLS
    avail_a = [c for c in key_cols if c in geo_a.columns]
    avail_b = [c for c in key_cols if c in geo_b.columns]

    merged = geo_a[avail_a].merge(geo_b[avail_b], on='Geographic Unit', suffixes=('_A', '_B'))

    numeric_base = ['Total Population', 'Avg Travel Time'] + CATEGORY_COLS
    delta_rows = {}
    for col in numeric_base:
        ca, cb = f"{col}_A", f"{col}_B"
        if ca in merged.columns and cb in merged.columns:
            merged[f"Δ {col}"] = merged[cb] - merged[ca]

    # Format display
    format_dict = {'Total Population_A': '{:,.0f}', 'Total Population_B': '{:,.0f}'}
    for col in ['Avg Travel Time'] + CATEGORY_COLS:
        for suf in ['_A', '_B']:
            if f'{col}{suf}' in merged.columns:
                unit = ' min' if 'Travel' in col else '%'
                format_dict[f'{col}{suf}'] = '{:.1f}' + unit
        delta_col = f'Δ {col}'
        if delta_col in merged.columns:
            unit = ' min' if 'Travel' in col else 'pp'
            format_dict[delta_col] = '{:+.1f}' + unit

    # Rename for readability
    rename = {}
    for col in numeric_base:
        if f'{col}_A' in merged.columns:
            rename[f'{col}_A'] = f'{col} ({label_a})'
        if f'{col}_B' in merged.columns:
            rename[f'{col}_B'] = f'{col} ({label_b})'
    merged = merged.rename(columns=rename)

    # Update format_dict keys after rename
    new_fmt = {}
    for k, v in format_dict.items():
        new_k = rename.get(k, k)
        new_fmt[new_k] = v
    format_dict = new_fmt

    # Build display cols in order
    display_cols = ['Geographic Unit']
    for col in numeric_base:
        if f'{col} ({label_a})' in merged.columns:
            display_cols.append(f'{col} ({label_a})')
        if f'{col} ({label_b})' in merged.columns:
            display_cols.append(f'{col} ({label_b})')
        if f'Δ {col}' in merged.columns:
            display_cols.append(f'Δ {col}')

    available_display = [c for c in display_cols if c in merged.columns]

    def color_deltas(row):
        styles = []
        for c in row.index:
            if c.startswith('Δ'):
                try:
                    val = float(str(row[c]).replace('+', '').replace(' min', '').replace('pp', '').replace('%', ''))
                    better = 'lower' if 'Travel' in c else 'higher'
                    clr = delta_color(val, better)
                    styles.append(f'color: {clr}; font-weight: bold')
                except Exception:
                    styles.append('')
            else:
                styles.append('')
        return styles

    fmt_available = {k: v for k, v in format_dict.items() if k in available_display}
    st.dataframe(
        merged[available_display].sort_values(
            f'Total Population ({label_a})' if f'Total Population ({label_a})' in merged.columns
            else available_display[1], ascending=False
        ).style.apply(color_deltas, axis=1).format(fmt_available, na_rep='—'),
        use_container_width=True, hide_index=True, height=500
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class ComparisonApp:
    def __init__(self):
        st.set_page_config(
            page_title="Accessibility Comparison",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run(self):
        st.title("🚌 Transportation Accessibility — Scenario Comparison")
        st.markdown(
            "Upload two datasets (e.g. baseline vs. proposed scenario) to compare "
            "**Population Accessibility**, **Travel Time Distributions**, and **Geographic Analysis**."
        )
        st.markdown("---")

        # ── Sidebar ──────────────────────────────────────────────────────────
        st.sidebar.markdown("## 📁 Upload Scenarios")
        file_a = st.sidebar.file_uploader("Scenario A (baseline)", type=['xlsx', 'xls'], key="file_a")
        file_b = st.sidebar.file_uploader("Scenario B (comparison)", type=['xlsx', 'xls'], key="file_b")

        label_a = st.sidebar.text_input("Label for Scenario A", value="Scenario A")
        label_b = st.sidebar.text_input("Label for Scenario B", value="Scenario B")

        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ Metric Settings")

        if not file_a or not file_b:
            st.info("👈 Upload **both** Excel files in the sidebar to begin the comparison.")
            self._render_placeholder()
            return

        # Peek at columns to decide metric availability
        try:
            df_a_preview = pd.read_excel(file_a); file_a.seek(0)
            df_b_preview = pd.read_excel(file_b); file_b.seek(0)
        except Exception as e:
            st.error(f"Could not preview files: {e}")
            return

        metric_a = metric_selector(df_a_preview, label_a)
        metric_b = metric_selector(df_b_preview, label_b)

        st.sidebar.info(
            f"**{label_a}**: {'PJT' if metric_a == 'PJT' else 'JRT'}  \n"
            f"**{label_b}**: {'PJT' if metric_b == 'PJT' else 'JRT'}"
        )

        # ── Load analyzers ───────────────────────────────────────────────────
        with st.spinner("Loading and processing files…"):
            az = load_file(file_a, metric_a)
            bz = load_file(file_b, metric_b)

        if az is None or bz is None:
            return

        # ── KPI banner ───────────────────────────────────────────────────────
        st.markdown("### 📌 Key Metrics at a Glance")
        render_kpi_row(label_a, label_b, az, bz)
        st.markdown("---")

        # ── Tabs ─────────────────────────────────────────────────────────────
        tabs = st.tabs([
            "📊 Population Accessibility",
            "⏱️ Travel Time Distributions",
            "🗺️ Geographic Analysis",
        ])

        with tabs[0]:
            render_population_accessibility(az, bz, label_a, label_b)

        with tabs[1]:
            render_travel_time_distributions(az, bz, label_a, label_b)

        with tabs[2]:
            render_geographic_analysis(az, bz, label_a, label_b)

    @staticmethod
    def _render_placeholder():
        st.markdown("""
<div style="background:#F1F5F9;border-radius:12px;padding:40px;text-align:center;margin-top:40px">
  <div style="font-size:48px">📊</div>
  <div style="font-size:22px;font-weight:600;color:#334155;margin-top:12px">
    Scenario Comparison Ready
  </div>
  <div style="font-size:15px;color:#64748B;margin-top:8px">
    Upload Scenario A and Scenario B in the sidebar to start comparing<br>
    Population Accessibility · Travel Time Distributions · Geographic Analysis
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = ComparisonApp()
    app.run()