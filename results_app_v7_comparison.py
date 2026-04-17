import streamlit as st
import pandas as pd
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
        'good':      '🟡 Good (30-45min)',
        'fair':      '🟠 Fair (45-60min)',
        'moderate':  '🔶 Moderate (60-75min)',
        'poor':      '🔴 Poor (75-90min)',
        'very_poor': '⚫ Very Poor (>90min)'
    }

    SCENARIO_COLORS = ['#3a3a3a', '#A5CA70']
    DELTA_POS = '#16A34A'
    DELTA_NEG = '#D20B12'
    DELTA_NEU = '#6B7280'

    THRESHOLDS = [
        ('excellent', 0,                    30),
        ('good',      30,                   45),
        ('fair',      45,                   60),
        ('moderate',  60,                   75),
        ('poor',      75,                   90),
        ('very_poor', 90,                   float('inf')),
    ]


CONFIG = AccessibilityConfig()
CATEGORY_SHORT = ['Excellent', 'Good', 'Fair', 'Moderate', 'Poor', 'Very Poor']
CATEGORY_COLS  = [f'{c} (%)' for c in CATEGORY_SHORT]

REQUIRED_COLUMNS = [
    'Zona_Origen', 'Nombre_Origen', 'Poblacion', 'Municipio', 'Comarca',
    'Destino_Óptimo', 'Nombre_Destino', 'Necesita_Viaje', 'Tiempo_Trayecto',
    'Num_Transbordos',
]


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
            self.has_pjt = True
        else:
            self.df['Tiempo_Total_Minutos'] = self.df['Tiempo_Trayecto']
            self.active_metric_label = 'Journey Time (JRT)'
            self.has_pjt = False

        self.df = self.df[self.df['Tiempo_Total_Minutos'] <= 999]
        self.df_original = df.copy()
        self.df = self.df[self.df['Necesita_Viaje'] == 1]

        self.zone_populations = self._calculate_zone_populations()
        self.total_population  = self.zone_populations.sum()

    # ── helpers ──────────────────────────────────────────────────────────────

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

    # ── zone-level ───────────────────────────────────────────────────────────

    def get_zone_best_access(self) -> pd.DataFrame:
        """Best (minimum) travel time per zone across all POIs."""
        agg = self.df.groupby('Zona_Origen').agg(
            Tiempo_Total_Minutos=('Tiempo_Total_Minutos', 'min'),
            Poblacion=('Poblacion', 'first'),
            Nombre_Origen=('Nombre_Origen', 'first'),
        ).reset_index()
        agg['Best_Category'] = self.categorize_accessibility(agg['Tiempo_Total_Minutos'])
        return agg

    def get_zone_best_poi(self) -> pd.DataFrame:
        """Zone-level best travel time AND which POI delivers it."""
        idx = self.df.groupby('Zona_Origen')['Tiempo_Total_Minutos'].idxmin()
        best = self.df.loc[idx, ['Zona_Origen', 'Nombre_Origen', 'Poblacion',
                                  'Tiempo_Total_Minutos', 'Nombre_Destino',
                                  'Municipio', 'Comarca']].copy()
        best = best.rename(columns={
            'Tiempo_Total_Minutos': 'Best_Time',
            'Nombre_Destino':       'Best_POI',
        })
        best['Best_Category'] = self.categorize_accessibility(best['Best_Time'])
        return best.reset_index(drop=True)

    def get_all_poi_times(self) -> pd.DataFrame:
        """Full zone × POI travel-time matrix (used in drill-down)."""
        return (self.df
                .groupby(['Zona_Origen', 'Nombre_Destino'])['Tiempo_Total_Minutos']
                .mean()
                .reset_index()
                .rename(columns={'Tiempo_Total_Minutos': 'Travel_Time'}))

    # ── aggregate ────────────────────────────────────────────────────────────

    def get_population_by_category(self) -> Tuple[pd.Series, pd.Series]:
        zone_best = self.get_zone_best_access()
        pop_by_category = zone_best.groupby('Best_Category')['Poblacion'].sum()
        pcts = (pop_by_category / self.total_population * 100
                if self.total_population > 0 else pd.Series())
        return pop_by_category, pcts

    def get_average_travel_times(self) -> Dict:
        zone_best = self.get_zone_best_access()
        avg_zone = zone_best['Tiempo_Total_Minutos'].mean()
        tot_pop  = zone_best['Poblacion'].sum()
        w_avg    = ((zone_best['Tiempo_Total_Minutos'] * zone_best['Poblacion']).sum() / tot_pop
                    if tot_pop > 0 else 0)
        return {'avg_per_zone': avg_zone, 'weighted_avg_per_pop': w_avg}

    def get_geographic_analysis(self, geo_column: str) -> pd.DataFrame:
        zone_best = self.get_zone_best_access()
        zone_geo  = self.df[['Zona_Origen', geo_column, 'Num_Transbordos']].drop_duplicates()
        data = zone_best.merge(zone_geo, on='Zona_Origen')

        geo_analysis = data.groupby(geo_column).apply(
            lambda g: pd.Series({
                'Total Population': g['Poblacion'].sum(),
                'Avg Travel Time':  np.average(g['Tiempo_Total_Minutos'], weights=g['Poblacion']),
                'Avg Transfers':    np.average(g['Num_Transbordos'],       weights=g['Poblacion']),
                'Zones':            len(g),
            })
        ).reset_index()
        geo_analysis.columns = ['Geographic Unit', 'Total Population',
                                 'Avg Travel Time', 'Avg Transfers', 'Zones']

        for cat_short, (_, min_t, max_t) in zip(CATEGORY_SHORT, CONFIG.THRESHOLDS):
            pct_list = []
            for geo_unit in geo_analysis['Geographic Unit']:
                ud = data[data[geo_column] == geo_unit]
                if max_t == float('inf'):
                    cz = ud[ud['Tiempo_Total_Minutos'] > min_t]
                elif min_t == 0:
                    cz = ud[ud['Tiempo_Total_Minutos'] <= max_t]
                else:
                    cz = ud[(ud['Tiempo_Total_Minutos'] > min_t) &
                             (ud['Tiempo_Total_Minutos'] <= max_t)]
                tot = ud['Poblacion'].sum()
                pct_list.append(cz['Poblacion'].sum() / tot * 100 if tot > 0 else 0)
            geo_analysis[f'{cat_short} (%)'] = pct_list

        return geo_analysis.sort_values('Total Population', ascending=False)

    # ── inequality ───────────────────────────────────────────────────────────

    def get_inequality_metrics(self) -> Dict:
        zone_best = self.get_zone_best_access()
        t = zone_best['Tiempo_Total_Minutos'].values
        w = zone_best['Poblacion'].values
        w_norm = w / w.sum()

        # Population-weighted Gini
        sorted_idx  = np.argsort(t)
        t_sorted    = t[sorted_idx]
        w_sorted    = w_norm[sorted_idx]
        cum_w       = np.cumsum(w_sorted)
        cum_wt      = np.cumsum(w_sorted * t_sorted)
        gini        = 1 - 2 * np.trapezoid(cum_wt / cum_wt[-1], cum_w)

        w_mean = np.average(t, weights=w)
        p10  = np.percentile(t, 10)
        p90  = np.percentile(t, 90)
        ratio = p90 / p10 if p10 > 0 else float('nan')

        return {
            'gini':        gini,
            'p90_p10':     ratio,
            'p10':         p10,
            'p90':         p90,
            'weighted_mean': w_mean,
            'std':         float(np.std(t)),
        }


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


def delta_color(val: float, better_direction: str = 'lower') -> str:
    if abs(val) < 0.5:
        return CONFIG.DELTA_NEU
    if better_direction == 'lower':
        return CONFIG.DELTA_POS if val < 0 else CONFIG.DELTA_NEG
    return CONFIG.DELTA_POS if val > 0 else CONFIG.DELTA_NEG


# ============================================================================
# DATA LOADING
# ============================================================================

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
# KPI BANNER  (now 7 cards: +≤60 min)
# ============================================================================

def _kpi_card(col, label: str, val_a: str, val_b: str,
              delta: float, direction: str,
              label_a: str, label_b: str, unit: str = ''):
    color = delta_color(delta, direction)
    sign  = "+" if delta > 0 else ""
    col.markdown(f"""
<div style="background:#F8FAFC;border-radius:8px;padding:10px 8px;
            text-align:center;border:1px solid #E2E8F0;height:100%">
  <div style="font-size:11px;color:#64748B;margin-bottom:3px">{label}</div>
  <div style="font-size:13px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[0]}">{label_a}: {val_a}</div>
  <div style="font-size:13px;font-weight:600;color:{CONFIG.SCENARIO_COLORS[1]}">{label_b}: {val_b}</div>
  <div style="font-size:12px;font-weight:700;color:{color};margin-top:3px">
      Δ {sign}{delta:.1f}{unit}</div>
</div>""", unsafe_allow_html=True)


def render_kpi_row(label_a: str, label_b: str,
                   az: AccessibilityAnalyzer, bz: AccessibilityAnalyzer):
    a_times = az.get_average_travel_times()
    b_times = bz.get_average_travel_times()
    _, a_pcts = az.get_population_by_category()
    _, b_pcts = bz.get_population_by_category()

    def pct_under(pcts, *keys):
        return sum(pcts.get(CONFIG.CATEGORY_LABELS[k], 0) for k in keys)

    a_45  = pct_under(a_pcts, 'excellent', 'good')
    b_45  = pct_under(b_pcts, 'excellent', 'good')
    a_60  = pct_under(a_pcts, 'excellent', 'good', 'fair')
    b_60  = pct_under(b_pcts, 'excellent', 'good', 'fair')
    a_poor = pct_under(a_pcts, 'poor', 'very_poor')
    b_poor = pct_under(b_pcts, 'poor', 'very_poor')

    cols = st.columns(7)
    specs = [
        ("Total Population",     f"{az.total_population:,.0f}",             f"{bz.total_population:,.0f}",
         bz.total_population - az.total_population, 'higher', ''),
        ("Total Zones",          f"{len(az.zone_populations):,}",           f"{len(bz.zone_populations):,}",
         len(bz.zone_populations) - len(az.zone_populations), 'higher', ''),
        ("Avg Time / Zone",      f"{a_times['avg_per_zone']:.1f} min",      f"{b_times['avg_per_zone']:.1f} min",
         b_times['avg_per_zone'] - a_times['avg_per_zone'], 'lower', ' min'),
        ("Pop-Weighted Avg",     f"{a_times['weighted_avg_per_pop']:.1f} min", f"{b_times['weighted_avg_per_pop']:.1f} min",
         b_times['weighted_avg_per_pop'] - a_times['weighted_avg_per_pop'], 'lower', ' min'),
        ("% Pop ≤45 min",        f"{a_45:.1f}%",  f"{b_45:.1f}%",  b_45 - a_45,   'higher', 'pp'),
        ("% Pop ≤60 min",        f"{a_60:.1f}%",  f"{b_60:.1f}%",  b_60 - a_60,   'higher', 'pp'),
        ("% Pop >75 min",        f"{a_poor:.1f}%",f"{b_poor:.1f}%",b_poor - a_poor,'lower',  'pp'),
    ]
    for col, (lbl, va, vb, delta, direction, unit) in zip(cols, specs):
        _kpi_card(col, lbl, va, vb, delta, direction, label_a, label_b, unit)
    st.markdown("")


# ============================================================================
# TAB 1 — POPULATION ACCESSIBILITY
# ============================================================================

def render_population_accessibility(az, bz, label_a, label_b):
    st.markdown("## 📊 Population Accessibility Comparison")

    a_pop, a_pcts = az.get_population_by_category()
    b_pop, b_pcts = bz.get_population_by_category()

    st.markdown("### Population Count by Accessibility Category")
    _render_grouped_category_bar(a_pop, b_pop, label_a, label_b,
                                 "Population", "Population Count by Accessibility Category")
    st.markdown("---")

    st.markdown("### Share of Population (%)")
    _render_grouped_category_bar(a_pcts, b_pcts, label_a, label_b,
                                 "% of Total Population", "Population Share by Accessibility Category")
    st.markdown("---")

    st.markdown("### Delta: Change in Population Share (B − A)")
    _render_delta_category_bar(a_pcts, b_pcts, label_a, label_b)
    st.markdown("---")

    st.markdown("### Population Flow Between Categories (Sankey)")
    _render_category_sankey(az, bz, label_a, label_b)
    st.markdown("---")

    st.markdown("### Breakdown Pie Charts")
    c1, c2 = st.columns(2)
    with c1: _render_pie(a_pop, label_a)
    with c2: _render_pie(b_pop, label_b)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _render_grouped_category_bar(a_series, b_series, label_a, label_b, y_title, title):
    cat_keys  = list(CONFIG.COLORS.keys())                  # ordered: excellent … very_poor
    all_cats  = [CONFIG.CATEGORY_LABELS[k] for k in cat_keys]
    a_vals    = [float(a_series.get(c, 0)) for c in all_cats]
    b_vals    = [float(b_series.get(c, 0)) for c in all_cats]

    # Per-bar fill = category color @ 45 % opacity; border = scenario color
    fill_colors = [_hex_to_rgba(CONFIG.COLORS[k], 0.45) for k in cat_keys]

    fig = go.Figure()
    for vals, label, border_color in [
        (a_vals, label_a, CONFIG.SCENARIO_COLORS[0]),
        (b_vals, label_b, CONFIG.SCENARIO_COLORS[1]),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=all_cats,
            y=vals,
            marker=dict(
                color=fill_colors,
                line=dict(color=border_color, width=2.5),
            ),
            text=[f"{v:,.0f}" if "Population" == y_title else f"{v:.1f}%" for v in vals],
            textposition='outside',
        ))
    fig = STYLER.apply_standard_styling(fig, title, "Accessibility Category", y_title, height=500)
    fig.update_layout(barmode='group',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)


def _render_delta_category_bar(a_pcts, b_pcts, label_a, label_b):
    all_cats = list(CONFIG.CATEGORY_LABELS.values())
    deltas   = [float(b_pcts.get(c, 0)) - float(a_pcts.get(c, 0)) for c in all_cats]
    # better = more excellent/good, less of everything else
    better_mask = [True, True, False, False, False, False]
    colors = []
    for d, better in zip(deltas, better_mask):
        if abs(d) < 0.5:
            colors.append(CONFIG.DELTA_NEU)
        elif better:
            colors.append(CONFIG.DELTA_POS if d > 0 else CONFIG.DELTA_NEG)
        else:
            colors.append(CONFIG.DELTA_POS if d < 0 else CONFIG.DELTA_NEG)

    fig = go.Figure(go.Bar(
        x=all_cats, y=deltas, marker_color=colors,
        text=[f"{'+' if d > 0 else ''}{d:.1f}pp" for d in deltas],
        textposition='outside'
    ))
    fig = STYLER.apply_standard_styling(
        fig, f"Population Share Change ({label_b} vs {label_a})",
        "Accessibility Category", "Δ Percentage Points", height=450)
    fig.add_hline(y=0, line_color='#94A3B8', line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)

    better_cats  = [all_cats[i] for i, (d, b) in enumerate(zip(deltas, better_mask)) if b and d > 0.5]
    degraded_cats = [all_cats[i] for i, (d, b) in enumerate(zip(deltas, better_mask)) if b and d < -0.5]
    if better_cats:  st.success(f"✅ B improves share in: **{', '.join(better_cats)}**")
    if degraded_cats: st.warning(f"⚠️ B reduces share in: **{', '.join(degraded_cats)}**")


def _render_category_sankey(az, bz, label_a, label_b):
    """Show how population moves between accessibility categories A→B."""
    best_a = az.get_zone_best_poi()[['Zona_Origen', 'Best_Category', 'Poblacion']].copy()
    best_b = bz.get_zone_best_poi()[['Zona_Origen', 'Best_Category']].copy()
    merged = best_a.merge(best_b, on='Zona_Origen', suffixes=('_A', '_B'))

    cats  = list(CONFIG.CATEGORY_LABELS.values())
    n     = len(cats)
    label = [f"{label_a}: {c}" for c in cats] + [f"{label_b}: {c}" for c in cats]

    sources, targets, values, link_colors = [], [], [], []
    cat_colors_hex = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
                      CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]

    def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    for i, cat_a in enumerate(cats):
        for j, cat_b in enumerate(cats):
            pop = merged[(merged['Best_Category_A'] == cat_a) &
                         (merged['Best_Category_B'] == cat_b)]['Poblacion'].sum()
            if pop > 0:
                sources.append(i)
                targets.append(n + j)
                values.append(float(pop))
                link_colors.append(hex_to_rgba(cat_colors_hex[i], 0.5))

    if not values:
        st.info("No overlapping zones between scenarios for Sankey.")
        return

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=label,
            color=cat_colors_hex * 2,
            line=dict(color='white', width=0.5)
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    ))
    fig.update_layout(
        title=dict(text=f"Population Flow: {label_a} → {label_b}", font=dict(size=18)),
        height=520,
        font=dict(size=13)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each band shows how much population moved from one accessibility category (left) to another (right). "
               "Same-category bands = no change.")


def _render_pie(pop_series, label):
    colors = [CONFIG.COLORS['excellent'], CONFIG.COLORS['good'], CONFIG.COLORS['fair'],
              CONFIG.COLORS['moderate'], CONFIG.COLORS['poor'], CONFIG.COLORS['very_poor']]
    fig = go.Figure(go.Pie(
        labels=pop_series.index, values=pop_series.values,
        hole=.35, textinfo='label+percent', textposition='outside',
        marker=dict(colors=colors[:len(pop_series)])
    ))
    fig.update_layout(title=dict(text=label, font=dict(size=16)), height=450,
                      margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2 — TRAVEL TIME DISTRIBUTIONS
# ============================================================================

def render_travel_time_distributions(az, bz, label_a, label_b):
    st.markdown("## ⏱️ Travel Time Distribution Comparison")

    st.markdown("### Population-Weighted Distribution Overlay")
    st.plotly_chart(_create_overlay_distribution(az, bz, label_a, label_b, weighted=True),
                    use_container_width=True)
    st.markdown("---")

    st.markdown("### Zone-Level Distribution Overlay")
    st.plotly_chart(_create_overlay_distribution(az, bz, label_a, label_b, weighted=False),
                    use_container_width=True)
    st.markdown("---")

    st.markdown("### Cumulative Distribution (Population-Weighted)")
    st.plotly_chart(_create_cumulative_distribution(az, bz, label_a, label_b),
                    use_container_width=True)
    st.markdown("---")

    st.markdown("### 📐 Inequality Metrics")
    _render_inequality_table(az, bz, label_a, label_b)
    st.markdown("---")

    st.markdown("### Summary Statistics")
    _render_distribution_stats_table(az, bz, label_a, label_b)


def _weighted_histogram(analyzer, bins):
    df, zone_populations, total_population = (
        analyzer.df, analyzer.zone_populations, analyzer.total_population)
    out = []
    for i in range(len(bins) - 1):
        mask  = (df['Tiempo_Total_Minutos'] >= bins[i]) & (df['Tiempo_Total_Minutos'] < bins[i+1])
        zones = df[mask]['Zona_Origen'].unique()
        pop   = zone_populations[zone_populations.index.isin(zones)].sum()
        out.append(pop / total_population * 100 if total_population > 0 else 0)
    return np.array(out)


def _weighted_kde(analyzer):
    df = analyzer.df
    times = []
    for _, row in df.iterrows():
        times.extend([row['Tiempo_Total_Minutos']] * max(1, int(row['Poblacion'] / 100)))
    arr = np.array(times)
    kde = stats.gaussian_kde(arr)
    x   = np.linspace(arr.min(), arr.max(), 300)
    return x, kde(x)


def _create_overlay_distribution(az, bz, label_a, label_b, weighted: bool):
    fig = go.Figure()
    global_min = min(az.df['Tiempo_Total_Minutos'].min(), bz.df['Tiempo_Total_Minutos'].min())
    global_max = max(az.df['Tiempo_Total_Minutos'].max(), bz.df['Tiempo_Total_Minutos'].max())
    bins = np.linspace(global_min, global_max, 41)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for analyzer, label, color in [(az, label_a, CONFIG.SCENARIO_COLORS[0]),
                                    (bz, label_b, CONFIG.SCENARIO_COLORS[1])]:
        if weighted:
            y_vals = _weighted_histogram(analyzer, bins)
            y_lbl  = "% Population"
        else:
            zone_t = analyzer.get_zone_best_access()['Tiempo_Total_Minutos']
            counts, _ = np.histogram(zone_t, bins=bins)
            y_vals = counts / len(zone_t) * 100
            y_lbl  = "% Zones"

        fig.add_trace(go.Bar(
            x=bin_centers, y=y_vals, name=f"{label} (hist)",
            marker_color=color, opacity=0.35,
            width=(bins[1] - bins[0]) * 0.9
        ))

        # KDE
        if weighted:
            x_kde, y_kde_raw = _weighted_kde(analyzer)
        else:
            zone_t2 = analyzer.get_zone_best_access()['Tiempo_Total_Minutos'].dropna()
            kde     = stats.gaussian_kde(zone_t2)
            x_kde   = np.linspace(zone_t2.min(), zone_t2.max(), 300)
            y_kde_raw = kde(x_kde)
        y_kde_scaled = y_kde_raw * max(y_vals) / y_kde_raw.max() * 0.85 if y_kde_raw.max() > 0 else y_kde_raw
        fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde_scaled, mode='lines',
            name=f"{label} (KDE)", line=dict(color=color, width=2.5)
        ))

    for thresh, col_ref in [(30,'green'),(45,'orange'),(60,'red'),(75,'darkred'),(90,'maroon')]:
        fig.add_vline(x=thresh, line_dash='dash', line_color=col_ref,
                      annotation_text=f"{thresh}m", annotation_position="top",
                      annotation_font_size=11)

    title = ("Population-Weighted Travel Time Distribution" if weighted
             else "Zone-Level Travel Time Distribution")
    fig = STYLER.apply_standard_styling(fig, title, "Travel Time (min)", y_lbl, height=550)
    fig.update_layout(barmode='overlay',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig


def _create_cumulative_distribution(az, bz, label_a, label_b):
    fig    = go.Figure()
    x_max  = min(max(az.df['Tiempo_Total_Minutos'].max(), bz.df['Tiempo_Total_Minutos'].max()), 200)
    x_rng  = np.linspace(0, x_max, 400)

    for analyzer, label, color in [(az, label_a, CONFIG.SCENARIO_COLORS[0]),
                                    (bz, label_b, CONFIG.SCENARIO_COLORS[1])]:
        zone_best = analyzer.get_zone_best_access()
        cum = [(zone_best[zone_best['Tiempo_Total_Minutos'] <= t]['Poblacion'].sum()
                / analyzer.total_population * 100)
               for t in x_rng]
        fig.add_trace(go.Scatter(x=x_rng, y=cum, mode='lines',
                                  name=label, line=dict(color=color, width=2.5)))

    for thresh, col_ref in [(30,'green'),(45,'orange'),(60,'red'),(75,'darkred')]:
        fig.add_vline(x=thresh, line_dash='dash', line_color=col_ref,
                      annotation_text=f"{thresh}m", annotation_position="top right",
                      annotation_font_size=11)
    fig.add_hline(y=50, line_dash='dot', line_color='#94A3B8',
                  annotation_text="50%", annotation_position="right")
    fig = STYLER.apply_standard_styling(
        fig, "Cumulative Population Distribution by Travel Time",
        "Travel Time (min)", "Cumulative % Population", height=520)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig


def _render_inequality_table(az, bz, label_a, label_b):
    ma = az.get_inequality_metrics()
    mb = bz.get_inequality_metrics()
    rows = [
        ("Gini Coefficient (travel time)",  f"{ma['gini']:.4f}",    f"{mb['gini']:.4f}",
         mb['gini'] - ma['gini'], 'lower'),
        ("P90 / P10 Ratio",                 f"{ma['p90_p10']:.2f}x", f"{mb['p90_p10']:.2f}x",
         mb['p90_p10'] - ma['p90_p10'], 'lower'),
        ("P10 (fastest 10% of zones)",      f"{ma['p10']:.1f} min",  f"{mb['p10']:.1f} min",
         mb['p10'] - ma['p10'], 'lower'),
        ("P90 (slowest 10% of zones)",      f"{ma['p90']:.1f} min",  f"{mb['p90']:.1f} min",
         mb['p90'] - ma['p90'], 'lower'),
        ("Std Deviation",                   f"{ma['std']:.1f} min",  f"{mb['std']:.1f} min",
         mb['std'] - ma['std'], 'lower'),
    ]

    col1, col2, col3, col4 = st.columns(4)
    st.markdown(f"""
| Metric | {label_a} | {label_b} | Δ (B−A) |
|---|---|---|---|
""" + "\n".join([f"| {r[0]} | {r[1]} | {r[2]} | "
                  f"{'**' if abs(r[3]) >= 0.01 else ''}"
                  f"{'+' if r[3]>0 else ''}{r[3]:.3f}"
                  f"{'**' if abs(r[3]) >= 0.01 else ''} |" for r in rows]))

    st.caption("Gini = 0 means perfectly equal travel times; 1 = maximally unequal. "
               "P90/P10 ratio measures how much worse the 'worst-served' zones are vs the 'best-served'.")


def _render_distribution_stats_table(az, bz, label_a, label_b):
    def stats_for(analyzer):
        zb = analyzer.get_zone_best_access()
        t  = zb['Tiempo_Total_Minutos']
        w  = zb['Poblacion']
        return {
            'Min':              t.min(),
            'P10':              np.percentile(t, 10),
            'Median (zones)':   np.median(t),
            'Pop-Weighted Mean':np.average(t, weights=w),
            'P90':              np.percentile(t, 90),
            'Max':              t.max(),
            'Std Dev':          t.std(),
        }

    sa, sb = stats_for(az), stats_for(bz)
    rows   = []
    for k in sa:
        d = sb[k] - sa[k]
        rows.append({'Metric': k,
                     label_a: f"{sa[k]:.1f} min",
                     label_b: f"{sb[k]:.1f} min",
                     'Δ (B−A)': f"{'+' if d>0 else ''}{d:.1f} min",
                     '_d': d})

    def highlight(row):
        val = row['_d']
        clr = delta_color(val, 'lower')
        return ['', '', '', f'color:{clr};font-weight:bold', '']

    df_s = pd.DataFrame(rows)
    st.dataframe(
        df_s[['Metric', label_a, label_b, 'Δ (B−A)', '_d']]
            .style.apply(highlight, axis=1)
            .hide(axis='columns', subset=['_d']),
        use_container_width=True, hide_index=True
    )


# ============================================================================
# TAB 3 — GEOGRAPHIC ANALYSIS
# ============================================================================

def render_geographic_analysis(az, bz, label_a, label_b):
    st.markdown("## 🗺️ Geographic Analysis Comparison")

    for geo_col, unit_label in [('Municipio', 'Municipality'), ('Comarca', 'Comarca')]:
        st.markdown(f"### {unit_label} Level")
        geo_a = az.get_geographic_analysis(geo_col)
        geo_b = bz.get_geographic_analysis(geo_col)

        t1, t2, t3 = st.tabs([
            f"📊 Side-by-Side",
            f"📉 Delta Charts",
            f"📋 Delta Table"
        ])
        with t1: _render_geo_side_by_side(geo_a, geo_b, label_a, label_b, unit_label)
        with t2: _render_geo_delta_charts(geo_a, geo_b, label_a, label_b, unit_label)
        with t3: _render_geo_delta_table(geo_a, geo_b, label_a, label_b)
        st.markdown("---")


def _render_geo_side_by_side(geo_a, geo_b, label_a, label_b, unit_type):
    c1, c2 = st.columns(2)
    for col, geo, label in [(c1, geo_a, label_a), (c2, geo_b, label_b)]:
        with col:
            cd = geo.head(10)
            fig = go.Figure()
            cat_colors = [CONFIG.COLORS[k] for k in
                          ['excellent','good','fair','moderate','poor','very_poor']]
            cat_labels = ['🟢 Excellent','🟡 Good','🟠 Fair','🔶 Moderate','🔴 Poor','⚫ Very Poor']
            for cat_col, clr, lbl in zip(CATEGORY_COLS, cat_colors, cat_labels):
                if cat_col in cd.columns:
                    fig.add_trace(go.Bar(
                        name=lbl, y=cd['Geographic Unit'], x=cd[cat_col],
                        orientation='h', marker_color=clr,
                        text=cd[cat_col].round(0), textposition='inside',
                        texttemplate='%{text:.0f}%'
                    ))
            fig = STYLER.apply_standard_styling(
                fig, f"{label}", '% Population', unit_type,
                height=max(500, len(cd)*35))
            fig.update_layout(
                barmode='stack',
                yaxis=dict(categoryorder='total ascending'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                showlegend=(col == c2)
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_geo_delta_charts(geo_a, geo_b, label_a, label_b, unit_type):
    merged = geo_a[['Geographic Unit','Avg Travel Time','Total Population']].merge(
        geo_b[['Geographic Unit','Avg Travel Time']], on='Geographic Unit', suffixes=('_a','_b'))
    merged['Delta_Time'] = merged['Avg Travel Time_b'] - merged['Avg Travel Time_a']
    merged = merged.sort_values('Delta_Time')

    # Avg travel time delta
    colors = [delta_color(d, 'lower') for d in merged['Delta_Time']]
    fig = go.Figure(go.Bar(
        x=merged['Delta_Time'], y=merged['Geographic Unit'],
        orientation='h', marker_color=colors,
        text=[f"{'+'if d>0 else ''}{d:.1f} min" for d in merged['Delta_Time']],
        textposition='outside',
        customdata=merged[['Avg Travel Time_a','Avg Travel Time_b','Total Population']].values,
        hovertemplate=(f"<b>%{{y}}</b><br>{label_a}: %{{customdata[0]:.1f}} min<br>"
                       f"{label_b}: %{{customdata[1]:.1f}} min<br>"
                       "Δ: %{x:.1f} min<br>Pop: %{customdata[2]:,.0f}<extra></extra>")
    ))
    fig = STYLER.apply_standard_styling(
        fig, f"Avg Travel Time Change ({label_b} − {label_a}) by {unit_type}",
        "Δ Minutes", unit_type, height=max(500, len(merged)*30))
    fig.add_vline(x=0, line_color='#94A3B8', line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)

    # Excellent% delta
    if 'Excellent (%)' in geo_a.columns:
        m2 = geo_a[['Geographic Unit','Excellent (%)']].merge(
            geo_b[['Geographic Unit','Excellent (%)']], on='Geographic Unit', suffixes=('_a','_b'))
        m2['Delta_Exc'] = m2['Excellent (%)_b'] - m2['Excellent (%)_a']
        m2 = m2.sort_values('Delta_Exc')
        c2 = [CONFIG.DELTA_POS if d > 0.5 else (CONFIG.DELTA_NEG if d < -0.5 else CONFIG.DELTA_NEU)
              for d in m2['Delta_Exc']]
        fig2 = go.Figure(go.Bar(
            x=m2['Delta_Exc'], y=m2['Geographic Unit'], orientation='h',
            marker_color=c2,
            text=[f"{'+'if d>0 else ''}{d:.1f}pp" for d in m2['Delta_Exc']],
            textposition='outside'
        ))
        fig2 = STYLER.apply_standard_styling(
            fig2, f"Excellent Access Change ({label_b} − {label_a}) by {unit_type}",
            "Δ pp", unit_type, height=max(500, len(m2)*30))
        fig2.add_vline(x=0, line_color='#94A3B8', line_width=1.5)
        st.plotly_chart(fig2, use_container_width=True)


def _render_geo_delta_table(geo_a, geo_b, label_a, label_b):
    key_cols = ['Geographic Unit','Total Population','Avg Travel Time'] + CATEGORY_COLS
    avail_a  = [c for c in key_cols if c in geo_a.columns]
    avail_b  = [c for c in key_cols if c in geo_b.columns]
    merged   = geo_a[avail_a].merge(geo_b[avail_b], on='Geographic Unit', suffixes=('_A','_B'))

    numeric_base = ['Total Population','Avg Travel Time'] + CATEGORY_COLS
    for col in numeric_base:
        ca, cb = f"{col}_A", f"{col}_B"
        if ca in merged.columns and cb in merged.columns:
            merged[f"Δ {col}"] = merged[cb] - merged[ca]

    rename = {}
    for col in numeric_base:
        if f'{col}_A' in merged.columns: rename[f'{col}_A'] = f'{col} ({label_a})'
        if f'{col}_B' in merged.columns: rename[f'{col}_B'] = f'{col} ({label_b})'
    merged = merged.rename(columns=rename)

    display_cols = ['Geographic Unit']
    for col in numeric_base:
        for suf in [f' ({label_a})', f' ({label_b})']:
            if f'{col}{suf}' in merged.columns: display_cols.append(f'{col}{suf}')
        if f'Δ {col}' in merged.columns: display_cols.append(f'Δ {col}')
    display_cols = [c for c in display_cols if c in merged.columns]

    fmt = {}
    for c in display_cols:
        if 'Population' in c and not c.startswith('Δ'): fmt[c] = '{:,.0f}'
        elif 'Travel Time' in c and not c.startswith('Δ'): fmt[c] = '{:.1f} min'
        elif c.startswith('Δ') and 'Travel' in c: fmt[c] = '{:+.1f} min'
        elif c.startswith('Δ'): fmt[c] = '{:+.1f}pp'
        elif '(%)' in c: fmt[c] = '{:.1f}%'

    def color_deltas(row):
        styles = []
        for c in row.index:
            if c.startswith('Δ'):
                try:
                    raw = str(row[c]).replace('+','').replace(' min','').replace('pp','').replace('%','')
                    val = float(raw)
                    better = 'lower' if 'Travel' in c else 'higher'
                    styles.append(f'color:{delta_color(val, better)};font-weight:bold')
                except Exception:
                    styles.append('')
            else:
                styles.append('')
        return styles

    sort_col = next((c for c in display_cols if 'Total Population' in c and label_a in c), display_cols[1])
    st.dataframe(
        merged[display_cols].sort_values(sort_col, ascending=False)
            .style.apply(color_deltas, axis=1).format(fmt, na_rep='—'),
        use_container_width=True, hide_index=True, height=500
    )


# ============================================================================
# TAB 4 — ZONE-LEVEL WINNER / LOSER ANALYSIS
# ============================================================================

def render_zone_analysis(az, bz, label_a, label_b):
    st.markdown("## 🏆 Zone-Level Winner / Loser Analysis")
    st.markdown(
        "Each row is one origin zone. **Delta** = best travel time in B minus best travel time in A. "
        "Negative = zone improved; positive = zone degraded."
    )

    best_a = az.get_zone_best_poi().rename(columns={
        'Best_Time': 'Best_Time_A', 'Best_POI': 'Best_POI_A', 'Best_Category': 'Cat_A'})
    best_b = bz.get_zone_best_poi().rename(columns={
        'Best_Time': 'Best_Time_B', 'Best_POI': 'Best_POI_B', 'Best_Category': 'Cat_B'})

    merged = best_a[['Zona_Origen','Nombre_Origen','Poblacion','Municipio','Comarca',
                      'Best_Time_A','Best_POI_A','Cat_A']].merge(
        best_b[['Zona_Origen','Best_Time_B','Best_POI_B','Cat_B']], on='Zona_Origen', how='outer')

    merged['Delta_Time']   = merged['Best_Time_B'] - merged['Best_Time_A']
    merged['POI_Changed']  = merged['Best_POI_A'] != merged['Best_POI_B']
    merged['Cat_Changed']  = merged['Cat_A'].astype(str) != merged['Cat_B'].astype(str)

    # Threshold crossing
    thresholds = [30, 45, 60, 75, 90]
    def crossed(row):
        ta, tb = row.get('Best_Time_A', np.nan), row.get('Best_Time_B', np.nan)
        if pd.isna(ta) or pd.isna(tb): return ''
        crossings = []
        for th in thresholds:
            if ta <= th < tb:  crossings.append(f"↑{th}")   # degraded past threshold
            elif tb <= th < ta: crossings.append(f"↓{th}")  # improved past threshold
        return ', '.join(crossings)
    merged['Threshold_Cross'] = merged.apply(crossed, axis=1)

    # ── Scatter: Delta vs population ─────────────────────────────────────────
    st.markdown("### Travel Time Change per Zone (population-sized bubbles)")
    _render_zone_scatter(merged, label_a, label_b)
    st.markdown("---")

    # ── Threshold-crossing zones ─────────────────────────────────────────────
    st.markdown("### 🚦 Zones That Crossed an Accessibility Threshold")
    _render_threshold_crossing_table(merged, label_a, label_b)
    st.markdown("---")

    # ── Top improvers / degraders ─────────────────────────────────────────────
    top_n = st.slider("Show top N zones per direction", 5, 30, 10, key="zone_topn")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### ✅ Top {top_n} Most Improved Zones")
        _render_zone_bar(merged.dropna(subset=['Delta_Time']).nsmallest(top_n, 'Delta_Time'),
                         label_a, label_b, improved=True)
    with c2:
        st.markdown(f"#### ⚠️ Top {top_n} Most Degraded Zones")
        _render_zone_bar(merged.dropna(subset=['Delta_Time']).nlargest(top_n, 'Delta_Time'),
                         label_a, label_b, improved=False)

    st.markdown("---")

    # ── Full sortable table ───────────────────────────────────────────────────
    st.markdown("### Full Zone Comparison Table")
    _render_zone_full_table(merged, label_a, label_b)


def _render_zone_scatter(merged, label_a, label_b):
    df = merged.dropna(subset=['Delta_Time','Best_Time_A','Poblacion'])
    colors = [delta_color(d, 'lower') for d in df['Delta_Time']]

    fig = go.Figure(go.Scatter(
        x=df['Best_Time_A'],
        y=df['Delta_Time'],
        mode='markers',
        marker=dict(
            size=np.sqrt(df['Poblacion'] / df['Poblacion'].max()) * 30 + 5,
            color=colors, opacity=0.7,
            line=dict(color='white', width=0.5)
        ),
        text=df['Nombre_Origen'],
        customdata=df[['Poblacion','Best_POI_A','Best_POI_B','Best_Time_B']].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{label_a}: %{{x:.1f}} min (via %{{customdata[1]}})<br>"
            f"{label_b}: %{{customdata[3]:.1f}} min (via %{{customdata[2]}})<br>"
            "Δ: %{y:.1f} min<br>"
            "Population: %{customdata[0]:,.0f}<extra></extra>"
        )
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='#94A3B8')
    for thresh in [30, 45, 60, 75]:
        fig.add_vline(x=thresh, line_dash='dot', line_color='#CBD5E1',
                      annotation_text=f"{thresh}m", annotation_font_size=10)

    fig = STYLER.apply_standard_styling(
        fig,
        f"Zone Change: {label_a} Travel Time vs Δ (B−A) — bubble size = population",
        f"{label_a} Best Travel Time (min)", "Δ Travel Time (min, B−A)", height=560)
    fig.update_layout(margin=dict(l=80, r=80))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Green = improved in B  |  🔴 Red = degraded in B  |  Bubble size ∝ population")


def _render_threshold_crossing_table(merged, label_a, label_b):
    crossing = merged[merged['Threshold_Cross'] != ''].copy()
    if crossing.empty:
        st.info("No zones crossed an accessibility threshold between the two scenarios.")
        return

    improved_cross  = crossing[crossing['Delta_Time'] < 0]
    degraded_cross  = crossing[crossing['Delta_Time'] > 0]

    st.markdown(
        f"**{len(improved_cross)}** zones improved across a threshold  |  "
        f"**{len(degraded_cross)}** zones degraded across a threshold  |  "
        f"**{len(crossing)}** total"
    )

    display_cols = ['Nombre_Origen','Municipio','Poblacion',
                    'Best_Time_A','Best_POI_A','Best_Time_B','Best_POI_B',
                    'Delta_Time','Threshold_Cross']
    avail = [c for c in display_cols if c in crossing.columns]
    rename = {
        'Nombre_Origen':   'Zone',
        'Best_Time_A':     f'Time ({label_a})',
        'Best_POI_A':      f'Best POI ({label_a})',
        'Best_Time_B':     f'Time ({label_b})',
        'Best_POI_B':      f'Best POI ({label_b})',
        'Delta_Time':      'Δ Time',
        'Threshold_Cross': 'Threshold Crossed',
        'Poblacion':       'Population',
    }
    disp = crossing[avail].rename(columns=rename).sort_values('Δ Time')

    def hl(row):
        styles = []
        for c in row.index:
            if c == 'Δ Time':
                val = row[c]
                styles.append(f'color:{delta_color(val,"lower")};font-weight:bold')
            else:
                styles.append('')
        return styles

    fmt = {'Population': '{:,.0f}',
           f'Time ({label_a})': '{:.1f} min',
           f'Time ({label_b})': '{:.1f} min',
           'Δ Time': '{:+.1f} min'}
    st.dataframe(disp.style.apply(hl, axis=1).format(fmt, na_rep='—'),
                 use_container_width=True, hide_index=True, height=400)


def _render_zone_bar(df_subset, label_a, label_b, improved: bool):
    if df_subset.empty:
        st.info("No zones in this category.")
        return
    color = CONFIG.DELTA_POS if improved else CONFIG.DELTA_NEG
    fig = go.Figure(go.Bar(
        x=df_subset['Delta_Time'],
        y=df_subset['Nombre_Origen'],
        orientation='h',
        marker_color=color,
        text=[f"{'+' if d>0 else ''}{d:.1f} min" for d in df_subset['Delta_Time']],
        textposition='outside',
        customdata=df_subset[['Best_Time_A','Best_Time_B','Poblacion',
                               'Best_POI_A','Best_POI_B']].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{label_a}: %{{customdata[0]:.1f}} min (via %{{customdata[3]}})<br>"
            f"{label_b}: %{{customdata[1]:.1f}} min (via %{{customdata[4]}})<br>"
            "Δ: %{x:.1f} min | Pop: %{customdata[2]:,.0f}<extra></extra>"
        )
    ))
    fig = STYLER.apply_standard_styling(
        fig, '', 'Δ Travel Time (min)', 'Zone',
        height=max(400, len(df_subset)*35))
    fig.update_layout(margin=dict(l=160, r=80, t=40, b=40))
    fig.add_vline(x=0, line_color='#94A3B8', line_width=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_zone_full_table(merged, label_a, label_b):
    cols = ['Nombre_Origen','Municipio','Comarca','Poblacion',
            'Best_Time_A','Best_POI_A','Cat_A',
            'Best_Time_B','Best_POI_B','Cat_B',
            'Delta_Time','POI_Changed','Cat_Changed','Threshold_Cross']
    avail = [c for c in cols if c in merged.columns]
    disp  = merged[avail].copy().rename(columns={
        'Nombre_Origen':   'Zone',
        'Poblacion':       'Population',
        'Best_Time_A':     f'Time ({label_a})',
        'Best_POI_A':      f'Best POI ({label_a})',
        'Cat_A':           f'Category ({label_a})',
        'Best_Time_B':     f'Time ({label_b})',
        'Best_POI_B':      f'Best POI ({label_b})',
        'Cat_B':           f'Category ({label_b})',
        'Delta_Time':      'Δ Time',
        'POI_Changed':     'POI Changed?',
        'Cat_Changed':     'Category Changed?',
        'Threshold_Cross': 'Threshold Crossed',
    })

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    show_changed  = fc1.checkbox("Show only zones where POI changed", value=False)
    show_cat_chg  = fc2.checkbox("Show only zones where category changed", value=False)
    show_crossing = fc3.checkbox("Show only threshold-crossing zones", value=False)

    if show_changed and 'POI Changed?' in disp.columns:
        disp = disp[disp['POI Changed?'] == True]
    if show_cat_chg and 'Category Changed?' in disp.columns:
        disp = disp[disp['Category Changed?'] == True]
    if show_crossing and 'Threshold Crossed' in disp.columns:
        disp = disp[disp['Threshold Crossed'] != '']

    fmt = {
        'Population':         '{:,.0f}',
        f'Time ({label_a})':  '{:.1f} min',
        f'Time ({label_b})':  '{:.1f} min',
        'Δ Time':             '{:+.1f} min',
    }

    def hl(row):
        return [f'color:{delta_color(row["Δ Time"],"lower")};font-weight:bold'
                if c == 'Δ Time' else '' for c in row.index]

    st.dataframe(
        disp.sort_values('Δ Time').style.apply(hl, axis=1).format(fmt, na_rep='—'),
        use_container_width=True, hide_index=True, height=500
    )
    st.caption(f"Showing {len(disp):,} zones")


# ============================================================================
# TAB 5 — POI DRILL-DOWN  ("why did this zone change?")
# ============================================================================

def render_poi_drilldown(az, bz, label_a, label_b):
    st.markdown("## 🔍 POI Drill-Down — Why Did a Zone Change?")
    st.markdown(
        "Select an origin zone to see the full ranked list of POIs and their travel times "
        "in both scenarios. Reveals whether improvement came from a new POI, a faster route, "
        "or a change in which POI is now closest."
    )

    # Build zone list from zones that exist in both
    best_a = az.get_zone_best_poi()
    best_b = bz.get_zone_best_poi()
    common = pd.merge(best_a[['Zona_Origen','Nombre_Origen']],
                      best_b[['Zona_Origen']], on='Zona_Origen')
    common['Delta'] = (best_b.set_index('Zona_Origen')['Best_Time'] -
                       best_a.set_index('Zona_Origen')['Best_Time']).reindex(common['Zona_Origen']).values

    # Sort options
    sort_opt = st.radio("Sort zone selector by", ["Largest improvement", "Largest degradation",
                                                   "Zone name"], horizontal=True)
    if sort_opt == "Largest improvement":
        common = common.sort_values('Delta')
    elif sort_opt == "Largest degradation":
        common = common.sort_values('Delta', ascending=False)
    else:
        common = common.sort_values('Nombre_Origen')

    zone_options = common['Nombre_Origen'].tolist()
    zone_ids     = common['Zona_Origen'].tolist()

    if not zone_options:
        st.warning("No zones found in common between the two scenarios.")
        return

    selected_name = st.selectbox("Select zone", zone_options)
    selected_id   = zone_ids[zone_options.index(selected_name)]

    # Pull POI lists
    pois_a = az.get_all_poi_times()
    pois_b = bz.get_all_poi_times()

    zone_a = pois_a[pois_a['Zona_Origen'] == selected_id][['Nombre_Destino','Travel_Time']].copy()
    zone_b = pois_b[pois_b['Zona_Origen'] == selected_id][['Nombre_Destino','Travel_Time']].copy()

    zone_a = zone_a.rename(columns={'Travel_Time': f'Time ({label_a})'})
    zone_b = zone_b.rename(columns={'Travel_Time': f'Time ({label_b})'})

    poi_merged = zone_a.merge(zone_b, on='Nombre_Destino', how='outer')
    poi_merged['Delta'] = poi_merged[f'Time ({label_b})'] - poi_merged[f'Time ({label_a})']
    poi_merged['Status'] = poi_merged.apply(
        lambda r: '🆕 New in B' if pd.isna(r[f'Time ({label_a})'])
        else ('🗑️ Removed in B' if pd.isna(r[f'Time ({label_b})'])
              else ('⬇️ Faster' if r['Delta'] < -1 else ('⬆️ Slower' if r['Delta'] > 1 else '➡️ Unchanged'))),
        axis=1
    )
    poi_merged = poi_merged.sort_values(f'Time ({label_b})', na_position='last')

    # Summary banner
    best_a_row = zone_a.dropna().nsmallest(1, f'Time ({label_a})')
    best_b_row = zone_b.dropna().nsmallest(1, f'Time ({label_b})')
    btime_a = best_a_row[f'Time ({label_a})'].values[0] if len(best_a_row) else np.nan
    btime_b = best_b_row[f'Time ({label_b})'].values[0] if len(best_b_row) else np.nan
    bpoi_a  = best_a_row['Nombre_Destino'].values[0] if len(best_a_row) else '—'
    bpoi_b  = best_b_row['Nombre_Destino'].values[0] if len(best_b_row) else '—'
    delta_best = btime_b - btime_a if not (np.isnan(btime_a) or np.isnan(btime_b)) else np.nan

    dcolor = delta_color(delta_best, 'lower') if not np.isnan(delta_best) else CONFIG.DELTA_NEU
    dsign  = "+" if (not np.isnan(delta_best) and delta_best > 0) else ""

    st.markdown(f"""
<div style="background:#F8FAFC;border-radius:10px;padding:16px 20px;
            border:1px solid #E2E8F0;margin-bottom:16px">
  <div style="font-size:16px;font-weight:700;color:#1E293B;margin-bottom:8px">
      📍 {selected_name}
  </div>
  <div style="display:flex;gap:32px;flex-wrap:wrap">
    <div>
      <span style="font-size:12px;color:#64748B">{label_a} best access</span><br>
      <span style="font-size:18px;font-weight:700;color:{CONFIG.SCENARIO_COLORS[0]}">
          {btime_a:.1f} min</span>
      <span style="font-size:13px;color:#64748B"> via {bpoi_a}</span>
    </div>
    <div>
      <span style="font-size:12px;color:#64748B">{label_b} best access</span><br>
      <span style="font-size:18px;font-weight:700;color:{CONFIG.SCENARIO_COLORS[1]}">
          {btime_b:.1f} min</span>
      <span style="font-size:13px;color:#64748B"> via {bpoi_b}</span>
    </div>
    <div>
      <span style="font-size:12px;color:#64748B">Change</span><br>
      <span style="font-size:18px;font-weight:700;color:{dcolor}">
          {dsign}{delta_best:.1f} min</span>
      {'<span style="font-size:13px;color:#64748B"> same POI</span>' if bpoi_a == bpoi_b
       else f'<span style="font-size:13px;color:#F59E0B"> ⚡ POI changed</span>'}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Chart: all POIs ranked by Scenario B time
    st.markdown("#### All POIs — Travel Time Comparison")
    _render_poi_drilldown_chart(poi_merged, label_a, label_b, selected_name)

    st.markdown("#### Detail Table")
    _render_poi_drilldown_table(poi_merged, label_a, label_b)

    # JRT vs PJT divergence callout
    if az.has_pjt and bz.has_pjt:
        st.markdown("---")
        st.markdown("#### JRT vs PJT Divergence for This Zone")
        _render_jrt_pjt_divergence(az, bz, selected_id, label_a, label_b)


def _render_poi_drilldown_chart(poi_merged, label_a, label_b, zone_name):
    df = poi_merged.dropna(subset=[f'Time ({label_b})']).head(20)
    if df.empty:
        st.info("No POI data available.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=label_a, y=df['Nombre_Destino'], x=df[f'Time ({label_a})'],
        orientation='h', marker_color=CONFIG.SCENARIO_COLORS[0], opacity=0.8,
        text=df[f'Time ({label_a})'].round(1), textposition='inside'
    ))
    fig.add_trace(go.Bar(
        name=label_b, y=df['Nombre_Destino'], x=df[f'Time ({label_b})'],
        orientation='h', marker_color=CONFIG.SCENARIO_COLORS[1], opacity=0.8,
        text=df[f'Time ({label_b})'].round(1), textposition='inside'
    ))

    for thresh in [30, 45, 60]:
        fig.add_vline(x=thresh, line_dash='dot', line_color='#CBD5E1',
                      annotation_text=f"{thresh}m", annotation_font_size=10)

    fig = STYLER.apply_standard_styling(
        fig, f"POI Travel Times — {zone_name}",
        "Travel Time (min)", "POI",
        height=max(450, len(df) * 38))
    fig.update_layout(
        barmode='group',
        yaxis=dict(categoryorder='total ascending'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_poi_drilldown_table(poi_merged, label_a, label_b):
    disp = poi_merged.rename(columns={'Nombre_Destino': 'POI'})

    def hl(row):
        return [f'color:{delta_color(row["Delta"],"lower")};font-weight:bold'
                if c == 'Delta' else '' for c in row.index]

    fmt = {f'Time ({label_a})': '{:.1f} min',
           f'Time ({label_b})': '{:.1f} min',
           'Delta':             '{:+.1f} min'}
    st.dataframe(
        disp[['POI', f'Time ({label_a})', f'Time ({label_b})', 'Delta', 'Status']]
            .style.apply(hl, axis=1).format(fmt, na_rep='—'),
        use_container_width=True, hide_index=True, height=400
    )


def _render_jrt_pjt_divergence(az, bz, zone_id, label_a, label_b):
    """Show JRT vs PJT for the same zone — helps explain if more transfers cause PJT gap."""
    rows = []
    for analyzer, label in [(az, label_a), (bz, label_b)]:
        z_df = analyzer.df[analyzer.df['Zona_Origen'] == zone_id]
        if z_df.empty:
            continue
        for _, row in z_df.groupby('Nombre_Destino').first().reset_index().iterrows():
            entry = {'Scenario': label, 'POI': row['Nombre_Destino'],
                     'JRT (Tiempo_Trayecto)': row.get('Tiempo_Trayecto', np.nan)}
            if 'Tiempo_Viaje_Percibido' in row:
                entry['PJT (Tiempo_Viaje_Percibido)'] = row['Tiempo_Viaje_Percibido']
            if 'Num_Transbordos' in row:
                entry['Transfers'] = row['Num_Transbordos']
            rows.append(entry)

    if not rows:
        st.info("No data available for JRT/PJT divergence.")
        return

    df_div = pd.DataFrame(rows)
    if 'PJT (Tiempo_Viaje_Percibido)' in df_div.columns:
        df_div['PJT − JRT'] = df_div['PJT (Tiempo_Viaje_Percibido)'] - df_div['JRT (Tiempo_Trayecto)']
    st.dataframe(df_div.sort_values(['Scenario', 'JRT (Tiempo_Trayecto)']),
                 use_container_width=True, hide_index=True, height=350)
    st.caption("A large PJT − JRT gap for a POI suggests heavy transfer/waiting penalties on that route.")


# ============================================================================
# TAB 6 — INEQUALITY DEEP DIVE
# ============================================================================

def render_inequality(az, bz, label_a, label_b):
    st.markdown("## ⚖️ Inequality Deep Dive")
    st.markdown(
        "Do the two scenarios redistribute accessibility equally, "
        "or do gains concentrate in already well-served zones?"
    )

    ma = az.get_inequality_metrics()
    mb = bz.get_inequality_metrics()

    # KPI strip
    c1, c2, c3, c4, c5 = st.columns(5)
    ineq_kpis = [
        (c1, "Gini Coefficient",    f"{ma['gini']:.4f}",    f"{mb['gini']:.4f}",    mb['gini']    - ma['gini'],    'lower'),
        (c2, "P90 / P10 Ratio",     f"{ma['p90_p10']:.2f}x",f"{mb['p90_p10']:.2f}x",mb['p90_p10'] - ma['p90_p10'],'lower'),
        (c3, "P10 (best zones)",     f"{ma['p10']:.1f} min", f"{mb['p10']:.1f} min", mb['p10']     - ma['p10'],     'lower'),
        (c4, "P90 (worst zones)",    f"{ma['p90']:.1f} min", f"{mb['p90']:.1f} min", mb['p90']     - ma['p90'],     'lower'),
        (c5, "Std Deviation",        f"{ma['std']:.1f} min", f"{mb['std']:.1f} min", mb['std']     - ma['std'],     'lower'),
    ]
    for col, lbl, va, vb, delta, direction in ineq_kpis:
        _kpi_card(col, lbl, va, vb, delta, direction, label_a, label_b,
                  ' min' if 'Gini' not in lbl and 'Ratio' not in lbl else '')

    st.markdown("---")
    st.markdown("### Lorenz Curve — Population vs Travel Time")
    _render_lorenz(az, bz, label_a, label_b)
    st.markdown("---")

    st.markdown("### Population Decile Analysis")
    _render_decile_analysis(az, bz, label_a, label_b)
    st.markdown("---")

    st.markdown("### JRT vs PJT Delta per Zone (transfer burden)")
    _render_jrt_pjt_scatter(az, bz, label_a, label_b)


def _render_lorenz(az, bz, label_a, label_b):
    fig = go.Figure()
    # Perfect equality line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                              line=dict(color='#94A3B8', dash='dash'),
                              name='Perfect equality', showlegend=True))

    for analyzer, label, color in [(az, label_a, CONFIG.SCENARIO_COLORS[0]),
                                    (bz, label_b, CONFIG.SCENARIO_COLORS[1])]:
        zb = analyzer.get_zone_best_access().sort_values('Tiempo_Total_Minutos')
        w  = zb['Poblacion'].values / zb['Poblacion'].sum()
        t  = zb['Tiempo_Total_Minutos'].values
        cum_pop = np.concatenate([[0], np.cumsum(w)])
        cum_t   = np.concatenate([[0], np.cumsum(w * t) / (w * t).sum()])
        fig.add_trace(go.Scatter(x=cum_pop, y=cum_t, mode='lines',
                                  name=label, line=dict(color=color, width=2.5)))

    fig.update_layout(
        title=dict(text="Lorenz Curve: Cumulative Population vs Cumulative Travel Time Burden",
                   font=dict(size=18)),
        xaxis=dict(title="Cumulative Share of Population (sorted by travel time)",
                   tickformat='.0%', range=[0, 1]),
        yaxis=dict(title="Cumulative Share of Travel Time Burden", tickformat='.0%', range=[0, 1]),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Closer to the diagonal = more equal distribution of travel time burden across the population.")


def _render_decile_analysis(az, bz, label_a, label_b):
    """Average travel time by population decile (population-sorted)."""
    fig = go.Figure()

    for analyzer, label, color in [(az, label_a, CONFIG.SCENARIO_COLORS[0]),
                                    (bz, label_b, CONFIG.SCENARIO_COLORS[1])]:
        zb    = analyzer.get_zone_best_access().sort_values('Tiempo_Total_Minutos')
        zb['cum_pop'] = zb['Poblacion'].cumsum() / zb['Poblacion'].sum()
        deciles = []
        for d in range(1, 11):
            lower = (d - 1) / 10
            upper = d / 10
            mask  = (zb['cum_pop'] > lower) & (zb['cum_pop'] <= upper)
            sub   = zb[mask]
            if not sub.empty:
                avg = np.average(sub['Tiempo_Total_Minutos'], weights=sub['Poblacion'])
            else:
                avg = np.nan
            deciles.append(avg)

        fig.add_trace(go.Scatter(
            x=list(range(1, 11)), y=deciles, mode='lines+markers',
            name=label, line=dict(color=color, width=2.5),
            marker=dict(size=8),
            text=[f"{v:.1f} min" if not np.isnan(v) else '' for v in deciles],
            textposition='top center'
        ))

    fig.update_layout(
        title=dict(text="Average Travel Time by Population Decile "
                        "(Decile 1 = best-served, Decile 10 = worst-served)",
                   font=dict(size=16)),
        xaxis=dict(title="Population Decile", tickvals=list(range(1, 11)),
                   ticktext=[f"D{i}" for i in range(1, 11)]),
        yaxis=dict(title="Avg Travel Time (min)"),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A scenario that flattens this curve improves equity. "
               "If only D1–D3 improve while D8–D10 stay flat, gains are concentrated in already well-served areas.")


def _render_jrt_pjt_scatter(az, bz, label_a, label_b):
    """Compare JRT delta vs PJT delta per zone. Zones above the diagonal benefit more in raw time than perceived."""
    has_pjt_a = az.has_pjt
    has_pjt_b = bz.has_pjt

    if not (has_pjt_a and has_pjt_b):
        st.info("PJT column not present in both scenarios — skipping JRT vs PJT divergence chart.")
        return

    def zone_metrics(analyzer):
        zb_jrt = analyzer.df.groupby('Zona_Origen').agg(
            JRT=('Tiempo_Trayecto', 'min'),
            PJT=('Tiempo_Viaje_Percibido', 'min'),
            Pop=('Poblacion', 'first'),
            Name=('Nombre_Origen', 'first'),
        ).reset_index()
        return zb_jrt

    ma_df = zone_metrics(az).rename(columns={'JRT':'JRT_A','PJT':'PJT_A'})
    mb_df = zone_metrics(bz).rename(columns={'JRT':'JRT_B','PJT':'PJT_B'})
    m = ma_df.merge(mb_df[['Zona_Origen','JRT_B','PJT_B']], on='Zona_Origen')
    m['Delta_JRT'] = m['JRT_B'] - m['JRT_A']
    m['Delta_PJT'] = m['PJT_B'] - m['PJT_A']
    m['Transfer_Penalty'] = (m['PJT_A'] - m['JRT_A'])  # baseline penalty

    fig = go.Figure(go.Scatter(
        x=m['Delta_JRT'], y=m['Delta_PJT'],
        mode='markers',
        marker=dict(
            size=np.sqrt(m['Pop'] / m['Pop'].max()) * 25 + 5,
            color=m['Transfer_Penalty'],
            colorscale='RdYlGn_r',
            colorbar=dict(title='Baseline transfer<br>penalty (PJT−JRT, min)'),
            opacity=0.75,
            line=dict(color='white', width=0.5)
        ),
        text=m['Name'],
        customdata=m[['JRT_A','JRT_B','PJT_A','PJT_B','Transfer_Penalty']].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"JRT: %{{customdata[0]:.1f}}→%{{customdata[1]:.1f}} (Δ %{{x:.1f}} min)<br>"
            f"PJT: %{{customdata[2]:.1f}}→%{{customdata[3]:.1f}} (Δ %{{y:.1f}} min)<br>"
            "Transfer penalty (A): %{customdata[4]:.1f} min<extra></extra>"
        )
    ))
    fig.add_shape(type='line', x0=m['Delta_JRT'].min(), y0=m['Delta_JRT'].min(),
                  x1=m['Delta_JRT'].max(), y1=m['Delta_JRT'].max(),
                  line=dict(color='#94A3B8', dash='dash'))
    fig.add_hline(y=0, line_color='#E2E8F0')
    fig.add_vline(x=0, line_color='#E2E8F0')

    fig = STYLER.apply_standard_styling(
        fig, "JRT vs PJT Change per Zone (bubble = population, color = transfer penalty in A)",
        "Δ JRT (min, B−A)", "Δ PJT (min, B−A)", height=560)
    fig.update_layout(margin=dict(l=80, r=80))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Points **above** the diagonal: PJT worsened more than JRT → new/retained transfers add perceived cost. "
        "Points **below**: PJT improved more than JRT → route now avoids transfers."
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
            "Upload two datasets (e.g. baseline vs. proposed scenario) to compare across "
            "Population Accessibility, Travel Time Distributions, Geographic Analysis, "
            "Zone-Level Winners/Losers, POI Drill-Down and Inequality."
        )
        st.markdown("---")

        # ── Sidebar ──────────────────────────────────────────────────────────
        st.sidebar.markdown("## 📁 Upload Scenarios")
        file_a  = st.sidebar.file_uploader("Scenario A (baseline)",    type=['xlsx','xls'], key="file_a")
        file_b  = st.sidebar.file_uploader("Scenario B (comparison)",  type=['xlsx','xls'], key="file_b")
        label_a = st.sidebar.text_input("Label for Scenario A", value="Actual")
        label_b = st.sidebar.text_input("Label for Scenario B", value="Anteproyecto")
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ Metric Settings")

        if not file_a or not file_b:
            st.info("👈 Upload **both** Excel files in the sidebar to begin.")
            self._render_placeholder()
            return

        try:
            df_a_preview = pd.read_excel(file_a); file_a.seek(0)
            df_b_preview = pd.read_excel(file_b); file_b.seek(0)
        except Exception as e:
            st.error(f"Could not preview files: {e}")
            return

        metric_a = metric_selector(df_a_preview, label_a)
        metric_b = metric_selector(df_b_preview, label_b)
        st.sidebar.info(
            f"**{label_a}**: {'PJT' if metric_a=='PJT' else 'JRT'}  \n"
            f"**{label_b}**: {'PJT' if metric_b=='PJT' else 'JRT'}"
        )

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
            "🏆 Zone Winners / Losers",
            "🔍 POI Drill-Down",
            "⚖️ Inequality",
        ])

        with tabs[0]: render_population_accessibility(az, bz, label_a, label_b)
        with tabs[1]: render_travel_time_distributions(az, bz, label_a, label_b)
        with tabs[2]: render_geographic_analysis(az, bz, label_a, label_b)
        with tabs[3]: render_zone_analysis(az, bz, label_a, label_b)
        with tabs[4]: render_poi_drilldown(az, bz, label_a, label_b)
        with tabs[5]: render_inequality(az, bz, label_a, label_b)

    @staticmethod
    def _render_placeholder():
        st.markdown("""
<div style="background:#F1F5F9;border-radius:12px;padding:40px;text-align:center;margin-top:40px">
  <div style="font-size:48px">📊</div>
  <div style="font-size:22px;font-weight:600;color:#334155;margin-top:12px">
    Scenario Comparison Ready
  </div>
  <div style="font-size:15px;color:#64748B;margin-top:8px">
    Upload Scenario A and Scenario B in the sidebar to start<br><br>
    📊 Population Accessibility &nbsp;·&nbsp; ⏱️ Travel Time Distributions<br>
    🗺️ Geographic Analysis &nbsp;·&nbsp; 🏆 Zone Winners/Losers<br>
    🔍 POI Drill-Down &nbsp;·&nbsp; ⚖️ Inequality
  </div>
</div>""", unsafe_allow_html=True)


# ============================================================================
if __name__ == "__main__":
    ComparisonApp().run()