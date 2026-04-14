"""
Visum Vehicle Journeys Comparator
Compares two Visum PUTHPATHLEGS Excel exports with rich analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visum – Comparador de Expediciones",
    page_icon="🚌",
    layout="wide",
)

COLORS = {
    "current":      "#3a3a3a",   # dark gray — Actual
    "anteproyecto": "#D20B12",   # Bizkaibus red — Anteproyecto
    "diff_pos":     "#2ca02c",
    "diff_neg":     "#e07b00",
    "neutral":      "#7f7f7f",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def timedelta_to_minutes(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds() / 60

def minutes_to_hhmm(minutes: float) -> str:
    """Convert minutes-since-midnight to HH:MM display string.

    Visum encodes overnight services with extended notation:
      24:10 = 1450 min, 25:30 = 1530 min, etc.
    These are scheduling times (trip belongs to the previous service day).
    For display we always fold back to real clock time with % 1440:
      1450 -> 00:10,  1530 -> 01:30,  1440 -> 00:00
    Chronological sort order is kept separately via sort_key_overnight().
    """
    if minutes is None:
        return '--:--'
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        return '--:--'
    if np.isnan(minutes) or minutes < 0:
        return '--:--'
    total_min = int(round(minutes)) % 1440   # fold extended Visum times to real clock
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


def sort_key_overnight(dep_min: float, day_start: int = 240) -> float:
    """Chronological sort key that places post-midnight trips after 23:59.

    Two Visum encoding styles exist:
      - Extended:  24:10 stored as 1450 min  (already > 1440, sorts correctly)
      - Wrapped:   00:10 stored as    10 min  (needs +1440 to sort after 23:59)
    We only add 1440 for the wrapped style (dep < day_start = 04:00).
    """
    return dep_min + 1440 if dep_min < day_start else dep_min

def hour_ticks(t_start: float, t_end: float):
    """Return (tickvals, ticktext) for whole-hour marks, folded to clock time."""
    first_h = int(t_start // 60) * 60
    last_h  = min(int(t_end // 60) * 60 + 60, 1440)
    vals    = list(range(first_h, last_h + 1, 60))
    texts   = [minutes_to_hhmm(v) for v in vals]
    return vals, texts


def compute_headway(df: pd.DataFrame) -> pd.DataFrame:
    """Compute headway correctly per line × direction.

    Headway = (last_departure - first_departure) / (n_trips - 1)
    — measured only between the first and last trip of the day (>=04:00),
      so overnight gaps do not distort the average.
    — grouped by LineName × DirectionCode (if available), so outbound and
      inbound are never mixed together.

    Returns a DataFrame with columns:
        line, direction (int or NaN), n_trips, first_dep, last_dep, headway_min
    """
    DAY_START = 240  # 04:00
    has_dir   = "DirectionCode" in df.columns

    group_cols = ["LineName", "DirectionCode"] if has_dir else ["LineName"]
    rows = []
    for keys_g, grp in df.groupby(group_cols):
        line = keys_g[0] if has_dir else keys_g
        dirn = int(keys_g[1]) if has_dir else np.nan

        # Only daytime trips for headway (post-midnight would inflate the span)
        day = grp[grp["dep_min"].apply(sort_key_overnight) < 1440]
        day = day.sort_values("dep_min")
        n   = len(day)
        if n < 2:
            continue
        first_dep = day["dep_min"].iloc[0]
        last_dep  = day["dep_min"].iloc[-1]
        span      = last_dep - first_dep
        if span <= 0:
            continue
        headway   = span / (n - 1)
        rows.append({
            "line":        str(line),
            "direction":   dirn,
            "n_trips":     n,
            "first_dep":   first_dep,
            "last_dep":    last_dep,
            "headway_min": headway,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["line","direction","n_trips","first_dep","last_dep","headway_min"])


def load_journeys(file) -> pd.DataFrame:
    xl = pd.ExcelFile(file)
    sheet = "Vehicle journeys" if "Vehicle journeys" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    dep_col   = next((c for c in df.columns if c.lower() == "dep"), None)
    arr_col   = next((c for c in df.columns if c.lower() == "arr"), None)
    line_col  = next((c for c in df.columns if c.lower() == "linename"), None)
    route_col = next((c for c in df.columns if "route_name_unique" in c.lower()), None)
    dir_col   = next((c for c in df.columns if "directioncode" in c.lower()), None)
    dur_col   = next((c for c in df.columns if c.lower() == "duration"), None)
    len_col   = next((c for c in df.columns if c.lower() == "length"), None)

    if dep_col is None:
        st.error("No se encontró la columna 'Dep'.")
        return pd.DataFrame()

    # Departure minutes
    if pd.api.types.is_timedelta64_dtype(df[dep_col]):
        df["dep_min"] = timedelta_to_minutes(df[dep_col])
    else:
        df["dep_min"] = pd.to_timedelta(df[dep_col].astype(str)).dt.total_seconds() / 60

    # Arrival minutes + travel time
    if arr_col:
        if pd.api.types.is_timedelta64_dtype(df[arr_col]):
            df["arr_min"] = timedelta_to_minutes(df[arr_col])
        else:
            df["arr_min"] = pd.to_timedelta(df[arr_col].astype(str)).dt.total_seconds() / 60
        df["travel_min"] = df["arr_min"] - df["dep_min"]

    # Duration (stored as fraction of day)
    if dur_col:
        df["duration_min"] = df[dur_col] * 24 * 60
    elif "travel_min" in df.columns:
        df["duration_min"] = df["travel_min"]

    # Length (km)
    if len_col:
        df["length_km"] = df[len_col]

    # Keep relevant columns
    keep = {"dep_min": "dep_min"}
    for alias, col in [
        ("LineName", line_col), ("RouteNameUnique", route_col),
        ("DirectionCode", dir_col),
    ]:
        if col:
            keep[col] = alias
    for extra in ["arr_min", "travel_min", "duration_min", "length_km"]:
        if extra in df.columns:
            keep[extra] = extra

    df = df.rename(columns=keep)
    df = df[[c for c in keep.values() if c in df.columns]]
    df = df[df["dep_min"].between(0, 1560)]  # allow up to 26:00 for overnight services
    df["dep_min"] = df["dep_min"].round(2)

    # Normalise LineName → A0651 format
    if "RouteNameUnique" in df.columns and df["RouteNameUnique"].notna().any():
        mask = df["RouteNameUnique"].notna()
        df.loc[mask, "LineName"] = df.loc[mask, "RouteNameUnique"].astype(str).str.strip()
    elif "LineName" in df.columns:
        def _norm(v):
            s = str(v).strip()
            if s.upper().startswith("A"):
                return s.upper()
            try:
                return f"A{int(float(s)):04d}"
            except ValueError:
                return s
        df["LineName"] = df["LineName"].apply(_norm)

    return df


def bin_journeys(df: pd.DataFrame, interval: int, t_start=0, t_end=1440) -> pd.Series:
    bins = np.arange(0, 1441, interval)
    labels = bins[:-1]
    cut = pd.cut(df["dep_min"], bins=bins, labels=labels, right=False)
    s = cut.value_counts().reindex(labels, fill_value=0).sort_index()
    return s[(s.index >= t_start) & (s.index <= t_end)]


def make_x_labels(series: pd.Series) -> list:
    return [minutes_to_hhmm(m) for m in series.index]


def tick_step(interval: int) -> int:
    return max(1, 60 // interval)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuración")

st.sidebar.subheader("Cargar archivos")
file_current = st.sidebar.file_uploader("Escenario **Actual** (.xlsx)", type=["xlsx"], key="cur")
file_ante    = st.sidebar.file_uploader("Escenario **Anteproyecto** (.xlsx)", type=["xlsx"], key="ant")

st.sidebar.subheader("Agrupación temporal")
interval = st.sidebar.radio("Intervalo (minutos)", [5, 10, 15, 20, 30, 60], index=1, horizontal=True)

st.sidebar.subheader("Rango horario")
time_range = st.sidebar.slider("", min_value=0, max_value=1440, value=(300, 1440), step=interval)
t_start, t_end = time_range

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🚌 Comparador de Expediciones — Visum")

if not file_current and not file_ante:
    st.info("👆 Carga al menos un archivo Excel en la barra lateral para comenzar.")
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
dfs, file_labels = {}, {}
if file_current:
    dfs["current"]      = load_journeys(file_current)
    file_labels["current"] = file_current.name
if file_ante:
    dfs["anteproyecto"] = load_journeys(file_ante)
    file_labels["anteproyecto"] = file_ante.name

# ── Line filter (shared) ───────────────────────────────────────────────────────
all_lines = sorted(
    {ln for df in dfs.values() for ln in df["LineName"].dropna().unique()},
    key=str
)

with st.expander("🔍 Filtrar por línea (vacío = todas)"):
    selected_lines = st.multiselect("Líneas", options=all_lines, default=[])

filtered = {}
for key, df in dfs.items():
    fdf = df.copy()
    if selected_lines:
        fdf = fdf[fdf["LineName"].isin(selected_lines)]
    fdf = fdf[fdf["dep_min"].between(t_start, t_end, inclusive="both")]
    filtered[key] = fdf

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.subheader("📊 Resumen global")

has_len_g = any("length_km"    in df.columns for df in filtered.values())
has_dur_g = any("duration_min" in df.columns for df in filtered.values())

# Row 1 — expedition counts
kc     = st.columns(4)
totals = {k: len(v) for k, v in filtered.items()}
with kc[0]:
    st.metric("Expediciones — Actual",       f"{totals.get('current',0):,}"      if "current"      in totals else "–")
with kc[1]:
    st.metric("Expediciones — Anteproyecto", f"{totals.get('anteproyecto',0):,}" if "anteproyecto" in totals else "–")
with kc[2]:
    if len(totals) == 2:
        diff = totals["anteproyecto"] - totals["current"]
        pct  = diff / totals["current"] * 100 if totals["current"] else 0
        st.metric("Δ Expediciones", f"{diff:+,}", delta=f"{pct:+.1f}%")
with kc[3]:
    if len(totals) == 2 and totals["current"]:
        st.metric("Ratio Ante / Actual", f"{totals['anteproyecto']/totals['current']:.3f}")

# Row 2 — km offer
if has_len_g:
    km_cols = st.columns(4)
    for i, (key, lbl) in enumerate([("current","Actual"),("anteproyecto","Anteproyecto")]):
        if key in filtered and "length_km" in filtered[key].columns:
            km_cols[i].metric(f"Km totales ofertados — {lbl}",
                              f"{filtered[key]['length_km'].sum():,.0f} km")
    if len(filtered) == 2:
        keys = list(filtered.keys())
        if all("length_km" in filtered[k].columns for k in keys):
            dk    = filtered[keys[1]]["length_km"].sum() - filtered[keys[0]]["length_km"].sum()
            base  = filtered[keys[0]]["length_km"].sum()
            km_cols[2].metric("Δ Km totales", f"{dk:+,.0f} km")
            km_cols[3].metric("% cambio km oferta", f"{dk/base*100:+.1f}%" if base else "–")

# Row 3 — commercial speed & avg duration
if has_len_g and has_dur_g:
    spd_cols = st.columns(4)
    idx = 0
    for key, lbl in [("current","Actual"),("anteproyecto","Anteproyecto")]:
        if key in filtered:
            df_s = filtered[key]
            if "length_km" in df_s.columns and "duration_min" in df_s.columns:
                df_s = df_s.dropna(subset=["length_km","duration_min"])
                df_s = df_s[df_s["duration_min"] > 0]
                if len(df_s):
                    avg_spd = (df_s["length_km"] / (df_s["duration_min"] / 60)).mean()
                    spd_cols[idx].metric(f"Vel. comercial media — {lbl}", f"{avg_spd:.1f} km/h")
            if "duration_min" in filtered[key].columns:
                spd_cols[idx+1].metric(f"Duración media — {lbl}",
                                       f"{filtered[key]['duration_min'].mean():.1f} min")
        idx += 2

# Row 4 — network: lines, routes, delta
net_cols = st.columns(4)
for i, (key, lbl) in enumerate([("current","Actual"),("anteproyecto","Anteproyecto")]):
    if key in filtered:
        n_lines  = filtered[key]["LineName"].nunique()
        n_routes = (filtered[key]["RouteNameUnique"].nunique()
                    if "RouteNameUnique" in filtered[key].columns else "–")
        net_cols[i].metric(
            f"Líneas / Rutas activas — {lbl}",
            f"{n_lines} líneas",
            delta=f"{n_routes} rutas" if n_routes != "–" else None,
            delta_color="off",
        )
if len(filtered) == 2:
    keys = list(filtered.keys())
    dl   = filtered[keys[1]]["LineName"].nunique() - filtered[keys[0]]["LineName"].nunique()
    net_cols[2].metric("Δ Líneas activas", f"{dl:+d}")
    if all("RouteNameUnique" in filtered[k].columns for k in keys):
        dr = filtered[keys[1]]["RouteNameUnique"].nunique() - filtered[keys[0]]["RouteNameUnique"].nunique()
        net_cols[3].metric("Δ Rutas activas", f"{dr:+d}")

# Row 5 — headway (line×direction, span-based) & vehicle-hours
hw5_cols = st.columns(4)
hw_per = {}
for key in filtered:
    hw_df_c = compute_headway(filtered[key])
    if len(hw_df_c):
        # Weighted mean by n_trips so high-frequency lines count more
        hw_per[key] = np.average(hw_df_c["headway_min"], weights=hw_df_c["n_trips"])

for i, (key, lbl) in enumerate([("current","Actual"),("anteproyecto","Anteproyecto")]):
    if key in hw_per:
        hw5_cols[i].metric(
            f"Headway medio — {lbl}",
            f"{hw_per[key]:.1f} min",
            help="(última salida − primera salida) / (n expediciones − 1), por línea × dirección, ponderado por n expediciones",
        )
if "current" in hw_per and "anteproyecto" in hw_per:
    dhw = hw_per["anteproyecto"] - hw_per["current"]
    hw5_cols[2].metric("Δ Headway medio", f"{dhw:+.1f} min", delta_color="inverse")
if has_len_g and has_dur_g:
    for key, lbl in [("current","Actual"),("anteproyecto","Anteproyecto")]:
        if key in filtered and "length_km" in filtered[key].columns and "duration_min" in filtered[key].columns:
            df_vh = filtered[key].dropna(subset=["length_km","duration_min"])
            df_vh = df_vh[df_vh["duration_min"] > 0]
            if len(df_vh):
                hw5_cols[3].metric(f"Horas-vehículo — {lbl}",
                                   f"{df_vh['duration_min'].sum()/60:,.0f} h")
                break

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Expediciones por hora",
    "🔀 Comparación por línea",
    "📐 Análisis avanzado",
    "📏 Longitud & Duración",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Expediciones por hora
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    binned = {k: bin_journeys(v, interval, t_start, t_end) for k, v in filtered.items()}
    _ref   = binned.get("current") if "current" in binned else binned.get("anteproyecto")
    x_lbl  = make_x_labels(_ref)
    step   = tick_step(interval)

    # ── Chart 1a: Bar + Line overlay ──────────────────────────────────────────
    st.subheader(f"Expediciones por intervalo de {interval} min — barras + línea")

    fig1 = go.Figure()
    for key, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
        if key not in binned:
            continue
        name = f"{key.capitalize()} ({file_labels.get(key,'')})"
        y = binned[key].values
        fig1.add_trace(go.Bar(
            x=x_lbl, y=y, name=name,
            marker_color=color, opacity=0.6,
        ))
        fig1.add_trace(go.Scatter(
            x=x_lbl, y=y, name=f"{key.capitalize()} (tendencia)",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4),
            showlegend=True,
        ))

    fig1.update_layout(
        barmode="group",
        xaxis_title="Hora de salida", yaxis_title="Nº expediciones",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430, margin=dict(l=40, r=20, t=10, b=60),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10), tickvals=x_lbl[::step]),
        hovermode="x unified",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 1b: Difference ─────────────────────────────────────────────────
    if len(binned) == 2:
        st.subheader("Diferencia (Anteproyecto − Actual)")
        diff_vals = binned["anteproyecto"].values - binned["current"].values
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=x_lbl, y=diff_vals,
            marker_color=[COLORS["diff_pos"] if v >= 0 else COLORS["diff_neg"] for v in diff_vals],
            hovertemplate="%{x}: %{y:+d}<extra></extra>",
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")
        fig2.update_layout(
            xaxis_title="Hora", yaxis_title="Δ expediciones",
            height=300, margin=dict(l=40, r=20, t=10, b=60),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), tickvals=x_lbl[::step]),
            showlegend=False, hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 1c: Cumulative ─────────────────────────────────────────────────
    st.subheader("Expediciones acumuladas")
    fig3 = go.Figure()
    for key, color, dash in [("current", COLORS["current"], "solid"), ("anteproyecto", COLORS["anteproyecto"], "dash")]:
        if key not in binned:
            continue
        fig3.add_trace(go.Scatter(
            x=x_lbl, y=binned[key].cumsum().values,
            mode="lines", name=key.capitalize(),
            line=dict(color=color, width=2.5, dash=dash),
        ))
    fig3.update_layout(
        xaxis_title="Hora", yaxis_title="Acumulado",
        height=300, margin=dict(l=40, r=20, t=10, b=60),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10), tickvals=x_lbl[::step]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── First & Last 15 expeditions ───────────────────────────────────────────
    st.subheader("🕐 Primeras y últimas 15 expediciones")

    N = 15

    def build_first_last(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """First of day = dep >= 04:00 (240 min), sorted asc, first n.
        Last of day  = post-midnight services may be stored as either:
          - minutes > 1440  (e.g. 24:00 = 1440, 25:00 = 1500) — Visum current style
          - minutes 0-239   (e.g. 00:00, 01:00)               — Visum anteproyecto style
        We create a sort key that wraps 0-239 to 1440+ so they always rank AFTER
        the regular day services, then take the last n by that key.
        """
        cols = ["dep_min", "LineName"]
        if "arr_min"       in df.columns: cols.append("arr_min")
        if "travel_min"    in df.columns: cols.append("travel_min")
        if "DirectionCode" in df.columns: cols.append("DirectionCode")
        DAY_START = 240  # 04:00

        work = df.copy()
        # sort_key_overnight handles both Visum encoding styles:
        #   extended (24:10 = 1450 min) — already > 1440, no change needed
        #   wrapped  (00:10 =   10 min) — add 1440 so it sorts after 23:59
        work["_sort_key"] = work["dep_min"].apply(sort_key_overnight)

        # First of day: dep >= 04:00 ascending
        day_df = work[work["dep_min"] >= DAY_START].sort_values("_sort_key").reset_index(drop=True)
        first  = day_df.head(n)[cols].copy()

        # Last of day: all services by sort key, take last n
        all_sorted = work.sort_values("_sort_key").reset_index(drop=True)
        last = all_sorted.tail(n)[cols].copy()
        return first, last

    def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Salida"] = out["dep_min"].apply(minutes_to_hhmm)
        if "arr_min"    in out.columns: out["Llegada"]       = out["arr_min"].apply(minutes_to_hhmm)
        if "travel_min" in out.columns: out["Viaje (min)"]   = out["travel_min"].round(1)
        if "DirectionCode" in out.columns: out["Dir"] = out["DirectionCode"].astype(int)
        rename = {"LineName": "Línea"}
        drop   = ["dep_min", "arr_min", "travel_min", "DirectionCode"]
        return out.rename(columns=rename).drop(columns=[c for c in drop if c in out.columns])

    # Only show if at least one scenario loaded
    if filtered:
        # ── KPI strip: earliest & latest departure per scenario ───────────────
        kpi_fl = st.columns(len(filtered) * 2)
        col_idx = 0
        for key, df in filtered.items():
            label = "Actual" if key == "current" else "Anteproyecto"
            color_tag = "🔵" if key == "current" else "🟠"
            # 1ª expedición: first dep >= 04:00
            # última expedición: wrap post-midnight (< 04:00) to sort after 23:59
            day_deps  = df[df["dep_min"] >= 240]["dep_min"]
            earliest  = minutes_to_hhmm(day_deps.min()) if len(day_deps) else "–"
            sort_key   = df["dep_min"].apply(sort_key_overnight)
            latest_raw = df.loc[sort_key.idxmax(), "dep_min"]
            latest     = minutes_to_hhmm(latest_raw)  # % 1440 applied inside
            kpi_fl[col_idx].metric(f"{color_tag} {label} — 1ª expedición (≥04:00)", earliest)
            kpi_fl[col_idx + 1].metric(f"{color_tag} {label} — última expedición", latest)
            col_idx += 2

        # ── Side-by-side tables: FIRST 15 ─────────────────────────────────────
        st.markdown(f"#### 🟢 Primeras {N} expediciones del día")
        first_cols = st.columns(len(filtered))
        for i, (key, df) in enumerate(filtered.items()):
            label = "Actual" if key == "current" else "Anteproyecto"
            color = COLORS["current"] if key == "current" else COLORS["anteproyecto"]
            first_df, _ = build_first_last(df, N)
            first_cols[i].markdown(f"**{label}**")
            first_cols[i].dataframe(fmt_table(first_df), use_container_width=True, hide_index=True)

        # ── Side-by-side tables: LAST 15 ──────────────────────────────────────
        st.markdown(f"#### 🔴 Últimas {N} expediciones del día")
        last_cols = st.columns(len(filtered))
        for i, (key, df) in enumerate(filtered.items()):
            label = "Actual" if key == "current" else "Anteproyecto"
            _, last_df = build_first_last(df, N)
            last_cols[i].markdown(f"**{label}**")
            last_cols[i].dataframe(fmt_table(last_df), use_container_width=True, hide_index=True)

        # ── Visual comparison: departure times of first & last N ───────────────
        if len(filtered) == 2:
            st.markdown(f"#### Comparación visual — primeras y últimas {N} salidas")

            fig_fl = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f"Primeras {N}", f"Últimas {N}"),
            )
            all_first_x, all_last_x = [], []
            for key, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
                if key not in filtered:
                    continue
                label = "Actual" if key == "current" else "Anteproyecto"
                first_df, last_df = build_first_last(filtered[key], N)
                # Use sort_key_overnight so post-midnight last trips stay after 23:59 on axis
                last_x_num = last_df["dep_min"].apply(sort_key_overnight)
                all_first_x.extend(first_df["dep_min"].tolist())
                all_last_x.extend(last_x_num.tolist())

                fig_fl.add_trace(go.Scatter(
                    x=first_df["dep_min"],
                    y=list(range(1, N + 1)),
                    mode="markers+lines",
                    name=label,
                    marker=dict(color=color, size=8),
                    line=dict(color=color, width=1.5),
                    legendgroup=key,
                    showlegend=True,
                    hovertemplate="%{customdata}<br>Rango: %{y}<extra>" + label + "</extra>",
                    customdata=[minutes_to_hhmm(m) for m in first_df["dep_min"]],
                ), row=1, col=1)

                fig_fl.add_trace(go.Scatter(
                    x=last_x_num,
                    y=list(range(N, 0, -1)),
                    mode="markers+lines",
                    name=label,
                    marker=dict(color=color, size=8),
                    line=dict(color=color, width=1.5),
                    legendgroup=key,
                    showlegend=False,
                    hovertemplate="%{customdata}<br>Rango: %{y}<extra>" + label + "</extra>",
                    customdata=[minutes_to_hhmm(m) for m in last_x_num],
                ), row=1, col=2)

            def _ticks(vals):
                uniq  = sorted(set(vals))
                texts = [minutes_to_hhmm(v) for v in uniq]
                return uniq, texts
            tv1, tt1 = _ticks(all_first_x)
            tv2, tt2 = _ticks(all_last_x)
            fig_fl.update_yaxes(title_text="Ranking (1 = más temprana)", row=1, col=1)
            fig_fl.update_yaxes(title_text="Ranking (1 = más tardía)",   row=1, col=2)
            fig_fl.update_xaxes(tickangle=-45, tickvals=tv1, ticktext=tt1, row=1, col=1)
            fig_fl.update_xaxes(tickangle=-45, tickvals=tv2, ticktext=tt2, row=1, col=2)
            fig_fl.update_layout(
                height=400,
                margin=dict(l=40, r=20, t=40, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
            )
            st.plotly_chart(fig_fl, use_container_width=True)

            # ── Delta table: how much earlier/later are first & last ──────────
            st.markdown("#### Δ minutos entre escenarios — primera y última expedición")
            delta_rows = []
            keys_list = list(filtered.keys())
            def _sort_key(df):
                sk = df["dep_min"].apply(sort_key_overnight)
                return df.assign(_sk=sk).sort_values("_sk").drop(columns="_sk")
            df_a = _sort_key(filtered[keys_list[0]])
            df_b = _sort_key(filtered[keys_list[1]])

            for rank in range(1, N + 1):
                dep_a = df_a.iloc[rank - 1]["dep_min"]  if len(df_a) >= rank else np.nan
                dep_b = df_b.iloc[rank - 1]["dep_min"]  if len(df_b) >= rank else np.nan
                delta_rows.append({
                    "Rango": rank,
                    f"Salida {keys_list[0].capitalize()[:4]}": minutes_to_hhmm(dep_a) if not np.isnan(dep_a) else "–",
                    f"Salida {keys_list[1].capitalize()[:4]}": minutes_to_hhmm(dep_b) if not np.isnan(dep_b) else "–",
                    "Δ (min)": round(dep_b - dep_a, 1) if not (np.isnan(dep_a) or np.isnan(dep_b)) else "–",
                })
            delta_rows_last = []
            for rank in range(1, N + 1):
                dep_a = df_a.iloc[-rank]["dep_min"] if len(df_a) >= rank else np.nan
                dep_b = df_b.iloc[-rank]["dep_min"] if len(df_b) >= rank else np.nan
                delta_rows_last.append({
                    "Rango": rank,
                    f"Salida {keys_list[0].capitalize()[:4]}" : minutes_to_hhmm(dep_a) if not np.isnan(dep_a) else "–",
                    f"Salida {keys_list[1].capitalize()[:4]}": minutes_to_hhmm(dep_b) if not np.isnan(dep_b) else "–",
                    "Δ (min)": round(dep_b - dep_a, 1) if not (np.isnan(dep_a) or np.isnan(dep_b)) else "–",
                })

            tc1, tc2 = st.columns(2)
            tc1.markdown(f"**Primeras {N}**")
            tc1.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
            tc2.markdown(f"**Últimas {N}**")
            tc2.dataframe(pd.DataFrame(delta_rows_last), use_container_width=True, hide_index=True)

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("🗂️ Tabla de datos"):
        tdata = {"Hora": x_lbl}
        if "current"      in binned: tdata["Actual"]       = binned["current"].values
        if "anteproyecto" in binned: tdata["Anteproyecto"] = binned["anteproyecto"].values
        if len(binned) == 2:
            tdata["Δ"] = diff_vals
            tdata["% cambio"] = np.where(
                binned["current"].values > 0,
                diff_vals / binned["current"].values * 100, np.nan
            ).round(1)
        tdf = pd.DataFrame(tdata)
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV", tdf.to_csv(index=False).encode(), f"expediciones_{interval}min.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Comparación por línea
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Comparación por línea")

    # Build per-line summary
    line_rows = []
    for key, df in filtered.items():
        counts = df["LineName"].value_counts().rename("count")
        for line, cnt in counts.items():
            row = {"scenario": key, "line": line, "count": cnt}
            if "travel_min" in df.columns:
                ldf = df[df["LineName"] == line]
                row["avg_travel_min"] = ldf["travel_min"].mean()
            if "length_km" in df.columns:
                ldf = df[df["LineName"] == line]
                row["avg_length_km"]  = ldf["length_km"].mean()
            line_rows.append(row)

    if not line_rows:
        st.warning("No hay datos de líneas disponibles.")
    else:
        ldf_all = pd.DataFrame(line_rows)

        # Pivot for comparison
        pivot = ldf_all.pivot_table(index="line", columns="scenario", values="count", fill_value=0)
        pivot.columns.name = None
        if "current" in pivot.columns and "anteproyecto" in pivot.columns:
            pivot["delta"]   = pivot["anteproyecto"] - pivot["current"]
            pivot["pct_chg"] = (pivot["delta"] / pivot["current"].replace(0, np.nan) * 100).round(1)
        pivot = pivot.sort_values("delta" if "delta" in pivot.columns else pivot.columns[0], ascending=False)

        # ── Select individual line ────────────────────────────────────────────
        line_options = sorted(pivot.index.tolist(), key=str)
        selected_line = st.selectbox("Selecciona una línea para detalle", options=["(todas)"] + line_options)

        # ── Heatmap / bar: top N lines by absolute delta ─────────────────────
        st.markdown("#### Top 20 líneas por variación absoluta")
        top20 = pivot.nlargest(20, "delta") if "delta" in pivot.columns else pivot.head(20)

        fig_lines = go.Figure()
        if "current" in top20.columns:
            fig_lines.add_trace(go.Bar(
                y=top20.index.astype(str), x=top20["current"],
                name="Actual", orientation="h",
                marker_color=COLORS["current"], opacity=0.8,
            ))
        if "anteproyecto" in top20.columns:
            fig_lines.add_trace(go.Bar(
                y=top20.index.astype(str), x=top20["anteproyecto"],
                name="Anteproyecto", orientation="h",
                marker_color=COLORS["anteproyecto"], opacity=0.8,
            ))
        fig_lines.update_layout(
            barmode="group", height=520,
            xaxis_title="Nº expediciones", yaxis_title="Línea",
            margin=dict(l=80, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_lines, use_container_width=True)

        # ── Delta waterfall for top 20 ────────────────────────────────────────
        if "delta" in pivot.columns:
            st.markdown("#### Δ expediciones por línea (top 20 ganadores + perdedores)")
            top10_gain = pivot.nlargest(10, "delta")
            top10_loss = pivot.nsmallest(10, "delta")
            wf = pd.concat([top10_gain, top10_loss]).sort_values("delta", ascending=True)
            fig_wf = go.Figure(go.Bar(
                y=wf.index.astype(str), x=wf["delta"],
                orientation="h",
                marker_color=[COLORS["diff_pos"] if v >= 0 else COLORS["diff_neg"] for v in wf["delta"]],
            ))
            fig_wf.add_vline(x=0, line_dash="dash", line_color="grey")
            fig_wf.update_layout(
                height=460, xaxis_title="Δ expediciones", yaxis_title="Línea",
                margin=dict(l=80, r=20, t=10, b=40), showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        # ── Per-line temporal profile ─────────────────────────────────────────
        if selected_line != "(todas)":
            st.markdown(f"#### Perfil temporal — línea **{selected_line}**")
            binned_line = {}
            for key, df in dfs.items():
                ldf = df[(df["LineName"] == selected_line) & df["dep_min"].between(t_start, t_end)]
                if len(ldf):
                    binned_line[key] = bin_journeys(ldf, interval, t_start, t_end)

            if binned_line:
                _ref_l = binned_line.get("current") or next(iter(binned_line.values()))
                xl_lbl = make_x_labels(_ref_l)
                stepl  = tick_step(interval)

                fig_line_t = go.Figure()
                for key, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
                    if key not in binned_line:
                        continue
                    y = binned_line[key].values
                    fig_line_t.add_trace(go.Bar(
                        x=xl_lbl, y=y, name=key.capitalize(),
                        marker_color=color, opacity=0.6,
                    ))
                    fig_line_t.add_trace(go.Scatter(
                        x=xl_lbl, y=y, mode="lines+markers",
                        name=f"{key.capitalize()} (línea)",
                        line=dict(color=color, width=2), marker=dict(size=4),
                    ))
                fig_line_t.update_layout(
                    barmode="group", height=380,
                    xaxis_title="Hora", yaxis_title="Expediciones",
                    margin=dict(l=40, r=20, t=10, b=60),
                    xaxis=dict(tickangle=-45, tickvals=xl_lbl[::stepl]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_line_t, use_container_width=True)

                # Direction split
                if "DirectionCode" in dfs.get("current", pd.DataFrame()).columns or \
                   "DirectionCode" in dfs.get("anteproyecto", pd.DataFrame()).columns:
                    st.markdown(f"##### Desglose por dirección — {selected_line}")
                    dir_cols = st.columns(2)
                    for i, (key, color) in enumerate([("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]):
                        if key not in dfs or "DirectionCode" not in dfs[key].columns:
                            continue
                        ldf = dfs[key][(dfs[key]["LineName"] == selected_line) & dfs[key]["dep_min"].between(t_start, t_end)]
                        fig_dir = go.Figure()
                        for dir_val, dash in [(0, "solid"), (1, "dash")]:
                            sub = ldf[ldf["DirectionCode"] == dir_val]
                            if len(sub):
                                bs = bin_journeys(sub, interval, t_start, t_end)
                                fig_dir.add_trace(go.Scatter(
                                    x=make_x_labels(bs), y=bs.values,
                                    mode="lines", name=f"Dir {dir_val}",
                                    line=dict(color=color, dash=dash, width=2),
                                ))
                        fig_dir.update_layout(
                            title=key.capitalize(), height=260,
                            xaxis=dict(tickangle=-45, tickvals=make_x_labels(binned_line[key])[::stepl]),
                            margin=dict(l=30, r=10, t=30, b=50), hovermode="x unified",
                        )
                        dir_cols[i].plotly_chart(fig_dir, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        with st.expander("🗂️ Tabla resumen por línea"):
            disp = pivot.reset_index().rename(columns={"line": "Línea", "current": "Actual", "anteproyecto": "Anteproyecto", "delta": "Δ", "pct_chg": "% cambio"})
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.download_button("⬇️ CSV líneas", disp.to_csv(index=False).encode(), "lineas_comparacion.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Análisis avanzado (10 charts)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📐 Análisis avanzado")

    has_travel = all("travel_min" in df.columns for df in filtered.values())
    has_length = all("length_km" in df.columns for df in filtered.values())

    # ── Chart A3: Scatter — hora salida vs tiempo de viaje ───────────────────
    if has_travel:
        st.markdown("### A3 · Hora de salida vs tiempo de viaje")
        fig_sc = go.Figure()
        for key, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
            if key not in filtered: continue
            df = filtered[key].sample(min(2000, len(filtered[key])), random_state=42)
            fig_sc.add_trace(go.Scatter(
                x=df["dep_min"],
                y=df["travel_min"],
                mode="markers", name=key.capitalize(),
                marker=dict(color=color, size=4, opacity=0.5),
                hovertemplate="Salida: %{customdata}<br>Viaje: %{y:.1f} min<extra></extra>",
                customdata=[minutes_to_hhmm(m) for m in df["dep_min"]],
            ))
        hv3, ht3 = hour_ticks(t_start, t_end)
        fig_sc.update_layout(height=380, xaxis_title="Hora de salida",
                             yaxis_title="Tiempo de viaje (min)",
                             margin=dict(l=40, r=20, t=10, b=60),
                             xaxis=dict(tickangle=-45, tickvals=hv3, ticktext=ht3))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Chart A4: Histograma de tiempo de viaje ──────────────────────────────
    if has_travel:
        st.markdown("### A4 · Histograma de tiempo de viaje")
        fig_hist = go.Figure()
        for key, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
            if key not in filtered: continue
            fig_hist.add_trace(go.Histogram(
                x=filtered[key]["travel_min"], name=key.capitalize(),
                marker_color=color, opacity=0.6, nbinsx=40,
                histnorm="percent",
            ))
        fig_hist.update_layout(barmode="overlay", height=340,
                               xaxis_title="Tiempo de viaje (min)", yaxis_title="% expediciones",
                               margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Chart A6: Nº de líneas activas por hora ──────────────────────────────
    st.markdown("### A6 · Líneas únicas activas por hora")
    fig_uniq = go.Figure()
    for key, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
        if key not in filtered: continue
        df = filtered[key].copy()
        df["hour_bin"] = (df["dep_min"] // 60) * 60
        active = df.groupby("hour_bin")["LineName"].nunique().reset_index()
        fig_uniq.add_trace(go.Scatter(
            x=active["hour_bin"], y=active["LineName"],
            mode="lines+markers", name=key.capitalize(),
            line=dict(color=color, width=2.5),
            hovertemplate="%{customdata}<br>Líneas: %{y}<extra></extra>",
            customdata=[minutes_to_hhmm(v) for v in active["hour_bin"]],
        ))
    hv6, ht6 = hour_ticks(t_start, t_end)
    fig_uniq.update_layout(height=320, xaxis_title="Hora", yaxis_title="Líneas únicas activas",
                           margin=dict(l=40, r=20, t=10, b=60),
                           xaxis=dict(tickangle=-45, tickvals=hv6, ticktext=ht6),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_uniq, use_container_width=True)

    # ── Chart A7: Headway por línea × dirección ───────────────────────────────
    st.markdown("### A7 · Headway medio por línea × dirección (top 20 por variación)")
    st.caption(
        "Headway = (última salida − primera salida) / (n exp. − 1) por línea y dirección. "
        "Sólo se usan expediciones diurnas (≥04:00). Ponderado por número de expediciones."
    )
    hw_rows_a7 = []
    for key, df in filtered.items():
        hw_df_a7 = compute_headway(df)
        for _, row in hw_df_a7.iterrows():
            dirn_label = "" if np.isnan(row["direction"]) else f" Dir{int(row['direction'])}"
            hw_rows_a7.append({
                "scenario":    key,
                "line_dir":    f"{row['line']}{dirn_label}",
                "line":        row["line"],
                "direction":   row["direction"],
                "headway_min": row["headway_min"],
                "n_trips":     row["n_trips"],
                "first_dep":   minutes_to_hhmm(row["first_dep"]),
                "last_dep":    minutes_to_hhmm(row["last_dep"]),
            })

    if hw_rows_a7:
        hw_df7     = pd.DataFrame(hw_rows_a7)
        hw_pivot7  = hw_df7.pivot_table(index="line_dir", columns="scenario", values="headway_min")
        hw_pivot7.columns.name = None
        if "current" in hw_pivot7 and "anteproyecto" in hw_pivot7:
            hw_pivot7["delta_hw"] = hw_pivot7["anteproyecto"] - hw_pivot7["current"]
            top20_hw = hw_pivot7.dropna().reindex(
                hw_pivot7.dropna()["delta_hw"].abs().nlargest(20).index
            )
        else:
            top20_hw = hw_pivot7.head(20)

        fig_hw = go.Figure()
        for col, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
            if col in top20_hw.columns:
                fig_hw.add_trace(go.Bar(
                    y=top20_hw.index.astype(str), x=top20_hw[col],
                    name=("Actual" if col=="current" else "Anteproyecto"),
                    orientation="h", marker_color=color, opacity=0.8,
                ))
        if "delta_hw" in top20_hw.columns:
            fig_hw.add_trace(go.Scatter(
                y=top20_hw.index.astype(str), x=top20_hw["delta_hw"],
                name="Δ (Ante−Act)", mode="markers",
                marker=dict(
                    symbol="diamond", size=9,
                    color=[COLORS["diff_pos"] if v>=0 else COLORS["diff_neg"] for v in top20_hw["delta_hw"]],
                ),
            ))
        fig_hw.update_layout(
            barmode="group", height=max(400, len(top20_hw)*28),
            xaxis_title="Headway medio (min)", yaxis_title="Línea × Dirección",
            margin=dict(l=110, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_hw, use_container_width=True)

        # Detail table with first/last dep and n_trips per line×dir
        with st.expander("🗂️ Detalle headway por línea × dirección"):
            hw_detail = hw_df7.pivot_table(
                index="line_dir", columns="scenario",
                values=["headway_min","n_trips","first_dep","last_dep"],
                aggfunc="first",
            )
            hw_detail.columns = [f"{col[0]}_{col[1]}" for col in hw_detail.columns]
            hw_detail = hw_detail.reset_index()
            st.dataframe(hw_detail, use_container_width=True, hide_index=True)
            st.download_button("⬇️ CSV headway",
                               hw_detail.to_csv(index=False).encode(),
                               "headway_linea_dir.csv", "text/csv")

    # ── Route count info ──────────────────────────────────────────────────────
    if any("RouteNameUnique" in filtered[k].columns for k in filtered):
        st.markdown("### A7b · Número de rutas por línea")
        st.caption(
            "Una *línea* puede operar varias *rutas* (variantes de recorrido). "
            "Este gráfico muestra cuántas rutas distintas existen por línea en cada escenario."
        )
        rte_rows = []
        for key, df in filtered.items():
            if "RouteNameUnique" not in df.columns: continue
            for line, grp in df.groupby("LineName"):
                rte_rows.append({
                    "scenario": key,
                    "line":     str(line),
                    "n_routes": grp["RouteNameUnique"].nunique(),
                })
        if rte_rows:
            rte_df  = pd.DataFrame(rte_rows)
            rte_piv = rte_df.pivot_table(index="line", columns="scenario", values="n_routes", fill_value=0)
            rte_piv.columns.name = None
            if "current" in rte_piv and "anteproyecto" in rte_piv:
                rte_piv["delta_r"] = rte_piv["anteproyecto"] - rte_piv["current"]
                rte_top = rte_piv.reindex(rte_piv["delta_r"].abs().nlargest(20).index)
            else:
                rte_top = rte_piv.head(20)
            fig_rte = go.Figure()
            for col, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if col in rte_top.columns:
                    fig_rte.add_trace(go.Bar(
                        y=rte_top.index.astype(str), x=rte_top[col],
                        name=("Actual" if col=="current" else "Anteproyecto"),
                        orientation="h", marker_color=color, opacity=0.8,
                    ))
            fig_rte.update_layout(
                barmode="group", height=max(350, len(rte_top)*28),
                xaxis_title="Nº de rutas", yaxis_title="Línea",
                margin=dict(l=80,r=20,t=10,b=40),
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
            )
            st.plotly_chart(fig_rte, use_container_width=True)

    # ── Chart A8: Longitud media de viaje por línea ───────────────────────────
    if has_length:
        st.markdown("### A8 · Longitud media (km) por línea — top 20")
        len_rows = []
        for key, df in filtered.items():
            for line, grp in df.groupby("LineName"):
                len_rows.append({"scenario": key, "line": str(line), "avg_km": grp["length_km"].mean()})
        len_df = pd.DataFrame(len_rows)
        len_pivot = len_df.pivot_table(index="line", columns="scenario", values="avg_km")
        len_pivot.columns.name = None
        top20_len = len_pivot.dropna().assign(
            mean=lambda x: x.mean(axis=1)
        ).nlargest(20, "mean").drop(columns="mean")

        fig_len = go.Figure()
        for col, color in [("current", COLORS["current"]), ("anteproyecto", COLORS["anteproyecto"])]:
            if col in top20_len.columns:
                fig_len.add_trace(go.Bar(
                    y=top20_len.index.astype(str), x=top20_len[col],
                    name=col.capitalize(), orientation="h",
                    marker_color=color, opacity=0.8,
                ))
        fig_len.update_layout(barmode="group", height=500,
                              xaxis_title="Longitud media (km)", yaxis_title="Línea",
                              margin=dict(l=80, r=20, t=10, b=40),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_len, use_container_width=True)

    # ── Chart A9: Expediciones nuevas / eliminadas / mantenidas ─────────────
    if len(filtered) == 2:
        st.markdown("### A9 · Líneas nuevas, eliminadas y modificadas")
        lines_cur  = set(filtered["current"]["LineName"].dropna().unique())
        lines_ante = set(filtered["anteproyecto"]["LineName"].dropna().unique())
        new_lines  = sorted(lines_ante - lines_cur,  key=str)
        del_lines  = sorted(lines_cur  - lines_ante, key=str)
        keep_lines = sorted(lines_cur  & lines_ante, key=str)

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Líneas nuevas",     len(new_lines))
        c2.metric("🔴 Líneas eliminadas", len(del_lines))
        c3.metric("🔵 Líneas comunes",    len(keep_lines))

        fig_pie = go.Figure(go.Pie(
            labels=["Nuevas", "Eliminadas", "Mantenidas"],
            values=[len(new_lines), len(del_lines), len(keep_lines)],
            marker_colors=[COLORS["diff_pos"], COLORS["diff_neg"], COLORS["current"]],
            hole=0.4,
        ))
        fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

        if new_lines or del_lines:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Líneas nuevas:**")
                st.write(", ".join(str(l) for l in new_lines) or "—")
            with col_b:
                st.markdown("**Líneas eliminadas:**")
                st.write(", ".join(str(l) for l in del_lines) or "—")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Longitud & Duración
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("📏 Análisis de Longitud y Duración")

    has_len4 = any("length_km"    in df.columns for df in filtered.values())
    has_dur4 = any("duration_min" in df.columns for df in filtered.values())

    if not has_len4 and not has_dur4:
        st.warning("No se encontraron columnas 'Length' ni 'Duration' en los archivos cargados.")
    else:
        hv4, ht4 = hour_ticks(t_start, t_end)

        # ── Transport-planner KPI panel ───────────────────────────────────────
        st.markdown("### 📌 Indicadores de oferta")

        s_stats = {}
        for key, df in filtered.items():
            s = {}
            if "length_km" in df.columns:
                s["total_km"]  = df["length_km"].sum()
                s["avg_km"]    = df["length_km"].mean()
            if "duration_min" in df.columns:
                s["avg_dur"]     = df["duration_min"].mean()
                s["total_dur_h"] = df["duration_min"].sum() / 60
            if "length_km" in df.columns and "duration_min" in df.columns:
                df_sp = df.dropna(subset=["length_km","duration_min"])
                df_sp = df_sp[df_sp["duration_min"] > 0]
                if len(df_sp):
                    spd = df_sp["length_km"] / (df_sp["duration_min"] / 60)
                    s["avg_speed"]    = spd.mean()
                    s["p10_speed"]    = spd.quantile(0.10)
                    s["p90_speed"]    = spd.quantile(0.90)
                    s["total_veh_km"] = df_sp["length_km"].sum()
                    s["total_veh_h"]  = df_sp["duration_min"].sum() / 60
            s_stats[key] = s

        kpi_A = st.columns(4)
        kpi_B = st.columns(4)
        kpi_C = st.columns(4)

        for ci, (key, lbl) in enumerate([("current","Actual"),("anteproyecto","Anteproyecto")]):
            if key not in s_stats: continue
            s   = s_stats[key]
            tag = "🔵" if key == "current" else "🟠"
            if "total_km"     in s: kpi_A[ci  ].metric(f"{tag} Km totales — {lbl}",          f"{s['total_km']:,.0f} km")
            if "avg_km"       in s: kpi_A[ci+2].metric(f"{tag} Km medios/exp. — {lbl}",      f"{s['avg_km']:.2f} km")
            if "avg_speed"    in s: kpi_B[ci  ].metric(f"{tag} Vel. comercial — {lbl}",      f"{s['avg_speed']:.1f} km/h")
            if "avg_dur"      in s: kpi_B[ci+2].metric(f"{tag} Duración media — {lbl}",      f"{s['avg_dur']:.1f} min")
            if "total_veh_h"  in s: kpi_C[ci  ].metric(f"{tag} Horas-vehículo — {lbl}",     f"{s['total_veh_h']:,.0f} h")
            if "total_veh_km" in s: kpi_C[ci+2].metric(f"{tag} Km-vehículo totales — {lbl}",f"{s['total_veh_km']:,.0f} km")

        if len(filtered) == 2:
            keys2  = list(s_stats.keys())
            sa, sb = s_stats.get(keys2[0],{}), s_stats.get(keys2[1],{})
            kpi_D  = st.columns(4)
            if "total_km"    in sa and "total_km"    in sb:
                d = sb["total_km"] - sa["total_km"]
                kpi_D[0].metric("Δ Km totales", f"{d:+,.0f} km",
                                delta=f"{d/sa['total_km']*100:+.1f}%" if sa["total_km"] else None)
            if "avg_speed"   in sa and "avg_speed"   in sb:
                kpi_D[1].metric("Δ Vel. media", f"{sb['avg_speed']-sa['avg_speed']:+.1f} km/h")
            if "avg_dur"     in sa and "avg_dur"     in sb:
                d = sb["avg_dur"] - sa["avg_dur"]
                kpi_D[2].metric("Δ Duración media", f"{d:+.1f} min", delta_color="inverse")
            if "total_veh_h" in sa and "total_veh_h" in sb:
                d = sb["total_veh_h"] - sa["total_veh_h"]
                kpi_D[3].metric("Δ Horas-vehículo", f"{d:+,.0f} h",
                                delta=f"{d/sa['total_veh_h']*100:+.1f}%" if sa.get("total_veh_h") else None)
            if "p10_speed" in sa and "p10_speed" in sb:
                st.caption(
                    f"Rango velocidad (P10–P90) — "
                    f"Actual: {sa['p10_speed']:.1f}–{sa['p90_speed']:.1f} km/h  |  "
                    f"Anteproyecto: {sb['p10_speed']:.1f}–{sb['p90_speed']:.1f} km/h"
                )

        st.divider()

        # ── L1: Histograma longitud ───────────────────────────────────────────
        if has_len4:
            st.markdown("### L1 · Distribución de longitud por expedición")
            fig_lh = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered or "length_km" not in filtered[key].columns: continue
                fig_lh.add_trace(go.Histogram(
                    x=filtered[key]["length_km"],
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    marker_color=color, opacity=0.6, nbinsx=50, histnorm="percent",
                ))
            fig_lh.update_layout(barmode="overlay", height=340,
                                  xaxis_title="Longitud (km)", yaxis_title="% expediciones",
                                  margin=dict(l=40,r=20,t=10,b=40),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_lh, use_container_width=True)

        # ── L2: Box-plot longitud ─────────────────────────────────────────────
        if has_len4:
            st.markdown("### L2 · Box-plot de longitud")
            fig_lb = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered or "length_km" not in filtered[key].columns: continue
                fig_lb.add_trace(go.Box(
                    y=filtered[key]["length_km"],
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    marker_color=color, boxmean="sd",
                ))
            fig_lb.update_layout(height=380, yaxis_title="Longitud (km)",
                                  margin=dict(l=40,r=20,t=10,b=40))
            st.plotly_chart(fig_lb, use_container_width=True)

        # ── L3: Top 20 líneas — longitud media ────────────────────────────────
        len_r2 = []
        if has_len4:
            st.markdown("### L3 · Longitud media por línea — top 20 por variación")
            for key, df in filtered.items():
                if "length_km" not in df.columns: continue
                for line, grp in df.groupby("LineName"):
                    len_r2.append({"scenario":key,"line":str(line),"avg_km":grp["length_km"].mean(),"total_km":grp["length_km"].sum()})
            if len_r2:
                lp2 = pd.DataFrame(len_r2).pivot_table(index="line",columns="scenario",values="avg_km")
                lp2.columns.name = None
                if "current" in lp2.columns and "anteproyecto" in lp2.columns:
                    lp2["delta_km"] = lp2["anteproyecto"] - lp2["current"]
                    top20_l = lp2.dropna().reindex(lp2.dropna()["delta_km"].abs().nlargest(20).index)
                else:
                    top20_l = lp2.head(20)
                fig_l3 = go.Figure()
                for col, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                    if col in top20_l.columns:
                        fig_l3.add_trace(go.Bar(y=top20_l.index.astype(str), x=top20_l[col],
                                                name=("Actual" if col=="current" else "Anteproyecto"),
                                                orientation="h", marker_color=color, opacity=0.8))
                fig_l3.update_layout(barmode="group", height=520,
                                      xaxis_title="Longitud media (km)", yaxis_title="Línea",
                                      margin=dict(l=80,r=20,t=10,b=40),
                                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                st.plotly_chart(fig_l3, use_container_width=True)

        # ── L4: Waterfall Δ longitud ──────────────────────────────────────────
        if has_len4 and len(filtered) == 2 and len_r2:
            st.markdown("### L4 · Δ longitud media — top 10 ganadores + perdedores")
            lp3 = pd.DataFrame(len_r2).pivot_table(index="line",columns="scenario",values="avg_km")
            lp3.columns.name = None
            if "current" in lp3.columns and "anteproyecto" in lp3.columns:
                lp3["delta_km"] = lp3["anteproyecto"] - lp3["current"]
                wfl = pd.concat([lp3.dropna().nlargest(10,"delta_km"),
                                  lp3.dropna().nsmallest(10,"delta_km")]).sort_values("delta_km",ascending=True)
                fig_l4 = go.Figure(go.Bar(
                    y=wfl.index.astype(str), x=wfl["delta_km"], orientation="h",
                    marker_color=[COLORS["diff_pos"] if v>=0 else COLORS["diff_neg"] for v in wfl["delta_km"]],
                    hovertemplate="%{y}: %{x:+.2f} km<extra></extra>",
                ))
                fig_l4.add_vline(x=0, line_dash="dash", line_color="grey")
                fig_l4.update_layout(height=460, xaxis_title="Δ longitud media (km)", yaxis_title="Línea",
                                      margin=dict(l=80,r=20,t=10,b=40), showlegend=False)
                st.plotly_chart(fig_l4, use_container_width=True)

        # ── L5: Scatter longitud vs hora ──────────────────────────────────────
        if has_len4:
            st.markdown("### L5 · Longitud vs hora de salida")
            fig_ls = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered or "length_km" not in filtered[key].columns: continue
                df_s = filtered[key].sample(min(3000,len(filtered[key])),random_state=42)
                fig_ls.add_trace(go.Scatter(
                    x=df_s["dep_min"], y=df_s["length_km"], mode="markers",
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    marker=dict(color=color,size=4,opacity=0.45),
                    hovertemplate="Salida: %{customdata}<br>Long.: %{y:.1f} km<extra></extra>",
                    customdata=[minutes_to_hhmm(m) for m in df_s["dep_min"]],
                ))
            fig_ls.update_layout(height=380, xaxis_title="Hora de salida", yaxis_title="Longitud (km)",
                                  margin=dict(l=40,r=20,t=10,b=60),
                                  xaxis=dict(tickangle=-45,tickvals=hv4,ticktext=ht4),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_ls, use_container_width=True)

        # ── L6: Longitud media horaria ────────────────────────────────────────
        if has_len4:
            st.markdown("### L6 · Longitud media por franja horaria")
            fig_lt = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered or "length_km" not in filtered[key].columns: continue
                df_h = filtered[key].copy()
                df_h["hour_bin"] = (df_h["dep_min"]//60)*60
                agg = df_h.groupby("hour_bin")["length_km"].mean().reset_index()
                fig_lt.add_trace(go.Scatter(
                    x=agg["hour_bin"], y=agg["length_km"], mode="lines+markers",
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    line=dict(color=color,width=2.5), marker=dict(size=7),
                    hovertemplate="%{customdata}<br>Media: %{y:.2f} km<extra></extra>",
                    customdata=[minutes_to_hhmm(v) for v in agg["hour_bin"]],
                ))
            fig_lt.update_layout(height=320, xaxis_title="Franja horaria", yaxis_title="Longitud media (km)",
                                  margin=dict(l=40,r=20,t=10,b=60),
                                  xaxis=dict(tickangle=-45,tickvals=hv4,ticktext=ht4),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                                  hovermode="x unified")
            st.plotly_chart(fig_lt, use_container_width=True)

        # ── L7: Km-vehículo por hora ──────────────────────────────────────────
        if has_len4:
            st.markdown("### L7 · Km-vehículo ofertados por franja horaria")
            fig_kv = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered or "length_km" not in filtered[key].columns: continue
                df_h = filtered[key].copy()
                df_h["hour_bin"] = (df_h["dep_min"]//60)*60
                agg = df_h.groupby("hour_bin")["length_km"].sum().reset_index()
                fig_kv.add_trace(go.Bar(
                    x=agg["hour_bin"], y=agg["length_km"],
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    marker_color=color, opacity=0.7,
                    hovertemplate="%{customdata}<br>Km: %{y:,.0f}<extra></extra>",
                    customdata=[minutes_to_hhmm(v) for v in agg["hour_bin"]],
                ))
            fig_kv.update_layout(barmode="group", height=340,
                                  xaxis_title="Franja horaria", yaxis_title="Km-vehículo",
                                  margin=dict(l=40,r=20,t=10,b=60),
                                  xaxis=dict(tickangle=-45,tickvals=hv4,ticktext=ht4),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                                  hovermode="x unified")
            st.plotly_chart(fig_kv, use_container_width=True)

        st.divider()

        # ── D1: Histograma duración ───────────────────────────────────────────
        if has_dur4:
            st.markdown("### D1 · Distribución de duración por expedición")
            fig_dh = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered or "duration_min" not in filtered[key].columns: continue
                fig_dh.add_trace(go.Histogram(
                    x=filtered[key]["duration_min"],
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    marker_color=color, opacity=0.6, nbinsx=50, histnorm="percent",
                ))
            fig_dh.update_layout(barmode="overlay", height=340,
                                  xaxis_title="Duración (min)", yaxis_title="% expediciones",
                                  margin=dict(l=40,r=20,t=10,b=40),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_dh, use_container_width=True)

        # ── D2: Duración media por línea ──────────────────────────────────────
        if has_dur4:
            st.markdown("### D2 · Duración media por línea — top 20 por variación")
            dur_r = []
            for key, df in filtered.items():
                if "duration_min" not in df.columns: continue
                for line, grp in df.groupby("LineName"):
                    dur_r.append({"scenario":key,"line":str(line),"avg_dur":grp["duration_min"].mean()})
            if dur_r:
                dp = pd.DataFrame(dur_r).pivot_table(index="line",columns="scenario",values="avg_dur")
                dp.columns.name = None
                if "current" in dp.columns and "anteproyecto" in dp.columns:
                    dp["delta_dur"] = dp["anteproyecto"] - dp["current"]
                    top20_d = dp.dropna().reindex(dp.dropna()["delta_dur"].abs().nlargest(20).index)
                else:
                    top20_d = dp.head(20)
                fig_d2 = go.Figure()
                for col, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                    if col in top20_d.columns:
                        fig_d2.add_trace(go.Bar(y=top20_d.index.astype(str), x=top20_d[col],
                                                name=("Actual" if col=="current" else "Anteproyecto"),
                                                orientation="h", marker_color=color, opacity=0.8))
                fig_d2.update_layout(barmode="group", height=520,
                                      xaxis_title="Duración media (min)", yaxis_title="Línea",
                                      margin=dict(l=80,r=20,t=10,b=40),
                                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                st.plotly_chart(fig_d2, use_container_width=True)

        st.divider()

        # ── LD1: Scatter longitud vs duración (velocidad comercial) ──────────
        if has_len4 and has_dur4:
            st.markdown("### LD1 · Longitud vs Duración — velocidad comercial implícita (km/h)")
            fig_ld = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered: continue
                df_ld = filtered[key].dropna(subset=["length_km","duration_min"])
                df_ld = df_ld[df_ld["duration_min"]>0].sample(min(3000,len(df_ld)),random_state=42)
                spd   = df_ld["length_km"]/(df_ld["duration_min"]/60)
                last_k = list(filtered.keys())[-1]
                fig_ld.add_trace(go.Scatter(
                    x=df_ld["length_km"], y=df_ld["duration_min"], mode="markers",
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    marker=dict(color=spd, colorscale="RdYlGn", size=5, opacity=0.5,
                                colorbar=dict(title="km/h") if key==last_k else None,
                                showscale=(key==last_k)),
                    hovertemplate="Long.: %{x:.1f} km<br>Dur.: %{y:.1f} min<br>Vel.: %{customdata:.1f} km/h<extra></extra>",
                    customdata=spd,
                ))
            fig_ld.update_layout(height=420, xaxis_title="Longitud (km)", yaxis_title="Duración (min)",
                                  margin=dict(l=40,r=60,t=10,b=40),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_ld, use_container_width=True)

        # ── LD2: Velocidad media por línea ────────────────────────────────────
        if has_len4 and has_dur4:
            st.markdown("### LD2 · Velocidad comercial media (km/h) por línea — top 20")
            spd_r = []
            for key, df in filtered.items():
                if "length_km" not in df.columns or "duration_min" not in df.columns: continue
                df_sp = df.dropna(subset=["length_km","duration_min"])
                df_sp = df_sp[df_sp["duration_min"]>0].copy()
                df_sp["spd"] = df_sp["length_km"]/(df_sp["duration_min"]/60)
                for line, grp in df_sp.groupby("LineName"):
                    spd_r.append({"scenario":key,"line":str(line),"avg_spd":grp["spd"].mean()})
            if spd_r:
                sp = pd.DataFrame(spd_r).pivot_table(index="line",columns="scenario",values="avg_spd")
                sp.columns.name = None
                if "current" in sp.columns and "anteproyecto" in sp.columns:
                    sp["delta_spd"] = sp["anteproyecto"] - sp["current"]
                    top20_sp = sp.dropna().reindex(sp.dropna()["delta_spd"].abs().nlargest(20).index)
                else:
                    top20_sp = sp.head(20)
                fig_s2 = go.Figure()
                for col, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                    if col in top20_sp.columns:
                        fig_s2.add_trace(go.Bar(y=top20_sp.index.astype(str), x=top20_sp[col],
                                                name=("Actual" if col=="current" else "Anteproyecto"),
                                                orientation="h", marker_color=color, opacity=0.8))
                fig_s2.update_layout(barmode="group", height=520,
                                      xaxis_title="Velocidad media (km/h)", yaxis_title="Línea",
                                      margin=dict(l=80,r=20,t=10,b=40),
                                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                st.plotly_chart(fig_s2, use_container_width=True)

        # ── LD3: Velocidad por franja horaria ─────────────────────────────────
        if has_len4 and has_dur4:
            st.markdown("### LD3 · Velocidad comercial media por franja horaria")
            fig_sh = go.Figure()
            for key, color in [("current",COLORS["current"]),("anteproyecto",COLORS["anteproyecto"])]:
                if key not in filtered: continue
                df_sh = filtered[key].dropna(subset=["length_km","duration_min"])
                df_sh = df_sh[df_sh["duration_min"]>0].copy()
                df_sh["spd"]      = df_sh["length_km"]/(df_sh["duration_min"]/60)
                df_sh["hour_bin"] = (df_sh["dep_min"]//60)*60
                agg = df_sh.groupby("hour_bin")["spd"].mean().reset_index()
                fig_sh.add_trace(go.Scatter(
                    x=agg["hour_bin"], y=agg["spd"], mode="lines+markers",
                    name=("Actual" if key=="current" else "Anteproyecto"),
                    line=dict(color=color,width=2.5), marker=dict(size=7),
                    hovertemplate="%{customdata}<br>Vel.: %{y:.1f} km/h<extra></extra>",
                    customdata=[minutes_to_hhmm(v) for v in agg["hour_bin"]],
                ))
            fig_sh.update_layout(height=320, xaxis_title="Franja horaria", yaxis_title="Velocidad media (km/h)",
                                  margin=dict(l=40,r=20,t=10,b=60),
                                  xaxis=dict(tickangle=-45,tickvals=hv4,ticktext=ht4),
                                  legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                                  hovermode="x unified")
            st.plotly_chart(fig_sh, use_container_width=True)

        # ── Summary download ──────────────────────────────────────────────────
        with st.expander("🗂️ Tabla resumen — longitud, duración y velocidad por línea"):
            sum_r = []
            for key, df in filtered.items():
                label = "Actual" if key=="current" else "Anteproyecto"
                for line, grp in df.groupby("LineName"):
                    row = {"Escenario":label,"Línea":str(line),"Expediciones":len(grp)}
                    if "length_km"    in grp.columns:
                        row["Long. media (km)"]  = round(grp["length_km"].mean(),3)
                        row["Long. total (km)"]  = round(grp["length_km"].sum(),1)
                    if "duration_min" in grp.columns:
                        row["Duración media (min)"] = round(grp["duration_min"].mean(),2)
                    if "length_km" in grp.columns and "duration_min" in grp.columns:
                        dp2 = grp[grp["duration_min"]>0]
                        if len(dp2):
                            row["Velocidad media (km/h)"] = round(
                                (dp2["length_km"]/(dp2["duration_min"]/60)).mean(),2)
                    sum_r.append(row)
            if sum_r:
                sum_df = pd.DataFrame(sum_r).sort_values(["Línea","Escenario"])
                st.dataframe(sum_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ CSV longitud & duración",
                                   sum_df.to_csv(index=False).encode(),
                                   "longitud_duracion.csv","text/csv")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("App Visum PUTHPATHLEGS · columna Dep (hora de salida) · líneas normalizadas a formato A0651")