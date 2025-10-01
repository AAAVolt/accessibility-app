import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

st.set_page_config(
    page_title="Bus Stop Accessibility Analysis",
    layout="wide",
    page_icon="🚏"
)

# Title
st.title("🚏 Bus Stop Accessibility Analysis")
st.markdown("""
**Population-focused analysis of bus stop proximity**  
*Understanding how many residents have convenient access to public transport*
""")

# Methodology expander
with st.expander("📋 About This Analysis", expanded=False):
    st.markdown("""
    ### Data Structure:

    **Zone Information:**
    - `origin_id` → Unique zone identifier
    - `Núcleo` → Locality/settlement name
    - `Municipio` → Municipality name
    - `Pop` → Number of residents in the zone

    **Accessibility Metrics:**
    - `nearest_stop` → ID of closest bus stop
    - `dist_m` → Walking distance to nearest stop (meters)

    ### Accessibility Categories:
    Based on walking distance to nearest bus stop:
    - 🟢 **Excellent**: < 300m (≈4 min walk)
    - 🟡 **Good**: 300-500m (4-6 min walk)
    - 🟠 **Fair**: 500-1000m (6-12 min walk)
    - 🔴 **Poor**: > 1000m (>12 min walk)

    ### Key Calculations:

    1. **Population-weighted accessibility**: Each zone's population is used to calculate 
       what percentage of total residents have good vs poor access to bus stops.

    2. **Walking time estimates**: Based on average walking speed of 80m/min (4.8 km/h).

    ### Important Notes:
    - All percentages reflect actual population impact, not just geographic coverage
    - Priority areas combine poor accessibility with high population counts
    """)

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with bus stop data", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Calculate walking time (assuming 80m/min walking speed)
    df['walk_time_min'] = df['dist_m'] / 80

    # Calculate total population
    total_population = df['Pop'].sum()

    # Sidebar info
    st.sidebar.header("📊 Dataset Summary")
    st.sidebar.markdown(f"**Total zones:** {len(df):,}")
    st.sidebar.markdown(f"**Total population:** {total_population:,.0f} residents")
    st.sidebar.markdown(f"**Unique municipalities:** {df['Municipio'].nunique()}")
    st.sidebar.markdown(f"**Unique localities:** {df['Núcleo'].nunique()}")

    # Key metrics
    st.header("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Population-weighted average distance
        weighted_avg_distance = (df['dist_m'] * df['Pop']).sum() / total_population
        st.metric("Avg Distance to Stop", f"{weighted_avg_distance:.0f}m")

    with col2:
        # Population-weighted average walk time
        weighted_avg_walk_time = (df['walk_time_min'] * df['Pop']).sum() / total_population
        st.metric("Avg Walk Time", f"{weighted_avg_walk_time:.1f} min")

    with col3:
        pop_within_500m = df[df['dist_m'] <= 500]['Pop'].sum()
        pct_within_500m = (pop_within_500m / total_population * 100) if total_population > 0 else 0
        st.metric("Pop within 500m", f"{pct_within_500m:.1f}%")

    with col4:
        # Population-weighted median (approximation using cumulative sum)
        df_sorted = df.sort_values('dist_m')
        df_sorted['cumsum_pop'] = df_sorted['Pop'].cumsum()
        median_idx = (df_sorted['cumsum_pop'] >= total_population / 2).idxmax()
        pop_weighted_median = df_sorted.loc[median_idx, 'dist_m']
        st.metric("Median Distance (pop-weighted)", f"{pop_weighted_median:.0f}m")

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Accessibility Overview",
        "⚠️ Priority Areas",
        "🏛 Municipality Breakdown"
    ])

    # TAB 1: Accessibility Overview
    with tab1:
        st.subheader("Population Accessibility Distribution")
        st.markdown("*What percentage of the population has good vs poor access to bus stops?*")

        # Create accessibility categories
        df['Accessibility_Category'] = pd.cut(
            df['dist_m'],
            bins=[0, 300, 500, 1000, float('inf')],
            labels=['🟢 Excellent (<300m)', '🟡 Good (300-500m)',
                    '🟠 Fair (500-1000m)', '🔴 Poor (>1000m)']
        )

        # Calculate population-weighted percentages
        pop_by_category = df.groupby('Accessibility_Category')['Pop'].sum()
        category_pcts = (pop_by_category / total_population * 100) if total_population > 0 else pd.Series()

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart with population-weighted percentages
            categories = ['🟢 Excellent (<300m)', '🟡 Good (300-500m)',
                          '🟠 Fair (500-1000m)', '🔴 Poor (>1000m)']
            populations = [pop_by_category.get(cat, 0) for cat in categories]
            percentages = [category_pcts.get(cat, 0) for cat in categories]

            fig1 = px.bar(
                x=categories,
                y=populations,
                color=categories,
                color_discrete_map={
                    '🟢 Excellent (<300m)': '#2ecc71',
                    '🟡 Good (300-500m)': '#f1c40f',
                    '🟠 Fair (500-1000m)': '#e67e22',
                    '🔴 Poor (>1000m)': '#e74c3c'
                },
                text=[f"{int(pop):,}<br><b>{pct:.1f}%</b>" for pop, pct in zip(populations, percentages)]
            )
            fig1.update_layout(
                title="Distribution of Population by Bus Stop Accessibility",
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

        # Distance distribution histogram - POPULATION WEIGHTED
        st.markdown("---")
        st.markdown("### Distance Distribution to Nearest Bus Stop (Population-Weighted)")
        st.markdown("*Shows what % of the population experiences each distance range*")

        # Create bins manually with population weighting
        bins = np.linspace(df['dist_m'].min(), df['dist_m'].max(), 51)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate population in each bin
        bin_populations = []
        for i in range(len(bins) - 1):
            mask = (df['dist_m'] >= bins[i]) & (df['dist_m'] < bins[i + 1])
            pop_in_bin = df[mask]['Pop'].sum()
            bin_populations.append(pop_in_bin / total_population * 100)

        # Create bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=bin_centers,
            y=bin_populations,
            marker_color='#EBEBEB',
            width=(bins[1] - bins[0]) * 0.9,
            showlegend=False
        ))

        # Add KDE curve - population weighted
        distances_weighted = []
        for _, row in df.iterrows():
            distances_weighted.extend([row['dist_m']] * int(row['Pop'] / 100))

        if len(distances_weighted) > 0:
            distances_weighted = np.array(distances_weighted)
            kde = stats.gaussian_kde(distances_weighted)
            x_range = np.linspace(df['dist_m'].min(), df['dist_m'].max(), 200)
            y_kde = kde(x_range)
            y_kde_scaled = y_kde * max(bin_populations) / y_kde.max() * 0.8

            fig2.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                name='Density Curve',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

        # Calculate population percentages at each threshold
        pop_under_300 = df[df['dist_m'] <= 300]['Pop'].sum()
        pop_under_500 = df[df['dist_m'] <= 500]['Pop'].sum()
        pop_under_1000 = df[df['dist_m'] <= 1000]['Pop'].sum()

        pct_under_300 = (pop_under_300 / total_population * 100) if total_population > 0 else 0
        pct_under_500 = (pop_under_500 / total_population * 100) if total_population > 0 else 0
        pct_under_1000 = (pop_under_1000 / total_population * 100) if total_population > 0 else 0

        # Add reference lines with percentages
        fig2.add_vline(x=300, line_dash="dash", line_color="green",
                       annotation_text=f"300m<br>({pct_under_300:.1f}%)",
                       annotation_position="top")
        fig2.add_vline(x=500, line_dash="dash", line_color="orange",
                       annotation_text=f"500m<br>({pct_under_500:.1f}%)",
                       annotation_position="top")
        fig2.add_vline(x=1000, line_dash="dash", line_color="red",
                       annotation_text=f"1000m<br>({pct_under_1000:.1f}%)",
                       annotation_position="top")

        fig2.update_layout(
            xaxis_title="Distance to Nearest Bus Stop (meters)",
            yaxis_title="Percentage of Population (%)",
            xaxis_range=[0, df['dist_m'].max()],
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Distance distribution by ZONES (not population-weighted)
        st.markdown("---")
        st.markdown("### Distance Distribution by Zones")
        st.markdown("*Shows what % of zones have each distance range (not weighted by population)*")

        fig2b = px.histogram(
            df,
            x='dist_m',
            nbins=50,
            histnorm='percent',
            color_discrete_sequence=['#EBEBEB']
        )

        # Add KDE curve
        distances = df['dist_m'].dropna()
        if len(distances) > 1:
            kde = stats.gaussian_kde(distances)
            x_range = np.linspace(distances.min(), distances.max(), 200)
            y_kde = kde(x_range)
            y_kde_normalized = y_kde * 100 * (distances.max() - distances.min()) / 50

            fig2b.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_normalized,
                mode='lines',
                name='Density Curve',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

        # Calculate zone percentages at each threshold
        total_zones = len(df)
        zones_under_300 = (df['dist_m'] <= 300).sum()
        zones_under_500 = (df['dist_m'] <= 500).sum()
        zones_under_1000 = (df['dist_m'] <= 1000).sum()

        pct_zones_300 = (zones_under_300 / total_zones * 100) if total_zones > 0 else 0
        pct_zones_500 = (zones_under_500 / total_zones * 100) if total_zones > 0 else 0
        pct_zones_1000 = (zones_under_1000 / total_zones * 100) if total_zones > 0 else 0

        fig2b.add_vline(x=300, line_dash="dash", line_color="green",
                        annotation_text=f"300m<br>({pct_zones_300:.1f}% zones)",
                        annotation_position="top")
        fig2b.add_vline(x=500, line_dash="dash", line_color="orange",
                        annotation_text=f"500m<br>({pct_zones_500:.1f}% zones)",
                        annotation_position="top")
        fig2b.add_vline(x=1000, line_dash="dash", line_color="red",
                        annotation_text=f"1000m<br>({pct_zones_1000:.1f}% zones)",
                        annotation_position="top")

        fig2b.update_layout(
            xaxis_title="Distance to Nearest Bus Stop (meters)",
            yaxis_title="Percentage of Zones (%)",
            xaxis_range=[0, df['dist_m'].max()],
            height=400
        )
        st.plotly_chart(fig2b, use_container_width=True)

        # Walking time distribution - POPULATION WEIGHTED
        st.markdown("---")
        st.markdown("### Walking Time Distribution (Population-Weighted)")
        st.markdown("*Shows what % of the population experiences each walking time range*")

        # Create bins manually with population weighting
        bins_time = np.linspace(df['walk_time_min'].min(), df['walk_time_min'].max(), 41)
        bin_centers_time = (bins_time[:-1] + bins_time[1:]) / 2

        # Calculate population in each bin
        bin_populations_time = []
        for i in range(len(bins_time) - 1):
            mask = (df['walk_time_min'] >= bins_time[i]) & (df['walk_time_min'] < bins_time[i + 1])
            pop_in_bin = df[mask]['Pop'].sum()
            bin_populations_time.append(pop_in_bin / total_population * 100)

        # Create bar chart
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=bin_centers_time,
            y=bin_populations_time,
            marker_color='#EBEBEB',
            width=(bins_time[1] - bins_time[0]) * 0.9,
            showlegend=False
        ))

        # Add KDE curve - population weighted
        walk_times_weighted = []
        for _, row in df.iterrows():
            walk_times_weighted.extend([row['walk_time_min']] * int(row['Pop'] / 100))

        if len(walk_times_weighted) > 0:
            walk_times_weighted = np.array(walk_times_weighted)
            kde = stats.gaussian_kde(walk_times_weighted)
            x_range = np.linspace(df['walk_time_min'].min(), df['walk_time_min'].max(), 200)
            y_kde = kde(x_range)
            y_kde_scaled = y_kde * max(bin_populations_time) / y_kde.max() * 0.8

            fig3.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                name='Density Curve',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

        # Calculate population percentages at each threshold
        pop_under_5min = df[df['walk_time_min'] <= 5]['Pop'].sum()
        pop_under_10min = df[df['walk_time_min'] <= 10]['Pop'].sum()
        pop_under_15min = df[df['walk_time_min'] <= 15]['Pop'].sum()

        pct_under_5min = (pop_under_5min / total_population * 100) if total_population > 0 else 0
        pct_under_10min = (pop_under_10min / total_population * 100) if total_population > 0 else 0
        pct_under_15min = (pop_under_15min / total_population * 100) if total_population > 0 else 0

        fig3.add_vline(x=5, line_dash="dash", line_color="green",
                       annotation_text=f"5 min<br>({pct_under_5min:.1f}%)",
                       annotation_position="top")
        fig3.add_vline(x=10, line_dash="dash", line_color="orange",
                       annotation_text=f"10 min<br>({pct_under_10min:.1f}%)",
                       annotation_position="top")
        fig3.add_vline(x=15, line_dash="dash", line_color="red",
                       annotation_text=f"15 min<br>({pct_under_15min:.1f}%)",
                       annotation_position="top")

        fig3.update_layout(
            xaxis_title="Walking Time to Nearest Bus Stop (minutes)",
            yaxis_title="Percentage of Population (%)",
            xaxis_range=[0, df['walk_time_min'].max()],
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Walking time distribution by ZONES (not population-weighted)
        st.markdown("---")
        st.markdown("### Walking Time Distribution by Zones")
        st.markdown("*Shows what % of zones have each walking time range (not weighted by population)*")

        fig3b = px.histogram(
            df,
            x='walk_time_min',
            nbins=40,
            histnorm='percent',
            color_discrete_sequence=['#EBEBEB']
        )

        # Add KDE curve
        walk_times = df['walk_time_min'].dropna()
        if len(walk_times) > 1:
            kde = stats.gaussian_kde(walk_times)
            x_range = np.linspace(walk_times.min(), walk_times.max(), 200)
            y_kde = kde(x_range)
            y_kde_normalized = y_kde * 100 * (walk_times.max() - walk_times.min()) / 40

            fig3b.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_normalized,
                mode='lines',
                name='Density Curve',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

        # Calculate zone percentages at each threshold
        zones_under_5min = (df['walk_time_min'] <= 5).sum()
        zones_under_10min = (df['walk_time_min'] <= 10).sum()
        zones_under_15min = (df['walk_time_min'] <= 15).sum()

        pct_zones_5min = (zones_under_5min / total_zones * 100) if total_zones > 0 else 0
        pct_zones_10min = (zones_under_10min / total_zones * 100) if total_zones > 0 else 0
        pct_zones_15min = (zones_under_15min / total_zones * 100) if total_zones > 0 else 0

        fig3b.add_vline(x=5, line_dash="dash", line_color="green",
                        annotation_text=f"5 min<br>({pct_zones_5min:.1f}% zones)",
                        annotation_position="top")
        fig3b.add_vline(x=10, line_dash="dash", line_color="orange",
                        annotation_text=f"10 min<br>({pct_zones_10min:.1f}% zones)",
                        annotation_position="top")
        fig3b.add_vline(x=15, line_dash="dash", line_color="red",
                        annotation_text=f"15 min<br>({pct_zones_15min:.1f}% zones)",
                        annotation_position="top")

        fig3b.update_layout(
            xaxis_title="Walking Time to Nearest Bus Stop (minutes)",
            yaxis_title="Percentage of Zones (%)",
            xaxis_range=[0, df['walk_time_min'].max()],
            height=400
        )
        st.plotly_chart(fig3b, use_container_width=True)

    # TAB 2: Priority Areas
    with tab2:
        st.subheader("Priority Areas for Service Improvement")
        st.markdown("*Zones with poor bus stop access affecting the most residents*")

        # Identify poor access zones (>1000m)
        poor_zones = df[df['dist_m'] > 1000].copy()
        poor_zones = poor_zones.sort_values('Pop', ascending=False)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("#### 🔴 High-Population Zones with Worst Access")
            if len(poor_zones) > 0:
                total_affected = poor_zones['Pop'].sum()
                pct_affected = (total_affected / total_population * 100) if total_population > 0 else 0

                st.markdown(
                    f"**{len(poor_zones)} zones** with **{total_affected:,.0f} residents** "
                    f"({pct_affected:.1f}% of population) are >1000m from nearest stop"
                )

                top_poor = poor_zones.head(15)
                fig4 = px.bar(
                    top_poor,
                    x='Pop',
                    y='Núcleo',
                    orientation='h',
                    color='dist_m',
                    color_continuous_scale='Reds',
                    hover_data=['Municipio', 'dist_m', 'walk_time_min'],
                    text='Pop'
                )
                fig4.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig4.update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title='Population (residents)',
                    yaxis_title='Locality (Núcleo)',
                    coloraxis_colorbar_title='Distance (m)'
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.success("✅ No zones with poor access (>1000m) found!")

        with col2:
            st.markdown("#### Key Impact Metrics")

            if len(poor_zones) > 0:
                avg_dist_poor = poor_zones['dist_m'].mean()
                avg_walk_poor = poor_zones['walk_time_min'].mean()
                max_dist_poor = poor_zones['dist_m'].max()

                st.error(f"**Avg distance in poor zones:** {avg_dist_poor:.0f}m")
                st.warning(f"**Avg walk time:** {avg_walk_poor:.1f} minutes")
                st.info(f"**Max distance:** {max_dist_poor:.0f}m")

                # Top affected zones list
                st.markdown("#### Most Critical Zones")
                st.markdown("*(by population affected)*")
                for idx, row in poor_zones.head(5).iterrows():
                    st.write(f"**{row['Núcleo']}**")
                    st.write(f"→ {row['Pop']:,.0f} residents")
                    st.write(f"→ {row['dist_m']:.0f}m ({row['walk_time_min']:.1f} min walk)")
                    st.markdown("---")
            else:
                st.success("✅ All zones have good accessibility!")

        # Show data table
        if len(poor_zones) > 0:
            st.markdown("---")
            st.markdown("#### Complete List of Priority Zones")
            display_poor = poor_zones[['Núcleo', 'Municipio', 'Pop', 'dist_m', 'walk_time_min']].copy()
            st.dataframe(
                display_poor.style.format({
                    'Pop': '{:,.0f}',
                    'dist_m': '{:.0f}',
                    'walk_time_min': '{:.1f}'
                }).background_gradient(subset=['dist_m'], cmap='Reds'),
                hide_index=True,
                use_container_width=True,
                height=400
            )

    # TAB 3: Municipality Breakdown
    with tab3:
        st.subheader("Accessibility by Municipality")
        st.markdown("*Comparing bus stop access across different municipalities*")

        # Municipality statistics - POPULATION WEIGHTED
        muni_stats = df.groupby('Municipio').agg({
            'Pop': 'sum',
            'origin_id': 'count'
        }).reset_index()
        muni_stats.columns = ['Municipality', 'Population', 'Number of Zones']

        # Calculate population-weighted average distance per municipality
        weighted_dist_by_muni = df.groupby('Municipio').apply(
            lambda x: (x['dist_m'] * x['Pop']).sum() / x['Pop'].sum()
        ).reset_index()
        weighted_dist_by_muni.columns = ['Municipality', 'Avg Distance (m)']

        # Calculate population-weighted average walk time per municipality
        weighted_time_by_muni = df.groupby('Municipio').apply(
            lambda x: (x['walk_time_min'] * x['Pop']).sum() / x['Pop'].sum()
        ).reset_index()
        weighted_time_by_muni.columns = ['Municipality', 'Avg Walk Time (min)']

        # Merge statistics
        muni_stats = muni_stats.merge(weighted_dist_by_muni, on='Municipality')
        muni_stats = muni_stats.merge(weighted_time_by_muni, on='Municipality')

        # Calculate % with good access per municipality
        good_access_by_muni = df[df['dist_m'] <= 500].groupby('Municipio')['Pop'].sum()
        muni_stats['Good Access Pop'] = muni_stats['Municipality'].map(good_access_by_muni).fillna(0)
        muni_stats['% Good Access'] = (muni_stats['Good Access Pop'] / muni_stats['Population'] * 100)

        muni_stats = muni_stats.sort_values('Population', ascending=False)

        # Top municipalities chart
        top_munis = muni_stats.head(15)

        fig5 = px.bar(
            top_munis,
            x='Municipality',
            y='Population',
            color='% Good Access',
            color_continuous_scale='RdYlGn',
            hover_data=['Avg Distance (m)', 'Number of Zones'],
            text='Population'
        )
        fig5.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig5.update_layout(
            title='Top 15 Municipalities by Population',
            xaxis_title='Municipality',
            yaxis_title='Population (residents)',
            height=500,
            coloraxis_colorbar_title='% Good Access'
        )
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)

        # Municipality comparison table
        st.markdown("---")
        st.markdown("#### Municipality Statistics (Population-Weighted)")
        st.dataframe(
            muni_stats.style.format({
                'Population': '{:,.0f}',
                'Avg Distance (m)': '{:.0f}',
                'Avg Walk Time (min)': '{:.1f}',
                'Number of Zones': '{:,.0f}',
                '% Good Access': '{:.1f}%'
            }).background_gradient(subset=['% Good Access'], cmap='RdYlGn')
            .background_gradient(subset=['Avg Distance (m)'], cmap='RdYlGn_r'),
            hide_index=True,
            use_container_width=True,
            height=500
        )

else:
    st.info("👆 Please upload an Excel file to begin the analysis")

    st.markdown("""
    ### This tool analyzes:

    1. **🎯 Accessibility Overview**
       - Population distribution by walking distance to nearest bus stop
       - **Population-weighted** distance and walking time distributions
       - Summary statistics on access quality

    2. **⚠️ Priority Areas**
       - Zones with worst access (>1000m) ranked by population impact
       - Total residents affected by poor access
       - Detailed list of priority zones for service improvement

    3. **🏛 Municipality Breakdown**
       - Accessibility comparison across municipalities
       - **Population-weighted** metrics by administrative area
       - Identification of municipalities needing more stops

    ### Expected Data Format:
    Your Excel file should contain these columns:
    - `origin_id`: Unique zone identifier
    - `nearest_stop`: ID of closest bus stop
    - `Núcleo`: Locality/settlement name
    - `Municipio`: Municipality name
    - `dist_m`: Distance to nearest stop (meters)
    - `Pop`: Population in the zone

    ### Key Features:
    - **Population-weighted analysis**: All metrics prioritize areas with more residents
    - **Walking time calculations**: Automatic conversion from distance to time
    - **Visual distributions**: Population-weighted histograms with density curves
    - **Priority identification**: Highlights zones needing new stops or improved service
    - **Threshold annotations**: All vertical lines show % of population within each distance/time
    """)