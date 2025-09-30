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

    # DEBUG: Print actual column names
    st.write("**Column names in your file:**")
    st.write(df.columns.tolist())

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
        avg_distance = df['dist_m'].mean()
        st.metric("Avg Distance to Stop", f"{avg_distance:.0f}m")

    with col2:
        avg_walk_time = df['walk_time_min'].mean()
        st.metric("Avg Walk Time", f"{avg_walk_time:.1f} min")

    with col3:
        pop_within_500m = df[df['dist_m'] <= 500]['Pop'].sum()
        pct_within_500m = (pop_within_500m / total_population * 100) if total_population > 0 else 0
        st.metric("Pop within 500m", f"{pct_within_500m:.1f}%")

    with col4:
        median_distance = df['dist_m'].median()
        st.metric("Median Distance", f"{median_distance:.0f}m")

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Accessibility Overview",
        "⚠️ Priority Areas",
        "📍 Municipality Breakdown"
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

        # Distance distribution histogram
        st.markdown("---")
        st.markdown("### Distance Distribution to Nearest Bus Stop")

        fig2 = px.histogram(
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

            fig2.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_normalized,
                mode='lines',
                name='Density Curve',
                line=dict(color='#C00000', width=3),
                showlegend=False
            ))

        # Add reference lines
        fig2.add_vline(x=300, line_dash="dash", line_color="green",
                       annotation_text="300m")
        fig2.add_vline(x=500, line_dash="dash", line_color="orange",
                       annotation_text="500m")
        fig2.add_vline(x=1000, line_dash="dash", line_color="red",
                       annotation_text="1000m")

        fig2.update_layout(
            xaxis_title="Distance to Nearest Bus Stop (meters)",
            yaxis_title="Percentage of Zones (%)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Walking time distribution
        st.markdown("---")
        st.markdown("### Walking Time Distribution")

        fig3 = px.histogram(
            df,
            x='walk_time_min',
            nbins=40,
            histnorm='percent',
            color_discrete_sequence=['#EBEBEB']  # Same gray color as fig2
        )

        # Add KDE curve (same as fig2)
        walk_times = df['walk_time_min'].dropna()
        if len(walk_times) > 1:
            kde = stats.gaussian_kde(walk_times)
            x_range = np.linspace(walk_times.min(), walk_times.max(), 200)
            y_kde = kde(x_range)
            y_kde_normalized = y_kde * 100 * (walk_times.max() - walk_times.min()) / 40  # 40 = nbins

            fig3.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_normalized,
                mode='lines',
                name='Density Curve',
                line=dict(color='#C00000', width=3),  # Same red color as fig2
                showlegend=False
            ))

        fig3.add_vline(x=5, line_dash="dash", line_color="green",
                       annotation_text="5 min")
        fig3.add_vline(x=10, line_dash="dash", line_color="orange",
                       annotation_text="10 min")
        fig3.add_vline(x=15, line_dash="dash", line_color="red",
                       annotation_text="15 min")

        fig3.update_layout(
            xaxis_title="Walking Time to Nearest Bus Stop (minutes)",
            yaxis_title="Percentage of Zones (%)",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

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

        # Municipality statistics
        muni_stats = df.groupby('Municipio').agg({
            'Pop': 'sum',
            'dist_m': 'mean',
            'walk_time_min': 'mean',
            'origin_id': 'count'
        }).reset_index()
        muni_stats.columns = ['Municipality', 'Population', 'Avg Distance (m)',
                              'Avg Walk Time (min)', 'Number of Zones']

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
        st.markdown("#### Municipality Statistics")
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
    ### This simplified tool analyzes:

    1. **🎯 Accessibility Overview**
       - Population distribution by walking distance to nearest bus stop
       - Distance and walking time distributions
       - Summary statistics on access quality

    2. **⚠️ Priority Areas**
       - Zones with worst access (>1000m) ranked by population impact
       - Total residents affected by poor access
       - Detailed list of priority zones for service improvement

    3. **📍 Municipality Breakdown**
       - Accessibility comparison across municipalities
       - Population-weighted metrics by administrative area
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
    - **Visual distributions**: Histograms and density curves for distance patterns
    - **Priority identification**: Highlights zones needing new stops or improved service
    """)