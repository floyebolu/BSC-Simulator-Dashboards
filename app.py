import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# 1. Setup the Page (This is the only "config" you need)
# -----------------------------------------------------------------
# Set the layout to be wide
st.set_page_config(layout="wide")
sns.set(font_scale=1.5)

# -----------------------------------------------------------------
# 2. Data Loading (Load the raw data once and cache it)
# -----------------------------------------------------------------
@st.cache_data # Caches the raw data for performance
def load_data():
    # Attempt to load real data
    try:
        with np.load('data/transfers_data.npz', allow_pickle=True,) as data:
            df = pd.DataFrame(data['moves_dist_df'], columns=data['columns'])
    except FileNotFoundError:
        # Generate synthetic data if file not found
        np.random.seed(42)
        n_rows = 5000
        scenarios = [f'Scenario_{i}' for i in range(1, 9)]
        shus = [f'SHU_{i}' for i in range(1, 15)]
        # Generate data with some zeros to demonstrate the filter
        data = {
            'daily_moves': np.where(np.random.rand(n_rows) < 0.1, 0, np.random.normal(50, 20, n_rows)).clip(min=0),
            'scenario': np.random.choice(scenarios, n_rows),
            'shu': np.random.choice(shus, n_rows)
        }
        df = pd.DataFrame(data)
    
    # Calculate the global min/max *before* any filtering for stable slider/heatmap bounds
    min_move = df['daily_moves'].min()
    max_move = df['daily_moves'].max()
    
    # NOTE: The combination logic is removed from here to allow dynamic filtering below.
    return df, min_move, max_move 

# Load the raw (uncombined) data and global bounds.
# df_raw holds the base data, min_move and max_move are used for stable plot ranges.
df_raw, min_move, max_move = load_data()


# -----------------------------------------------------------------
# 3. Widget UI (Replaces ipywidgets)
# -----------------------------------------------------------------
st.title('Number of RBC Units in Transfers with Different Transportation Scenarios')

with st.expander("About this Dashboard"):
    st.markdown("""
    This dashboard is designed to help explore the preferences of logistics staff by visualizing the results of a six-week long blood supply chain simulation. The goal is to understand the operational impact of different strategies for transferring red blood cell (RBC) units between stock holding units (SHUs).

    **Context:** The main cost is not the transportation itself, as transfers utilize the existing NHSBT transport schedule. Instead, the key effort is finding and retrieving units at each SHU for transfer. This dashboard explores various scenarios, each representing a different level of restriction on transferring units. These scenarios are controlled by a parameter, $\lambda_3$, which ranges from 0 to 1.

    *   A $\lambda_3$ value of 0 represents no constraints on transfers.
    *   A $\lambda_3$ value of 1 represents the most 'restrictive' scenario, where the penalty for transfers is as important as the penalty for mismatching an RBC unit to a patient (on non-mandatory antigens).

    The heatmaps below visualize the simulation outcomes for each scenario:
    *   **Left Plot (Distribution of Transfers):** For the days that there are transfers from a SHU, setting $\\alpha=95\%$ means $95\%$ of those days had equal or fewer transfers than the value shown in the plot. Default is $\\alpha=95\%$.
    *   **Right Plot (Probability of High-Volume Transfers):** This shows the probability that on a day requiring a transfer, more than $\\beta$ number of RBCs will need to be moved. Default is $\\beta=50$ RBCs.

    Use the sliders below to explore the data. There is also a checkbox to exclude days with no outgoing RBC transfers from the analysis, which may be useful for focusing on active transfer days.
    """)

# Create two columns for the filter and main sliders
col0, col1, col2 = st.columns([1, 2, 2])

# START of your requested feature: Checkbox
with col0:
    exclude_zero_moves = st.checkbox(
        'Exclude Days without Transfers', 
        value=False,
        help="If checked, only days where there are RBC transfers are included in the analysis."
    )
# END of your requested feature

with col1:
    percentile = st.slider(
        'Show transfer values for this percentile of days (%): $\\alpha$', 
        min_value=0.0, 
        max_value=100.0, 
        value=95.0, 
        step=0.1
    )

with col2:
    threshold = st.slider(
        'Threshold for Number of Units to Transfer Out: $\\beta$', 
        min_value=min_move, 
        max_value=max_move, 
        value=50, 
        step=1,
        format="%d"
    )

st.markdown("---") # Adds a horizontal line

# -----------------------------------------------------------------
# 4. Data Filtering and Combination (Happens on every slider/checkbox change)
# -----------------------------------------------------------------
# This section now dynamically creates the df_combined used by the plots.
df_filtered = df_raw.copy()

if exclude_zero_moves:
    # Apply filter: only keep rows where daily_moves is greater than 0
    df_filtered = df_filtered[df_filtered['daily_moves'] > 0].copy()
    
# Re-create the 'ALL' category and the final df_combined (now filtered if necessary)
df_all = df_filtered.copy()
df_all['shu'] = 'ALL (Total)'
df_combined = pd.concat([df_filtered, df_all], ignore_index=True)

# -----------------------------------------------------------------
# 5. Plotting Logic (The same as your helper module)
# -----------------------------------------------------------------
plot_col1, plot_col2 = st.columns(2)

# -- Plot 1: Inverse CDF --
with plot_col1:
    if exclude_zero_moves:
        st.subheader(f'$\\alpha={percentile:.1f}\\%$ of transfers in one day are below the value(s) shown')
    else:
        st.subheader(f'$\\alpha={percentile:.1f}\\%$ of days have transfers below the value(s) shown')
    
    # Calculate pivot
    q = percentile / 100.0
    pivot_inv = df_combined.pivot_table(
        index='scenario', columns='shu', values='daily_moves', 
        # For the quantile calculation, we ignore NaNs, which is fine.
        aggfunc=lambda x: np.quantile(x, q)
    )
    
    # Create the plot
    fig1, ax1 = plt.subplots(figsize=(20, 14), dpi=200)
    sns.heatmap(
        pivot_inv, annot=True, fmt=".0f", cmap="viridis", ax=ax1,
        cbar_kws={'label': 'Daily RBC Transfers'},
        vmin=min_move,
        vmax=max_move
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', )
    ax1.set_xlabel('SHU Location')
    ax1.set_ylabel('Transportation Penalty Scenario: $\lambda_3$')
    st.pyplot(fig1)

# -- Plot 2: Complementary CDF --
with plot_col2:
    if exclude_zero_moves:
        # Note: If excluding zeros, the probability is conditional on having a transfer day.
        # The title reflects this by stating "a day's transfers" (i.e., days in the current filtered set).
        st.subheader(f'Probability that a day\'s transfers exceeds $\\beta={threshold:d}$ RBC(s)')
    else:
        st.subheader(f'Probability that more than $\\beta={threshold:d}$ RBC(s) are transferred on a given day')
    
    # Calculate pivot
    pivot_ccdf = df_combined.pivot_table(
        index='scenario', columns='shu', values='daily_moves', 
        aggfunc=lambda x: (x > threshold).mean()
    )
    
    # Create the plot
    fig2, ax2 = plt.subplots(figsize=(20, 14), dpi=200)
    sns.heatmap(
        pivot_ccdf, annot=True, fmt=".0%", cmap="magma", ax=ax2,
        cbar_kws={'label': 'Probability'},
        vmin=0,  # Stable 0-1 range
        vmax=1
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_xlabel('SHU Location')
    ax2.set_ylabel('Transportation Penalty Scenario: $\lambda_3$')
    st.pyplot(fig2)
