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

# -----------------------------------------------------------------
# 2. Data Loading (Same as your notebook)
# -----------------------------------------------------------------
# In a real app, you would load your 'df' here
# For this example, we'll use the same synthetic data
@st.cache_data  # Caches the data for performance
def load_data():
    try:
        with np.load('data/transfers_data.npz', allow_pickle=True,) as data:
            df = pd.DataFrame(data['moves_dist_df'], columns=data['columns'])
    except FileNotFoundError:
        # Generate synthetic data if file not found
        np.random.seed(42)
        n_rows = 5000
        scenarios = [f'Scenario_{i}' for i in range(1, 9)]
        shus = [f'SHU_{i}' for i in range(1, 15)]
        data = {
            'daily_moves': np.random.normal(0, 1, n_rows),
            'scenario': np.random.choice(scenarios, n_rows),
            'shu': np.random.choice(shus, n_rows)
        }
        df = pd.DataFrame(data)
    
    # --- The SAME pre-processing logic ---
    df_all = df.copy()
    df_all['shu'] = 'ALL (Total)'
    df_combined = pd.concat([df, df_all], ignore_index=True)
    return df_combined, df['daily_moves'].min(), df['daily_moves'].max()

df_combined, min_move, max_move = load_data()


# -----------------------------------------------------------------
# 3. Widget UI (Replaces ipywidgets)
# -----------------------------------------------------------------
st.title('Number of RBC Units in Transfers with Different Transportation Scenarios')

with st.expander("About this Dashboard"):
    st.markdown("""
    This dashboard is designed to help gauge the preferences of logistics staff by visualizing the results of a blood supply chain simulation. The goal is to understand the operational impact of different strategies for transferring Red Blood Cell (RBC) units between Stock Holding Units (SHUs).

    **Context:** The main cost is not the transportation itself, as transfers utilize the existing NHSBT transport schedule. Instead, the key effort is finding and retrieving units for transfer. This dashboard explores various scenarios, each representing a different level of restriction on transferring units. These scenarios are controlled by a parameter, $\lambda_3$, which ranges from 0 to 1.

    *   A $\lambda_3$ value of 0 represents no constraints on transfers.
    *   A $\lambda_3$ value of 1 represents the most 'restrictive' scenario, where the penalty for transfers is as important as the penalty for mismatching an RBC unit to a patient.

    The heatmaps below visualize the simulation outcomes for each scenario:
    *   **Left Plot (Distribution of Transfers):** For the days that there are transfers from a SHU, this shows the maximum number of transfers for the first $\\alpha\%$ least busy days. Default is $\\alpha=95\%$, i.e., the value at which $95\%$ of the data falls below this number.
    *   **Right Plot (Probability of High-Volume Transfers):** This shows the probability that on a day requiring a transfer, more than $\\beta$ number of RBCs will need to be moved. Default is $\\beta=50$ RBCs.

    Use the sliders below to explore the data.
    """)

# Create two columns for the sliders
col1, col2 = st.columns(2)

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
# 4. Plotting Logic (The same as your helper module)
# -----------------------------------------------------------------
# Create two columns for the plots
plot_col1, plot_col2 = st.columns(2)

# -- Plot 1: Inverse CDF --
with plot_col1:
    st.subheader(f'$\\alpha={percentile:.1f}\%$ of transfers in one day are below the value(s) shown')
    
    # Calculate pivot
    q = percentile / 100.0
    pivot_inv = df_combined.pivot_table(
        index='scenario', columns='shu', values='daily_moves', 
        aggfunc=lambda x: np.quantile(x, q)
    )
    
    # Create the plot
    fig1, ax1 = plt.subplots(figsize=(14, 14))
    sns.heatmap(
        pivot_inv, annot=True, fmt=".0f", cmap="viridis", ax=ax1,
        cbar_kws={'label': 'Daily RBC Transfers'},
        vmin=min_move,  # Use stable min/max
        vmax=max_move
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('SHU Location')
    ax1.set_ylabel('Transportation Penalty Scenario: $\lambda_3$')
    st.pyplot(fig1) # This is how you show a matplotlib plot

# -- Plot 2: Complementary CDF --
with plot_col2:
    st.subheader(f'Probability that a day\'s transfers exceeds $\\beta={threshold:d}$ RBC(s)')
    
    # Calculate pivot
    pivot_ccdf = df_combined.pivot_table(
        index='scenario', columns='shu', values='daily_moves', 
        aggfunc=lambda x: (x > threshold).mean()
    )
    
    # Create the plot
    fig2, ax2 = plt.subplots(figsize=(14, 14))
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
