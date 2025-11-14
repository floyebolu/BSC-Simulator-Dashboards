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
st.title('Scenario & SHU Risk Analysis')

# Create two columns for the sliders
col1, col2 = st.columns(2)

with col1:
    percentile = st.slider(
        'Inverse CDF % (Percentile):', 
        min_value=0.0, 
        max_value=100.0, 
        value=95.0, 
        step=0.5
    )

with col2:
    threshold = st.slider(
        'CCDF Threshold (Daily Moves):', 
        min_value=min_move, 
        max_value=max_move, 
        value=0.0, 
        step=(max_move - min_move) / 100,
        format="%.4f"
    )

st.markdown("---") # Adds a horizontal line

# -----------------------------------------------------------------
# 4. Plotting Logic (The same as your helper module)
# -----------------------------------------------------------------
# Create two columns for the plots
plot_col1, plot_col2 = st.columns(2)

# -- Plot 1: Inverse CDF --
with plot_col1:
    st.subheader(f'Inverse CDF: Value at {percentile}% Percentile')
    
    # Calculate pivot
    q = percentile / 100.0
    pivot_inv = df_combined.pivot_table(
        index='scenario', columns='shu', values='daily_moves', 
        aggfunc=lambda x: np.quantile(x, q)
    )
    
    # Create the plot
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        pivot_inv, annot=True, fmt=".2f", cmap="viridis", ax=ax1,
        cbar_kws={'label': 'Daily Moves Value'},
        vmin=min_move,  # Use stable min/max
        vmax=max_move
    )
    st.pyplot(fig1) # This is how you show a matplotlib plot

# -- Plot 2: Complementary CDF --
with plot_col2:
    st.subheader(f'CCDF: Probability Moves > {threshold:.4f}')
    
    # Calculate pivot
    pivot_ccdf = df_combined.pivot_table(
        index='scenario', columns='shu', values='daily_moves', 
        aggfunc=lambda x: (x > threshold).mean()
    )
    
    # Create the plot
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        pivot_ccdf, annot=True, fmt=".1%", cmap="magma", ax=ax2,
        cbar_kws={'label': 'Probability'},
        vmin=0,  # Stable 0-1 range
        vmax=1
    )
    st.pyplot(fig2)
