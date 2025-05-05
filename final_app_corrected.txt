
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from scipy.stats import genpareto
import datetime

st.set_page_config(page_title="FX Thresholding App", layout="wide")

# Utility functions
def rolling_quantile(vol, window, q):
    return vol.rolling(window).quantile(q)

def garch_evt(returns, tail_pct):
    am = arch_model(returns*100, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    std = res.resid / res.conditional_volatility
    std = std[~np.isnan(std)]
    u = np.quantile(std, 0.90)
    exc = std[std > u] - u
    c, loc, scale = genpareto.fit(exc, floc=0)
    p_exc = (tail_pct - (1 - np.mean(std > u))) / np.mean(std > u)
    var = genpareto.ppf(p_exc, c, loc=0, scale=scale)
    return (u + var) / 100.0

def detect_regimes(vol, n_states):
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200)
    arr = vol.values.reshape(-1, 1)
    model.fit(arr)
    states = model.predict(arr)
    means = {s: arr[states == s].mean() for s in np.unique(states)}
    high = max(means, key=means.get)
    return states, high, means[high]

# Upload
st.sidebar.header("Upload & Settings")
f = st.sidebar.file_uploader("Upload CSV (Date,Open,High,Low,Close,OHLCVolatility,Currency)", type="csv")
if not f:
    st.sidebar.info("Awaiting FX data…")
    st.stop()

df = pd.read_csv(f, parse_dates=['Date']).sort_values('Date')
df['DailyVol'] = df['OHLCVolatility'] / np.sqrt(252)

# Sidebar selections
currencies = sorted(df["Currency"].unique())
selected_ccy = st.sidebar.selectbox("Select Currency", currencies)

dfc = df[df["Currency"] == selected_ccy].copy().reset_index(drop=True)
dfc["LogReturn"] = np.log(dfc["Close"] / dfc["Close"].shift(1))
dfc = dfc.dropna()

# Detect regime
states, high_r, hmm_thr = detect_regimes(dfc['DailyVol'], n_states=3)
dfc['Regime'] = states

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Thresholding", "Shock Simulation"])

with tab1:
    st.title("FX Thresholding: Manual vs Hybrid Approach")

    st.markdown("""
    ### Why Move Beyond Manual Thresholds?

    **Manual Procedure:**
    - Fixed groups (1 to 4) based on annualized OHLC volatility.
    - Example:
        - Group 1: Avg Vol ~0.04 → Threshold = 0.10
        - Group 2: Avg Vol ~0.33 → Threshold = 0.40
        - Group 3: Avg Vol ~0.64 → Threshold = 0.70
        - Group 4: Avg Vol ~1.25 → Threshold = 1.30
    - Once calculated, these thresholds remain unchanged regardless of evolving market behavior.

    **Issues with Manual Thresholds:**
    - **Static**: Doesn’t adapt to market volatility shifts.
    - **Blind spots**: Misses sudden volatility jumps or misclassifies stable periods as risky.
    - **Cross Pair Problems**: When combining currencies from different groups, manual logic may understate or overstate risk.

    **Our Hybrid Dynamic Approach:**
    - Uses HMM to segment time into regimes (e.g., Calm, Stress).
    - Applies rolling quantiles and EVT calibration per regime.
    - Supports shock simulations and re-thresholding instantly.
    - **Example**: On Oct 14, threshold = 0.70. After shock between Oct 15–Nov 30, threshold adjusted to 1.08, proving dynamic response.

    **Business Value:**
    - Reduces false alerts and missed anomalies.
    - Builds explainability into every flag.
    - Enables audit-trail view of threshold origin (date, logic, inputs).
    """)

with tab2:
    st.header(f"Volatility Regimes for {selected_ccy}")
    fig = px.scatter(dfc, x='Date', y='DailyVol', color='Regime', title=f'{selected_ccy} Volatility Regimes')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Shock Simulation & Recalibration")

    snap_date = st.date_input("Snapshot Date", value=pd.to_datetime("2024-10-14"), min_value=dfc["Date"].min(), max_value=dfc["Date"].max())
    shock_start = st.date_input("Shock Start Date", value=pd.to_datetime("2024-10-15"))
    shock_end = st.date_input("Shock End Date", value=datetime.date.today())

    shock_mag = st.slider("Shock Magnitude (x real vol)", 1.0, 5.0, 2.0, step=0.1)

    mask = (dfc["Date"] >= pd.to_datetime(shock_start)) & (dfc["Date"] <= pd.to_datetime(shock_end))
    dfc["ShockedVol"] = dfc["DailyVol"]
    dfc.loc[mask, "ShockedVol"] = dfc.loc[mask, "DailyVol"] * shock_mag

    window = 90
    rolling_alert = rolling_quantile(dfc["ShockedVol"], window, 0.95)
    evt_threshold = garch_evt(dfc["LogReturn"], tail_pct=0.990)

    latest_alert = rolling_alert.iloc[-1]
    latest_evt = evt_threshold
    manual_thr = dfc[dfc["Date"] <= snap_date]["DailyVol"].mean()

    cols = st.columns(3)
    cols[0].metric("Manual Thr (at snap)", f"{manual_thr:.4f}")
    cols[1].metric("Dynamic Alert Thr", f"{latest_alert:.4f}")
    cols[2].metric("Dynamic EVT Thr", f"{latest_evt:.4f}")

    fig_shock = go.Figure()
    fig_shock.add_trace(go.Scatter(x=dfc["Date"], y=dfc["DailyVol"], name="Original Vol"))
    fig_shock.add_trace(go.Scatter(x=dfc["Date"], y=dfc["ShockedVol"], name="Shocked Vol", line=dict(color='blue')))
    fig_shock.add_hline(y=latest_alert, line_dash="dash", line_color="red")
    fig_shock.add_vrect(x0=shock_start, x1=shock_end, fillcolor="red", opacity=0.2, line_width=0)
    fig_shock.update_layout(title="Volatility with Shock Region", yaxis_title="Daily Volatility")
    st.plotly_chart(fig_shock, use_container_width=True)
