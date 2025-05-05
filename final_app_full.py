
# FX Volatility Thresholding App (Updated Version)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import genpareto
from hmmlearn.hmm import GaussianHMM

# --- Utility Functions ---
def rolling_quantile(series, window, quantile):
    return series.rolling(window=window, min_periods=1).quantile(quantile)

def garch_evt(returns, tail_pct=0.990):
    from arch import arch_model
    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid[~np.isnan(std_resid)]
    threshold = np.quantile(std_resid, 0.90)
    exceedances = std_resid[std_resid > threshold] - threshold
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    var = genpareto.ppf(tail_pct, shape, loc=0, scale=scale)
    return (threshold + var) / 100

# --- App Layout ---
st.set_page_config(page_title="FX Threshold Comparison", layout="wide")

tab1, tab2, tab3 = st.tabs(["Overview", "Volatility Regimes", "Shock Simulation"])

# --- File Upload & Selection ---
uploaded = st.sidebar.file_uploader("Upload FX Volatility Data", type=["csv"])
selected_ccy = st.sidebar.selectbox("Currency", [])

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["Date"])
    df = df.sort_values("Date")
    df["DailyVol"] = df["OHLCVolatility"] / np.sqrt(252)
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    # Compute dynamic threshold (95th quantile over 90-day rolling)
    window = 90
    quantile_level = 0.95
    df["DynamicThreshold"] = rolling_quantile(df["DailyVol"], window, quantile_level)
    df["Regime"] = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200).fit(df["DailyVol"].values.reshape(-1, 1)).predict(df["DailyVol"].values.reshape(-1, 1))
    df["RegimeName"] = df["Regime"].map({0: "Low", 1: "Medium", 2: "High"})
    selected_ccy = df["Currency"].iloc[0]

    # Manual threshold baseline
    manual_threshold = df["DailyVol"].mean()

    # Fill in tab code (excluded here for space, matches previous markdown)
    with open("/mnt/data/updated_threshold_tabs.txt", "r") as file:
        exec(file.read())

else:
    tab1.write("Please upload a file with columns: Date, Currency, Close, OHLCVolatility")
