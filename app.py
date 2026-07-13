import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from scipy import stats
import os

# Conditional import for yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Factor Model Analyzer",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING (same as before – included for completeness)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

    :root {
        --bg:        #08090d;
        --surface:   #0f1117;
        --surface2:  #161820;
        --border:    #1e2030;
        --border2:   #2a2d42;
        --accent:    #4fffb0;
        --accent2:   #00c9ff;
        --warn:      #ff6b6b;
        --success:   #4fffb0;
        --muted:     #4a4f6a;
        --text:      #e8eaf0;
        --text2:     #8b90ab;
        --font-head: 'Space Grotesk', 'Trebuchet MS', sans-serif;
        --font-mono: 'IBM Plex Mono', 'Courier New', monospace;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main { background: var(--bg) !important; }

    * { font-family: var(--font-mono) !important; }

    [data-testid="stHeader"],
    #stDecoration,
    [data-testid="stToolbar"] { display: none !important; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1600px !important;
    }

    [data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    [data-testid="stSidebar"] * { color: var(--text) !important; }
    
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stMultiSelect select,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stDateInput input {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text) !important;
        border-radius: 4px !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text2) !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stSidebar"] h3 {
        font-family: var(--font-head) !important;
        color: var(--accent) !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        margin-top: 1.2rem !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 0.8rem 0 !important;
    }

    .header-block {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 2rem;
        padding-bottom: 1.4rem;
        border-bottom: 1px solid var(--border);
    }

    .logo {
        font-family: 'Space Grotesk', 'Trebuchet MS', Arial, sans-serif !important;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.02em;
    }

    .logo span { color: var(--accent); }

    .subtitle {
        font-family: 'Space Grotesk', 'Trebuchet MS', Arial, sans-serif !important;
        font-size: 0.85rem;
        color: var(--text2);
        letter-spacing: 0.05em;
    }

    .badge {
        margin-left: auto;
        background: rgba(79,255,176,0.07);
        border: 1px solid rgba(79,255,176,0.25);
        color: var(--accent);
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.25rem 0.6rem;
        border-radius: 3px;
        white-space: nowrap;
    }

    .sec-label {
        font-family: var(--font-head);
        font-size: 0.6rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.8rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .sec-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    .sec-label .dot {
        width: 4px; height: 4px;
        border-radius: 50%;
        background: var(--accent);
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .stat-cell {
        background: var(--surface);
        padding: 1rem 1.2rem;
        position: relative;
    }

    .stat-cell:hover { background: var(--surface2); }

    .stat-label {
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text2);
        margin-bottom: 0.35rem;
    }

    .stat-value {
        font-family: var(--font-head);
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent);
        line-height: 1;
    }

    .stat-sub {
        font-size: 0.55rem;
        color: var(--muted);
        margin-top: 0.25rem;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        font-size: 0.6rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text2);
        margin-bottom: 0.8rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    .rank-badge {
        display: inline-block;
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
    }

    .rank-1 { background: rgba(79,255,176,0.15); color: var(--accent); }
    .rank-2 { background: rgba(0,201,255,0.15); color: var(--accent2); }
    .rank-3 { background: rgba(255,107,107,0.15); color: var(--warn); }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }

    table thead {
        background: var(--surface2);
        border-bottom: 2px solid var(--border2);
    }

    table th {
        padding: 0.7rem;
        text-align: left;
        color: var(--text2);
        font-weight: 600;
        letter-spacing: 0.08em;
        font-size: 0.65rem;
        text-transform: uppercase;
    }

    table td {
        padding: 0.6rem 0.7rem;
        border-bottom: 1px solid var(--border);
        color: var(--text);
    }

    table tbody tr:hover { background: rgba(79,255,176,0.03); }

    .metric-highlight {
        color: var(--accent);
        font-weight: 600;
    }

    .metric-warn {
        color: var(--warn);
        font-weight: 600;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA with fallback to file upload
# ═══════════════════════════════════════════════════════════════════════════════

def get_file_path(filename):
    """Return the absolute path to a file in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

@st.cache_data
def load_factor_data(uploaded_ff3=None, uploaded_ff5=None):
    """Load factor data from files or uploaded content."""
    try:
        if uploaded_ff3 is not None and uploaded_ff5 is not None:
            # Use uploaded files
            ff3 = pd.read_csv(uploaded_ff3, skiprows=6, index_col=0)
            ff5 = pd.read_csv(uploaded_ff5, skiprows=6, index_col=0)
        else:
            # Try to load from disk
            ff3_path = get_file_path('Europe_3_Factors.csv')
            ff5_path = get_file_path('Europe_5_Factors.csv')
            ff3 = pd.read_csv(ff3_path, skiprows=6, index_col=0)
            ff5 = pd.read_csv(ff5_path, skiprows=6, index_col=0)
        
        for df in [ff3, ff5]:
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m')
            df.replace(-99.99, np.nan, inplace=True)
            df = df / 100
        return ff3, ff5
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return None, None

# Try loading data initially
ff3, ff5 = load_factor_data()

# If missing, present file uploader in main area
if ff3 is None or ff5 is None:
    st.warning("⚠️ Factor data files not found in the app directory.")
    st.info("Please upload the required CSV files below to continue.")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_ff3 = st.file_uploader("Upload Europe_3_Factors.csv", type=['csv'])
    with col2:
        uploaded_ff5 = st.file_uploader("Upload Europe_5_Factors.csv", type=['csv'])
    
    if uploaded_ff3 is not None and uploaded_ff5 is not None:
        with st.spinner("Loading data..."):
            ff3, ff5 = load_factor_data(uploaded_ff3, uploaded_ff5)
            if ff3 is not None and ff5 is not None:
                st.success("✅ Data loaded successfully! You can now use the app.")
                st.rerun()
            else:
                st.error("Failed to load data. Please check file format.")
    else:
        st.stop()
else:
    # Portfolio data is optional
    try:
        port_path = get_file_path('Europe_25_Portfolios_ME_BE-ME.csv')
        port_data = pd.read_csv(port_path, skiprows=9)
    except:
        port_data = None

# ═══════════════════════════════════════════════════════════════════════════════
# MACRO PERIODS
# ═══════════════════════════════════════════════════════════════════════════════

MACRO_PERIODS = {
    "Full Period (1990-2024)": (datetime(1990, 1, 1), datetime(2024, 12, 31)),
    "Pre-GFC (1990-2007)": (datetime(1990, 1, 1), datetime(2007, 12, 31)),
    "GFC Crisis (2008-2009)": (datetime(2008, 1, 1), datetime(2009, 12, 31)),
    "Post-GFC Recovery (2010-2012)": (datetime(2010, 1, 1), datetime(2012, 12, 31)),
    "ECB Stabilization (2013-2019)": (datetime(2013, 1, 1), datetime(2019, 12, 31)),
    "COVID Pandemic (2020-2021)": (datetime(2020, 1, 1), datetime(2021, 12, 31)),
    "Post-COVID (2022-2024)": (datetime(2022, 1, 1), datetime(2024, 12, 31)),
}

# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION FUNCTIONS (same as before)
# ═══════════════════════════════════════════════════════════════════════════════

def run_capm(returns, risk_free, market_premium):
    if len(returns) < 12:
        return None
    y = returns.values - risk_free.values
    X = market_premium.values.reshape(-1, 1)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y - y.mean())**2)
    r_squared = 1 - (rss / tss)
    n = len(y)
    mse = rss / (n - 2)
    var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    std_errors = np.sqrt(np.diag(var_covar))
    t_stats = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - 2))
    return {
        'alpha': beta[0], 'beta': beta[1], 'r_squared': r_squared,
        'alpha_se': std_errors[0], 'beta_se': std_errors[1],
        'alpha_t': t_stats[0], 'beta_t': t_stats[1],
        'alpha_p': p_values[0], 'beta_p': p_values[1],
        'n_obs': n
    }

def run_ff3(returns, risk_free, market_premium, smb, hml):
    common = returns.index.intersection(market_premium.index).intersection(smb.index).intersection(hml.index)
    if len(common) < 12:
        return None
    y = (returns.loc[common] - risk_free.loc[common]).values
    X = np.column_stack([market_premium.loc[common].values, smb.loc[common].values, hml.loc[common].values])
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y - y.mean())**2)
    r_squared = 1 - (rss / tss)
    n = len(y)
    mse = rss / (n - 4)
    var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    std_errors = np.sqrt(np.diag(var_covar))
    t_stats = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - 4))
    return {
        'alpha': beta[0], 'mkt': beta[1], 'smb': beta[2], 'hml': beta[3],
        'r_squared': r_squared, 'alpha_se': std_errors[0], 'alpha_p': p_values[0],
        'n_obs': n
    }

def run_ff5(returns, risk_free, market_premium, smb, hml, rmw, cma):
    common = returns.index.intersection(market_premium.index).intersection(smb.index)\
                   .intersection(hml.index).intersection(rmw.index).intersection(cma.index)
    if len(common) < 12:
        return None
    y = (returns.loc[common] - risk_free.loc[common]).values
    X = np.column_stack([market_premium.loc[common].values, smb.loc[common].values,
                         hml.loc[common].values, rmw.loc[common].values, cma.loc[common].values])
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y - y.mean())**2)
    r_squared = 1 - (rss / tss)
    n = len(y)
    mse = rss / (n - 6)
    var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    std_errors = np.sqrt(np.diag(var_covar))
    t_stats = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - 6))
    return {
        'alpha': beta[0], 'mkt': beta[1], 'smb': beta[2], 'hml': beta[3],
        'rmw': beta[4], 'cma': beta[5],
        'r_squared': r_squared, 'alpha_se': std_errors[0], 'alpha_p': p_values[0],
        'n_obs': n
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📊 Configuration")
    
    if not YFINANCE_AVAILABLE:
        st.warning("⚠️ `yfinance` not installed. Individual stock download is disabled. Using European Portfolios only.")
        asset_type = "European Portfolios"
        st.markdown("**Asset Type**")
        st.info("European Portfolios (only option available)")
        selected_portfolio = st.selectbox("Select Portfolio", 
                                          ["Portfolio 1", "Portfolio 2", "Portfolio 3", "Portfolio 4", 
                                           "Portfolio 5", "Portfolio 10", "Portfolio 20", "Portfolio 25"])
    else:
        asset_type = st.radio("**Asset Type**", ["European Portfolios", "Individual Stocks"], 
                             label_visibility="collapsed", horizontal=True)
        
        if asset_type == "European Portfolios":
            st.markdown("**Select Portfolio**")
            selected_portfolio = st.selectbox("", ["Portfolio 1", "Portfolio 2", "Portfolio 3", "Portfolio 4", 
                                               "Portfolio 5", "Portfolio 10", "Portfolio 20", "Portfolio 25"],
                                             label_visibility="collapsed")
        else:
            st.markdown("**Enter Tickers**")
            ticker_input = st.text_input("Comma-separated", "ASML,SAP,HSBA,BBVA", 
                                        label_visibility="collapsed")
            selected_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    st.divider()
    
    st.markdown("**Macro Period**")
    selected_period = st.selectbox("", list(MACRO_PERIODS.keys()), label_visibility="collapsed")
    period_start, period_end = MACRO_PERIODS[selected_period]
    
    st.divider()
    
    st.markdown("**⚙️ Advanced**")
    risk_free_override = st.number_input("Risk-Free Rate (% annual)", 0.0, 10.0, 2.5, 0.1) / 100

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-block">
    <div>
        <div class="logo">FACTOR <span>MODEL</span> ANALYZER</div>
        <div class="subtitle">CAPM vs FF3 vs FF5 Comparison</div>
    </div>
    <div class="badge">📈 Analysis Ready</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION (only if data loaded)
# ═══════════════════════════════════════════════════════════════════════════════

if ff3 is None or ff5 is None:
    # This should not happen if we stopped earlier, but just in case
    st.error("Data not loaded. Please upload the required CSV files.")
    st.stop()

try:
    ff3_period = ff3[(ff3.index >= period_start) & (ff3.index <= period_end)].copy()
    ff5_period = ff5[(ff5.index >= period_start) & (ff5.index <= period_end)].copy()
    
    if ff3_period.empty:
        st.error("No data available for selected period.")
        st.stop()
    
    # Prepare returns
    if asset_type == "European Portfolios" or not YFINANCE_AVAILABLE:
        returns = ff3_period[['Mkt-RF']] * 0.5  # simplified proxy
        returns_display = f"**{selected_portfolio}** (Proxy)"
    else:
        returns_dict = {}
        for ticker in selected_tickers:
            try:
                df = yf.download(ticker, start=period_start, end=period_end, 
                               progress=False, auto_adjust=True)
                if not df.empty:
                    returns_dict[ticker] = df['Close'].pct_change() * 100
            except:
                pass
        if not returns_dict:
            st.warning("Could not fetch stock data. Using factor proxy.")
            returns = ff3_period[['Mkt-RF']]
            returns_display = "Factor Market Return (Proxy)"
        else:
            returns = pd.DataFrame(returns_dict).mean(axis=1) * 100
            returns_display = ", ".join(selected_tickers)
    
    common_idx = returns.index.intersection(ff3_period.index)
    if len(common_idx) < 12:
        st.error("Insufficient overlapping data.")
        st.stop()
    
    returns = returns.loc[common_idx]
    rf = ff3_period.loc[common_idx, 'RF']
    mkt = ff3_period.loc[common_idx, 'Mkt-RF']
    smb = ff3_period.loc[common_idx, 'SMB']
    hml = ff3_period.loc[common_idx, 'HML']
    rmw = ff5_period.loc[common_idx, 'RMW'] if 'RMW' in ff5_period.columns else pd.Series(0, index=common_idx)
    cma = ff5_period.loc[common_idx, 'CMA'] if 'CMA' in ff5_period.columns else pd.Series(0, index=common_idx)
    
    capm_res = run_capm(returns, rf, mkt)
    ff3_res = run_ff3(returns, rf, mkt, smb, hml)
    ff5_res = run_ff5(returns, rf, mkt, smb, hml, rmw, cma)
    
    if capm_res is None or ff3_res is None or ff5_res is None:
        st.error("Insufficient data for regression analysis.")
        st.stop()
    
    # ─── RESULTS DISPLAY ──────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-label"><span class="dot"></span> Analysis Results ({selected_period})</div>',
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-cell">
            <div class="stat-label">Asset</div>
            <div class="stat-value" style="font-size:1.2rem">{returns_display}</div>
            <div class="stat-sub">Analysis Target</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Period</div>
            <div class="stat-value" style="font-size:1rem">{capm_res['n_obs']} months</div>
            <div class="stat-sub">Observations</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Market Return</div>
            <div class="stat-value">{mkt.mean()*12:.1f}%</div>
            <div class="stat-sub">Annualized</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Risk-Free Rate</div>
            <div class="stat-value">{rf.mean()*12:.2f}%</div>
            <div class="stat-sub">Annualized</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sec-label"><span class="dot"></span> Model Comparison</div>',
                unsafe_allow_html=True)
    
    comparison_data = {
        'Metric': ['Alpha (%)', 'Alpha t-stat', 'Alpha p-value', 'R² (Goodness-of-Fit)', 'Model Factors', 'Observations'],
        'CAPM': [f"{capm_res['alpha']*100:.3f}", f"{capm_res['alpha_t']:.2f}", f"{capm_res['alpha_p']:.4f}", f"{capm_res['r_squared']:.4f}", "1 (Market)", f"{capm_res['n_obs']}"],
        'FF3': [f"{ff3_res['alpha']*100:.3f}", f"{ff3_res['alpha_t']:.2f}", f"{ff3_res['alpha_p']:.4f}", f"{ff3_res['r_squared']:.4f}", "3 (MKT, SMB, HML)", f"{ff3_res['n_obs']}"],
        'FF5': [f"{ff5_res['alpha']*100:.3f}", f"{ff5_res['alpha_t']:.2f}", f"{ff5_res['alpha_p']:.4f}", f"{ff5_res['r_squared']:.4f}", "5 (MKT, SMB, HML, RMW, CMA)", f"{ff5_res['n_obs']}"]
    }
    comp_df = pd.DataFrame(comparison_data)
    
    st.markdown('<div class="panel"><div class="panel-title">// Factor Model Regression Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**CAPM**\n- α = {capm_res['alpha']*100:.3f}% (t={capm_res['alpha_t']:.2f})\n- β = {capm_res['beta']:.3f}\n- R² = {capm_res['r_squared']:.4f}\n- n = {capm_res['n_obs']}")
    with col2:
        st.markdown(f"**Fama-French 3**\n- α = {ff3_res['alpha']*100:.3f}% (t={ff3_res['alpha_t']:.2f})\n- MKT β = {ff3_res['mkt']:.3f}\n- SMB β = {ff3_res['smb']:.3f}\n- HML β = {ff3_res['hml']:.3f}\n- R² = {ff3_res['r_squared']:.4f}")
    with col3:
        st.markdown(f"**Fama-French 5**\n- α = {ff5_res['alpha']*100:.3f}% (t={ff5_res['alpha_t']:.2f})\n- MKT β = {ff5_res['mkt']:.3f}\n- SMB β = {ff5_res['smb']:.3f}\n- HML β = {ff5_res['hml']:.3f}\n- RMW β = {ff5_res['rmw']:.3f}\n- CMA β = {ff5_res['cma']:.3f}\n- R² = {ff5_res['r_squared']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ─── RANKING ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label"><span class="dot"></span> Model Effectiveness Ranking</div>', unsafe_allow_html=True)
    r2_scores = {'CAPM': capm_res['r_squared'], 'FF3': ff3_res['r_squared'], 'FF5': ff5_res['r_squared']}
    alpha_sig = {'CAPM': abs(capm_res['alpha_p']) < 0.05, 'FF3': abs(ff3_res['alpha_p']) < 0.05, 'FF5': abs(ff5_res['alpha_p']) < 0.05}
    ranked = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
    
    ranking_html = '<div class="panel"><div class="panel-title">// Ranking by Explanatory Power (R²)</div><table><thead><tr><th>Rank</th><th>Model</th><th>R²</th><th>Improvement</th><th>Alpha Significant</th><th>Verdict</th></tr></thead><tbody>'
    for idx, (model, r2) in enumerate(ranked, 1):
        rank_class = f"rank-{idx}"
        badge = f'<span class="rank-badge {rank_class}">#{idx}</span>'
        improvement = "-" if idx == 1 else f"{(r2 - ranked[0][1]) / ranked[0][1] * 100:+.1f}%"
        is_sig = "✓ Yes" if alpha_sig[model] else "✗ No"
        sig_class = "metric-highlight" if alpha_sig[model] else ""
        verdict = "🏆 Best Model" if idx == 1 else ("🥈 Strong" if idx == 2 else "⚠️ Limited")
        verdict_class = "metric-highlight" if idx <= 2 else "metric-warn"
        ranking_html += f'<tr><td>{badge}</td><td><strong>{model}</strong></td><td><span class="metric-highlight">{r2:.4f}</span></td><td>{improvement}</td><td><span class="{sig_class}">{is_sig}</span></td><td><span class="{verdict_class}">{verdict}</span></td></tr>'
    ranking_html += '</tbody></table></div>'
    st.markdown(ranking_html, unsafe_allow_html=True)
    
    # ─── INSIGHTS ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label"><span class="dot"></span> Key Insights</div>', unsafe_allow_html=True)
    best_model = ranked[0][0]
    best_r2 = ranked[0][1]
    improvement_3_to_capm = (ff3_res['r_squared'] - capm_res['r_squared']) / capm_res['r_squared'] * 100
    improvement_5_to_3 = (ff5_res['r_squared'] - ff3_res['r_squared']) / ff3_res['r_squared'] * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🎯 Best Performer\n**{best_model}** dominates with R² = {best_r2:.4f}\n\n- Explains {best_r2*100:.1f}% of return variation\n- {'Alpha is statistically significant' if alpha_sig[best_model] else 'Alpha is not statistically significant'}\n- **Recommendation:** Use {best_model} for this period")
    with col2:
        st.markdown(f"### 📊 Factor Effectiveness\n- **FF3 → CAPM:** +{improvement_3_to_capm:.1f}% improvement\n- **FF5 → FF3:** {improvement_5_to_3:+.1f}% change\n\n{f'SMB/HML factors substantially improve explanatory power' if improvement_3_to_capm > 5 else 'SMB/HML offer modest marginal benefit'}\n\n{f'RMW/CMA factors add value' if improvement_5_to_3 > 0 else 'RMW/CMA provide diminishing returns'}")
    
    st.markdown('<div class="sec-label"><span class="dot"></span> Detailed Metrics</div>', unsafe_allow_html=True)
    st.dataframe(comp_df.style.format({'CAPM': '{}', 'FF3': '{}', 'FF5': '{}'}), use_container_width=True)
    
    # ─── EXPORT ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label"><span class="dot"></span> Export Results</div>', unsafe_allow_html=True)
    export_data = {
        'Model': ['CAPM', 'FF3', 'FF5'],
        'R_Squared': [capm_res['r_squared'], ff3_res['r_squared'], ff5_res['r_squared']],
        'Alpha_%': [capm_res['alpha']*100, ff3_res['alpha']*100, ff5_res['alpha']*100],
        'Alpha_t_stat': [capm_res['alpha_t'], ff3_res['alpha_t'], ff5_res['alpha_t']],
        'Alpha_p_value': [capm_res['alpha_p'], ff3_res['alpha_p'], ff5_res['alpha_p']],
        'Observations': [capm_res['n_obs'], ff3_res['n_obs'], ff5_res['n_obs']],
        'Period': [selected_period] * 3,
        'Asset': [returns_display] * 3
    }
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    st.download_button("📥 Download Results (CSV)", csv.encode(), "factor_model_comparison.csv", "text/csv")

except Exception as e:
    st.error(f"Analysis Error: {e}")
    with st.expander("Debug Info"):
        import traceback
        st.code(traceback.format_exc())
