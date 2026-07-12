import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import io

# ---------------------------------------------------------------------------
# 1. Data loading & models (same as before)
# ---------------------------------------------------------------------------

REGIMES = {
    1: {"label": "European sovereign debt crisis", "start": "2010-01", "end": "2012-12", "n_obs": 36,
        "desc": "Sovereign debt tensions in peripheral Eurozone countries; high volatility."},
    2: {"label": "Post-crisis recovery / QE era", "start": "2013-01", "end": "2019-12", "n_obs": 84,
        "desc": "Gradual economic recovery, ECB quantitative easing, low inflation."},
    3: {"label": "COVID shock / post-pandemic inflation", "start": "2020-01", "end": "2024-12", "n_obs": 60,
        "desc": "Pandemic-induced market crash, rapid recovery, then high inflation and rate hikes."},
}

FULL_SAMPLE_START = "2010-01"
FULL_SAMPLE_END = "2024-12"
RF_TRANSITION_MONTH = "2019-10"

def _load_french_factor_file(content, factor_cols):
    lines = content.decode("utf-8").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().split(",")]
        if len(cells) >= 1 + len(factor_cols) and cells[1:1 + len(factor_cols)] == factor_cols:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate header row")
    records = []
    for line in lines[header_idx + 1:]:
        cells = [c.strip() for c in line.strip().split(",")]
        if not cells or cells[0] == "":
            break
        key = cells[0]
        if not (key.isdigit() and len(key) == 6):
            break
        records.append(cells[: 1 + len(factor_cols)])
    df = pd.DataFrame(records, columns=["yyyymm"] + factor_cols)
    df["date"] = pd.PeriodIndex(df["yyyymm"], freq="M")
    df = df.drop(columns=["yyyymm"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(-99.99, np.nan)
    return df

def load_ff3(content):
    return _load_french_factor_file(content, ["Mkt-RF", "SMB", "HML", "RF"])

def load_ff5(content):
    return _load_french_factor_file(content, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

def _load_25_portfolios(content):
    lines = content.decode("utf-8").splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "Average Value Weighted Returns" in line and "Monthly" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find Value Weighted Monthly block")
    header_idx = start_idx + 1
    header_cells = [c.strip() for c in lines[header_idx].strip().split(",")]
    col_names = [c for c in header_cells if c != ""]
    records = []
    for line in lines[header_idx + 1:]:
        cells = [c.strip() for c in line.strip().split(",")]
        if not cells or cells[0] == "":
            break
        key = cells[0]
        if not (key.isdigit() and len(key) == 6):
            break
        records.append(cells[: 1 + len(col_names)])
    df = pd.DataFrame(records, columns=["yyyymm"] + col_names)
    df["date"] = pd.PeriodIndex(df["yyyymm"], freq="M")
    df = df.drop(columns=["yyyymm"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(-99.99, np.nan)
    return df

def load_portfolios_beme(content):
    return _load_25_portfolios(content)

def load_portfolios_op(content):
    return _load_25_portfolios(content)

def load_portfolios_inv(content):
    return _load_25_portfolios(content)

def _load_euribor_1m(content):
    df = pd.read_csv(io.BytesIO(content))
    df.columns = ["date", "time_period", "rate_annual_pct"]
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = pd.PeriodIndex(df["date"], freq="M")
    df = df.set_index("month")[["rate_annual_pct"]]
    df["rate_annual_pct"] = pd.to_numeric(df["rate_annual_pct"], errors="coerce")
    return df

def _load_estr(content):
    df = pd.read_csv(io.BytesIO(content))
    df.columns = ["date", "time_period", "rate_annual_pct"]
    df["date"] = pd.to_datetime(df["date"])
    df["rate_annual_pct"] = pd.to_numeric(df["rate_annual_pct"], errors="coerce")
    df["month"] = pd.PeriodIndex(df["date"], freq="M")
    monthly = df.groupby("month")["rate_annual_pct"].mean().to_frame()
    return monthly

def build_rf_series(euribor_content, estr_content):
    euribor = _load_euribor_1m(euribor_content)
    estr = _load_estr(estr_content)
    euribor["monthly_pct"] = ((1 + euribor["rate_annual_pct"] / 100) ** (1 / 12) - 1) * 100
    estr["monthly_pct"] = ((1 + estr["rate_annual_pct"] / 100) ** (1 / 12) - 1) * 100
    transition = pd.Period(RF_TRANSITION_MONTH, freq="M")
    euribor_part = euribor.loc[euribor.index < transition, "monthly_pct"]
    estr_part = estr.loc[estr.index >= transition, "monthly_pct"]
    rf = pd.concat([euribor_part, estr_part]).sort_index()
    rf = rf[~rf.index.duplicated(keep="last")]
    rf.name = "RF_euro"
    return rf.to_frame()

def build_master_panel(files):
    ff3 = load_ff3(files["ff3"]).rename(columns={"RF": "RF_french"})
    ff5 = load_ff5(files["ff5"]).rename(columns={"RF": "RF_french"})
    beme = load_portfolios_beme(files["beme"])
    op = load_portfolios_op(files["op"])
    inv = load_portfolios_inv(files["inv"])
    rf_euro = build_rf_series(files["euribor"], files["estr"])
    full_idx = pd.period_range(FULL_SAMPLE_START, FULL_SAMPLE_END, freq="M")
    def _clip(df):
        return df.reindex(full_idx)
    ff3, ff5 = _clip(ff3), _clip(ff5)
    beme, op, inv = _clip(beme), _clip(op), _clip(inv)
    rf_euro = _clip(rf_euro)
    rf_check = pd.DataFrame({
        "RF_french": ff3["RF_french"],
        "RF_euro": rf_euro["RF_euro"],
    })
    rf_check["diff"] = rf_check["RF_euro"] - rf_check["RF_french"]
    excess_returns = {
        "beme": beme.sub(rf_euro["RF_euro"], axis=0),
        "op": op.sub(rf_euro["RF_euro"], axis=0),
        "inv": inv.sub(rf_euro["RF_euro"], axis=0),
    }
    factors_excess = {
        "ff3": ff3.drop(columns=["RF_french"]).join(rf_euro),
        "ff5": ff5.drop(columns=["RF_french"]).join(rf_euro),
    }
    missing_rf_months = rf_euro["RF_euro"].isna().sum()
    return {
        "ff3": ff3,
        "ff5": ff5,
        "portfolios": {"beme": beme, "op": op, "inv": inv},
        "rf": rf_euro,
        "rf_check": rf_check,
        "excess_returns": excess_returns,
        "factors_excess": factors_excess,
        "missing_rf_months": missing_rf_months,
    }

def split_by_regime(df):
    out = {}
    for rid, spec in REGIMES.items():
        start = pd.Period(spec["start"], freq="M")
        end = pd.Period(spec["end"], freq="M")
        sub = df.loc[(df.index >= start) & (df.index <= end)]
        out[rid] = sub
    return out

# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------

MODEL_FACTORS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}

def newey_west_lags(n_obs):
    lag = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
    return max(lag, 1)

def run_ols_hac(y, X_df):
    aligned = pd.concat([y, X_df], axis=1).dropna()
    yy = aligned.iloc[:, 0]
    XX = sm.add_constant(aligned.iloc[:, 1:])
    n = len(aligned)
    lag = newey_west_lags(n)
    model = sm.OLS(yy, XX)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True})
    return res, lag, n

def run_block(portfolio_returns, factors_df, model_name):
    factor_cols = MODEL_FACTORS[model_name]
    X = factors_df[factor_cols]
    alphas = {}
    adj_r2 = {}
    resid_dict = {}
    lag_used = None
    n_used = None
    for col in portfolio_returns.columns:
        y = portfolio_returns[col]
        res, lag, n = run_ols_hac(y, X)
        alphas[col] = res.params["const"]
        adj_r2[col] = res.rsquared_adj
        resid_dict[col] = res.resid
        lag_used = lag
        n_used = n
    alphas = pd.Series(alphas)
    adj_r2 = pd.Series(adj_r2)
    resid_mat = pd.DataFrame(resid_dict)
    return {
        "alphas": alphas,
        "adj_r2": adj_r2,
        "resid_mat": resid_mat,
        "lag": lag_used,
        "n_obs": n_used,
        "k_factors": len(factor_cols),
    }

def grs_test(alphas, resid_mat, factors_df, factor_cols):
    aligned = pd.concat([resid_mat, factors_df[factor_cols]], axis=1).dropna()
    resid_aligned = aligned[resid_mat.columns]
    factors_aligned = aligned[factor_cols]
    T = len(aligned)
    N = resid_mat.shape[1]
    K = len(factor_cols)
    alpha_vec = alphas.loc[resid_mat.columns].values.reshape(-1, 1)
    Sigma = (resid_aligned.T @ resid_aligned).values / T
    mu = factors_aligned.mean().values.reshape(-1, 1)
    Omega = np.cov(factors_aligned.values, rowvar=False, ddof=0)
    if K == 1:
        Omega = np.array([[Omega]]) if np.isscalar(Omega) else Omega.reshape(1, 1)
    Sigma_inv = np.linalg.inv(Sigma)
    Omega_inv = np.linalg.inv(Omega)
    quad_alpha = (alpha_vec.T @ Sigma_inv @ alpha_vec).item()
    quad_mu = (mu.T @ Omega_inv @ mu).item()
    df1 = N
    df2 = T - N - K
    if df2 <= 0:
        return {"GRS_F": np.nan, "df1": df1, "df2": df2, "p_value": np.nan, "note": "Insufficient df"}
    F = ((T - N - K) / N) * (quad_alpha / (1 + quad_mu))
    p_value = 1 - stats.f.cdf(F, df1, df2)
    return {"GRS_F": F, "df1": df1, "df2": df2, "p_value": p_value, "note": ""}

# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CAPM/FF3/FF5 Interactive Comparison", layout="wide")
st.title("📈 CAPM / FF3 / FF5 – Interactive Regime & Stock Selection")

st.markdown("""
Upload the 7 required CSV files, then **choose which regimes and portfolios** you want to analyze.
The app will show you how CAPM, FF3, and FF5 perform for your selections.
""")

required_files = {
    "ff3": "Europe_3_Factors.csv",
    "ff5": "Europe_5_Factors.csv",
    "beme": "Europe_25_Portfolios_ME_BE-ME.csv",
    "op": "Europe_25_Portfolios_ME_OP.csv",
    "inv": "Europe_25_Portfolios_ME_INV.csv",
    "euribor": "Euribor_1M.csv",
    "estr": "ESTR.csv",
}

uploaded = {}
cols = st.columns(3)
for i, (key, fname) in enumerate(required_files.items()):
    col = cols[i % 3]
    uploaded[key] = col.file_uploader(f"📄 {fname}", type="csv", key=key)

if all(uploaded.values()):
    if st.button("🚀 Load & Process Data", type="primary"):
        with st.spinner("Loading and processing data..."):
            try:
                files = {k: v.getvalue() for k, v in uploaded.items()}
                panel = build_master_panel(files)
                ff3_regimes = split_by_regime(panel["factors_excess"]["ff3"])
                ff5_regimes = split_by_regime(panel["factors_excess"]["ff5"])
                excess_beme_regimes = split_by_regime(panel["excess_returns"]["beme"])
                excess_op_regimes = split_by_regime(panel["excess_returns"]["op"])
                excess_inv_regimes = split_by_regime(panel["excess_returns"]["inv"])
                excess_by_asset = {"beme": excess_beme_regimes, "op": excess_op_regimes, "inv": excess_inv_regimes}
                factors_by_model = {"CAPM": ff3_regimes, "FF3": ff3_regimes, "FF5": ff5_regimes}

                # Run all regressions once and store results in session state
                all_results = []
                for model_name in ["CAPM", "FF3", "FF5"]:
                    for rid in REGIMES:
                        factors_df = factors_by_model[model_name][rid]
                        port_ret = excess_by_asset["beme"][rid]   # use BE/ME portfolios for main selection
                        block = run_block(port_ret, factors_df, model_name)
                        grs = grs_test(block["alphas"], block["resid_mat"], factors_df, MODEL_FACTORS[model_name])
                        for pf in block["alphas"].index:
                            all_results.append({
                                "model": model_name,
                                "regime": rid,
                                "regime_label": REGIMES[rid]["label"],
                                "portfolio": pf,
                                "alpha": block["alphas"][pf],
                                "adj_r2": block["adj_r2"][pf],
                                "grs_f": grs["GRS_F"],
                                "grs_p": grs["p_value"],
                                "n_obs": block["n_obs"],
                                "k_factors": block["k_factors"],
                            })
                st.session_state["results"] = pd.DataFrame(all_results)
                st.success("✅ Data processed! Now choose your selections below.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

# ---------------------------------------------------------------------------
# Interactive selection (only if results exist)
# ---------------------------------------------------------------------------

if "results" in st.session_state and st.session_state["results"] is not None:
    df = st.session_state["results"]

    st.header("🎯 Select your analysis parameters")

    # Regime selection
    regime_labels = df["regime_label"].unique().tolist()
    selected_regimes = st.multiselect(
        "Choose macroeconomic regime(s)",
        options=regime_labels,
        default=regime_labels,  # select all by default
    )

    # Portfolio selection
    portfolio_names = sorted(df["portfolio"].unique().tolist())
    select_all = st.checkbox("Select all portfolios", value=True)
    if select_all:
        selected_portfolios = portfolio_names
    else:
        selected_portfolios = st.multiselect(
            "Choose portfolios (see legend below)",
            options=portfolio_names,
            default=portfolio_names[:5],  # a few defaults
        )

    if selected_regimes and selected_portfolios:
        filtered = df[
            (df["regime_label"].isin(selected_regimes)) &
            (df["portfolio"].isin(selected_portfolios))
        ]

        if filtered.empty:
            st.warning("No data for these selections. Please adjust your choices.")
        else:
            # Show summary table
            st.subheader("📊 Model Comparison for Selected Stocks & Regimes")
            # Group by model and regime, compute averages
            summary = filtered.groupby(["model", "regime_label"]).agg({
                "adj_r2": "mean",
                "alpha": lambda x: x.abs().mean(),
                "grs_f": "first",
                "grs_p": "first",
                "n_obs": "first",
                "k_factors": "first",
            }).reset_index()
            summary["avg_abs_alpha"] = summary["alpha"]
            summary = summary.rename(columns={
                "adj_r2": "Avg Adj R²",
                "avg_abs_alpha": "Avg |α|",
                "grs_f": "GRS F",
                "grs_p": "GRS p-value",
            })
            summary = summary[["model", "regime_label", "Avg Adj R²", "Avg |α|", "GRS F", "GRS p-value"]]
            st.dataframe(summary.style.format({
                "Avg Adj R²": "{:.4f}",
                "Avg |α|": "{:.4f}",
                "GRS F": "{:.4f}",
                "GRS p-value": "{:.4f}",
            }))

            # Bar chart comparing models
            st.subheader("📊 Model Comparison (Bar Chart)")
            chart_data = summary.pivot(index="regime_label", columns="model", values="Avg Adj R²")
            st.bar_chart(chart_data)

            # Also show a chart for Avg |α| (lower is better)
            st.subheader("📊 Average Absolute Alpha (lower is better)")
            alpha_chart = summary.pivot(index="regime_label", columns="model", values="Avg |α|")
            st.bar_chart(alpha_chart)

            # Show selected portfolios legend
            with st.expander("📋 Selected portfolios"):
                st.write(selected_portfolios)

            # Download filtered data
            csv = filtered.to_csv(index=False)
            st.download_button("⬇️ Download selected results (CSV)", data=csv, file_name="selected_analysis.csv", mime="text/csv")

    else:
        st.info("Select at least one regime and one portfolio to see results.")

else:
    st.info("Please upload the data files and click 'Load & Process Data' first.")
