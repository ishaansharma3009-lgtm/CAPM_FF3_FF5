import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import io

# ---------------------------------------------------------------------------
# 1. Data loading functions (adapted for uploaded files)
# ---------------------------------------------------------------------------

REGIMES = {
    1: {"label": "European sovereign debt crisis", "start": "2010-01", "end": "2012-12", "n_obs": 36},
    2: {"label": "Post-crisis recovery / QE era",   "start": "2013-01", "end": "2019-12", "n_obs": 84},
    3: {"label": "COVID shock / post-pandemic inflation", "start": "2020-01", "end": "2024-12", "n_obs": 60},
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

def run_adf(series, regression="c", autolag="AIC"):
    s = series.dropna()
    stat, pvalue, usedlag, nobs, crit_values, icbest = adfuller(
        s, regression=regression, autolag=autolag
    )
    return {
        "adf_stat": stat,
        "p_value": pvalue,
        "used_lag": usedlag,
        "n_obs": nobs,
        "crit_1%": crit_values["1%"],
        "crit_5%": crit_values["5%"],
        "crit_10%": crit_values["10%"],
    }

def adf_table(factors_df, factor_cols):
    rows = []
    for col in factor_cols:
        if col not in factors_df.columns:
            continue
        r = run_adf(factors_df[col])
        r["factor"] = col
        rows.append(r)
    out = pd.DataFrame(rows).set_index("factor")
    out = out[["adf_stat", "p_value", "used_lag", "n_obs", "crit_1%", "crit_5%", "crit_10%"]]
    return out

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

st.set_page_config(page_title="CAPM/FF3/FF5 Regime Comparison", layout="wide")
st.title("📈 CAPM / FF3 / FF5 Regime‑Comparison Pipeline")
st.markdown("Upload the required data files and click **Run Analysis**.")

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
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
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
                ALL_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                MODEL_ASSET_SETS = {"CAPM": ["beme"], "FF3": ["beme"], "FF5": ["beme", "op", "inv"]}
                ASSET_SET_LABELS = {
                    "beme": "ME / BE-ME (size & book-to-market)",
                    "op": "ME / OP (size & operating profitability)",
                    "inv": "ME / INV (size & investment)",
                }

                # Validation
                st.subheader("✅ Validation checks")
                for rid, spec in REGIMES.items():
                    n = len(ff3_regimes[rid])
                    status = "✅" if n == spec["n_obs"] else "❌"
                    st.write(f"Regime {rid} ({spec['label']}): {n} obs (expected {spec['n_obs']}) {status}")
                st.write(f"Missing RF months: {panel['missing_rf_months']}")

                # ADF
                st.subheader("📊 ADF Stationarity Tests")
                adf_rows = []
                for rid in REGIMES:
                    tbl = adf_table(ff5_regimes[rid], ALL_FACTOR_COLS)
                    tbl.insert(0, "regime", rid)
                    tbl.insert(1, "regime_label", REGIMES[rid]["label"])
                    adf_rows.append(tbl)
                adf_full = pd.concat(adf_rows)
                st.dataframe(adf_full[["regime", "p_value"]])
                cma_r1 = adf_full.loc[(adf_full["regime"] == 1) & (adf_full.index == "CMA"), "p_value"]
                rmw_r3 = adf_full.loc[(adf_full["regime"] == 3) & (adf_full.index == "RMW"), "p_value"]
                if len(cma_r1):
                    st.write(f"CMA Regime 1 p-value: {cma_r1.values[0]:.3f} (expected ~0.193, unit-root not rejected)")
                if len(rmw_r3):
                    st.write(f"RMW Regime 3 p-value: {rmw_r3.values[0]:.3f} (expected ~0.064, not rejected @5%, rejected @10%)")

                # Descriptive stats
                st.subheader("📈 Descriptive Statistics")
                desc_rows = []
                for rid in REGIMES:
                    df = ff5_regimes[rid][ALL_FACTOR_COLS]
                    mean = df.mean()
                    std = df.std()
                    sharpe = mean / std
                    for col in ALL_FACTOR_COLS:
                        desc_rows.append({
                            "regime": rid,
                            "regime_label": REGIMES[rid]["label"],
                            "factor": col,
                            "mean_monthly_pct": mean[col],
                            "std_monthly_pct": std[col],
                            "sharpe_ratio": sharpe[col],
                        })
                desc_df = pd.DataFrame(desc_rows)
                st.dataframe(desc_df)

                # Run regression blocks
                st.subheader("📉 Regression Results")
                block_summary_rows = []
                per_portfolio_alpha_rows = []
                for model_name in ["CAPM", "FF3", "FF5"]:
                    for asset_set in MODEL_ASSET_SETS[model_name]:
                        for rid in REGIMES:
                            factors_df = factors_by_model[model_name][rid]
                            port_ret = excess_by_asset[asset_set][rid]
                            block = run_block(port_ret, factors_df, model_name)
                            grs = grs_test(block["alphas"], block["resid_mat"], factors_df, MODEL_FACTORS[model_name])
                            avg_adj_r2 = block["adj_r2"].mean()
                            avg_abs_alpha = block["alphas"].abs().mean()
                            block_summary_rows.append({
                                "model": model_name,
                                "regime": rid,
                                "regime_label": REGIMES[rid]["label"],
                                "test_assets": ASSET_SET_LABELS[asset_set],
                                "n_obs": block["n_obs"],
                                "n_portfolios": len(port_ret.columns),
                                "k_factors": block["k_factors"],
                                "newey_west_lag": block["lag"],
                                "avg_adj_r2": avg_adj_r2,
                                "avg_abs_alpha_pct": avg_abs_alpha,
                                "GRS_F": grs["GRS_F"],
                                "GRS_df1": grs["df1"],
                                "GRS_df2": grs["df2"],
                                "GRS_p_value": grs["p_value"],
                            })
                            for pf_name in block["alphas"].index:
                                per_portfolio_alpha_rows.append({
                                    "model": model_name,
                                    "regime": rid,
                                    "test_assets": asset_set,
                                    "portfolio": pf_name,
                                    "alpha_pct": block["alphas"][pf_name],
                                    "adj_r2": block["adj_r2"][pf_name],
                                })
                block_summary = pd.DataFrame(block_summary_rows)

                # Primary 3x3 matrix
                primary = block_summary[block_summary["test_assets"] == ASSET_SET_LABELS["beme"]]
                matrix_rows = []
                for model_name in ["CAPM", "FF3", "FF5"]:
                    for rid in REGIMES:
                        row = primary[(primary["model"] == model_name) & (primary["regime"] == rid)].iloc[0]
                        matrix_rows.append({
                            "model": model_name,
                            "regime": rid,
                            "regime_label": REGIMES[rid]["label"],
                            "adj_r2": row["avg_adj_r2"],
                            "avg_abs_alpha_pct": row["avg_abs_alpha_pct"],
                            "GRS_F": row["GRS_F"],
                            "GRS_p_value": row["GRS_p_value"],
                        })
                results_matrix = pd.DataFrame(matrix_rows)
                st.write("### Primary Results (BE-ME portfolios)")
                st.dataframe(results_matrix)

                # Pivoted tables
                st.write("### Pivoted tables")
                pivot_adj_r2 = results_matrix.pivot(index="model", columns="regime_label", values="adj_r2")
                pivot_alpha = results_matrix.pivot(index="model", columns="regime_label", values="avg_abs_alpha_pct")
                pivot_grs_f = results_matrix.pivot(index="model", columns="regime_label", values="GRS_F")
                pivot_grs_p = results_matrix.pivot(index="model", columns="regime_label", values="GRS_p_value")
                st.write("**Adj. R²**")
                st.dataframe(pivot_adj_r2)
                st.write("**Avg |α|**")
                st.dataframe(pivot_alpha)
                st.write("**GRS F-statistic**")
                st.dataframe(pivot_grs_f)
                st.write("**GRS p-value**")
                st.dataframe(pivot_grs_p)

                # Download buttons
                st.subheader("📥 Download Results")
                csv_data = {
                    "results_matrix_3x3_beme.csv": results_matrix,
                    "matrix_adj_r2.csv": pivot_adj_r2,
                    "matrix_avg_abs_alpha.csv": pivot_alpha,
                    "matrix_grs_f.csv": pivot_grs_f,
                    "matrix_grs_pvalue.csv": pivot_grs_p,
                    "block_summary_all.csv": block_summary,
                    "adf_test_results.csv": adf_full,
                    "descriptive_statistics.csv": desc_df,
                    "per_portfolio_alphas.csv": pd.DataFrame(per_portfolio_alpha_rows),
                    "rf_crosscheck_french_vs_euro.csv": panel["rf_check"],
                }
                for fname, df in csv_data.items():
                    csv = df.to_csv(index=True)
                    st.download_button(f"⬇️ Download {fname}", data=csv, file_name=fname, mime="text/csv")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

else:
    st.info("Please upload all 7 required CSV files to enable the analysis.")
