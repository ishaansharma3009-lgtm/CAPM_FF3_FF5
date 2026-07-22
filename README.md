# CAPM_FF3_FF5
# Factor Model Analyzer

A Streamlit app for comparing CAPM, the Fama-French three-factor model (FF3), and the Fama-French five-factor model (FF5) on European equities across different macroeconomic conditions.

Built to answer one question: **which asset pricing model best explains European stock returns, and does the answer change depending on the macroeconomic environment?**

## What it does

- Compares **CAPM vs. FF3 vs. FF5** using Newey-West (1987) HAC-adjusted regressions
- Tests all three models against **25 European portfolios** sorted on size and book-to-market (Kenneth French Data Library), using the **Gibbons-Ross-Shanken (1989) test** to check whether pricing errors are jointly zero across the full cross-section — not just a single aggregate series
- Splits the analysis across **three macroeconomic conditions**:
  - European Sovereign Debt Crisis (2010–2012)
  - Post-Crisis Recovery & Quantitative Easing (2013–2019)
  - COVID Shock & Post-Pandemic Inflation Cycle (2020–2024)
- Reports **adjusted R²**, **average absolute alpha**, and the **GRS F-statistic/p-value** for each model in each condition, side by side
- Visualises results as a per-condition model comparison chart, a 5×5 alpha heatmap (CAPM / FF3 / FF5 shown side by side), and a rolling 36-month market beta chart
- Exports results to CSV for further analysis (single-asset comparison, GRS summary, and per-portfolio alpha detail)

## Data sources

- **Kenneth French Data Library** — European factor data (Mkt-RF, SMB, HML, RMW, CMA) and the 25 Size/Book-to-Market portfolios, value-weighted, monthly
- **ECB Data Portal** — Euribor 1-month rate, used to reconstruct a Euro area risk-free rate and market factor in place of the US T-bill rate that French's regional factor files use by default (see *Methodology notes* below)

All data is embedded directly in the app — no external downloads or API keys needed to run it.

## Methodology notes

A couple of implementation details worth knowing if you're digging into the code:

- **Risk-free rate.** Kenneth French's regional factor files (Europe, Japan, Asia-Pacific) share a single risk-free column — the US one-month T-bill rate — regardless of region. This app reconstructs a Euribor-based risk-free rate and market factor instead:
  ```
  Mkt-RF (Euribor-based) = Mkt-RF (French, as provided) + RF (US T-bill) − RF (Euribor)
  ```
  SMB, HML, RMW, and CMA are long-short, self-financing portfolios that never subtract a risk-free rate in construction, so they're used as-is.
- **Standard errors.** All regressions use Newey-West (1987) HAC-consistent standard errors rather than plain OLS, since monthly equity return residuals routinely violate the homoskedasticity/no-autocorrelation assumptions.
- **GRS test.** There's no standard Python package for the Gibbons-Ross-Shanken test, so it's implemented directly from the original 1989 formula.

## Running it locally

```bash
pip install streamlit pandas numpy scipy
streamlit run factor_model_analyzer.py
```

No API keys, config files, or external data downloads required — everything the app needs is embedded in the script.

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling and regression math
- [SciPy](https://scipy.org/) — statistical distributions (t-test, F-test)

## References

- Fama, E.F. and French, K.R. (1993) 'Common risk factors in the returns on stocks and bonds', *Journal of Financial Economics*, 33(1), pp. 3–56.
- Fama, E.F. and French, K.R. (2015) 'A five-factor asset pricing model', *Journal of Financial Economics*, 116(1), pp. 1–22.
- Gibbons, M.R., Ross, S.A. and Shanken, J. (1989) 'A test of the efficiency of a given portfolio', *Econometrica*, 57(5), pp. 1121–1152.
- Newey, W.K. and West, K.D. (1987) 'A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix', *Econometrica*, 55(3), pp. 703–708.



MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Ishaan Sharma
