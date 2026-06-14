# Post-IPO Performance Analysis & SpaceX Valuation Proxy

A quantitative Python toolkit that analyses the post-listing performance of twelve high-profile 2019-2023 tech/"new economy" IPOs (Rivian, Palantir, Snowflake, Rocket Lab, Arm, Mobileye, Coinbase, AST SpaceMobile, Virgin Galactic, Planet Labs, BlackSky, Redwire) and uses them as historical proxies to model the risk profile of an anticipated **SpaceX (SPCX)** public listing.

The script downloads historical price and fundamental data via [yfinance](https://github.com/ranaroussi/yfinance), computes a series of valuation and risk metrics, and generates charts and CSV summaries for each analysis stage.

## Features

- **IPO Pop** – first-day price reaction (issue price vs. Day-1 open) for each ticker.
- **P/S Ratio compression** – quarterly Price-to-Sales ratio over time, tracking multiple "mean reversion".
- **Sharpe Ratio** – annualised risk-adjusted returns since listing.
- **Max Drawdown stress test** – worst peak-to-trough decline in the first 24 months post-IPO, translated into an implied dollar loss for a hypothetical SpaceX valuation.
- **Normalised price trajectories** – all tickers rebased to 100 at listing for direct comparison.
- **Directional matrix** – IPO Pop vs. Sharpe Ratio scatter plot.
- **SpaceX fundamentals model** – standalone S-1-style valuation built from 2025 segment revenues and growth projections to 2030.
- **Predictive regression model** – P/S ratio & IPO Pop vs. drawdown risk, applied to a SpaceX forecast.
- **Ornstein-Uhlenbeck simulation** – stochastic mean-reversion model for how SpaceX's P/S multiple could decay post-listing.

  ## Report

A full written analysis of the results (charts, tables, commentary) is available here: [SpaceX_IPO_Analysis_Report.pdf](docs/SpaceX_IPO_Analysis_Report.pdf)

## Project structure

ipo-spacex-analysis/

├── ipo_analysis.py        # main script (single entry point)

├── requirements.txt

├── LICENSE

├── .gitignore

└── README.md

All outputs (CSV files and PNG charts) are written to `ipo_analysis_output/`, created automatically on first run.

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - yfinance
  - scikit-learn

## Installation

```bash
git clone https://github.com/<your-username>/ipo-spacex-analysis.git
cd ipo-spacex-analysis
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python ipo_analysis.py
```

The script will:

1. Download historical prices, quarterly revenues and shares outstanding for all tickers.
2. Compute IPO Pop, P/S ratios, Sharpe ratios and max drawdowns.
3. Build the standalone SpaceX fundamentals model and growth projections.
4. Fit a regression model and run the Ornstein-Uhlenbeck simulation.
5. Save all charts (`.png`) and data tables (`.csv`) to `ipo_analysis_output/`.

## Configuration

All key parameters are defined at the top of `ipo_analysis.py` and can be edited directly:

- `TICKERS`, `IPO_DATES`, `IPO_PRICES` – the comparable basket.
- `SPACEX` – target valuation, cash/debt, 2025 segment revenues, 2030 projections.
- `RISK_FREE_RATE`, `DRAWDOWN_MONTHS` – assumptions for Sharpe ratio and drawdown window.
- `OU_PARAMS` – parameters for the Ornstein-Uhlenbeck simulation (mean-reversion target, speed, volatility, horizon, number of paths).

## Disclaimer

This project is for educational and illustrative purposes only. It does **not** constitute financial advice or an investment recommendation. SpaceX figures are based on public estimates and modelling assumptions, and a real SpaceX IPO may differ materially from any projections shown here.

## License

This project is licensed under the [MIT License](LICENSE).
