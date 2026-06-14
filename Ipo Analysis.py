import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Optional, Dict, Tuple, List
from sklearn.linear_model import LinearRegression


TICKERS = ["RIVN", "PLTR", "SNOW", "RKLB", "ARM", "MBLY", "COIN", "ASTS", "SPCE", "PL", "BKSY", "RDW"]

IPO_DATES = {
    "RIVN": "2021-11-10", "PLTR": "2020-09-30", "SNOW": "2020-09-16",
    "RKLB": "2021-08-25", "ARM":  "2023-09-14", "MBLY": "2022-10-26",
    "COIN": "2021-04-14", "ASTS": "2021-04-07", "SPCE": "2019-10-28",
    "PL":   "2021-12-08", "BKSY": "2021-09-10", "RDW":  "2021-09-03",
}

IPO_PRICES = {
    "RIVN": 78.00, "PLTR":  7.25, "SNOW": 120.00, "RKLB": 11.00,
    "ARM":  51.00, "MBLY": 21.00, "COIN": 250.00,  "ASTS": 10.00,
    "SPCE": 10.00, "PL":   10.00, "BKSY": 10.00,   "RDW":  10.00,
}

SPACEX = {
    "target_valuation_bn": 1780,
    "cash_bn": 25,
    "debt_bn": 23,
    "revenue_2025": {"Starlink": 11.4, "Space (Launch/HW)": 4.1, "AI (xAI/X)": 3.2},
    "projection_ai_2030_bn": 322,
    "projection_total_2030_bn": 474,
}

RISK_FREE_RATE  = 0.042
DRAWDOWN_MONTHS = 24
OUTPUT_DIR      = "ipo_analysis_output"

OU_PARAMS = {"mu": 20.0, "theta": 2.5, "sigma": 35.0, "years": 2,
             "days_per_year": 252, "n_sim": 1000}


def build_palette(tickers: List[str]) -> Dict[str, str]:
    palette = sns.color_palette("husl", len(tickers)).as_hex()
    return {t: palette[i] for i, t in enumerate(tickers)}


def save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


class DataFetcher:
    """Downloads prices, revenues and shares outstanding from Yahoo Finance."""

    def __init__(self, tickers: List[str], ipo_dates: Dict[str, str]):
        self.tickers   = tickers
        self.ipo_dates = ipo_dates

    def prices(self) -> pd.DataFrame:
        frames = []
        for ticker, start in self.ipo_dates.items():
            s = yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"]
            s.name = ticker
            frames.append(s)
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index)
        return df

    def revenues(self) -> pd.DataFrame:
        rows = []
        for ticker in self.tickers:
            try:
                fin = yf.Ticker(ticker).quarterly_income_stmt
                if "Total Revenue" not in fin.index:
                    continue
                for date, value in fin.loc["Total Revenue"].items():
                    rows.append({"Ticker": ticker,
                                 "Quarter": pd.to_datetime(date).date(),
                                 "Revenue": value})
            except Exception:
                pass
        return pd.DataFrame(rows).sort_values(["Ticker", "Quarter"]).reset_index(drop=True)

    def shares_outstanding(self) -> Dict[str, int]:
        result = {}
        for ticker in self.tickers:
            info = yf.Ticker(ticker).info
            shares = info.get("sharesOutstanding") or 100_000_000
            result[ticker] = shares
        return result

    def day1_open(self, ticker: str, ipo_date: str) -> Optional[float]:
        end = (pd.to_datetime(ipo_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=ipo_date, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return None
        col = data["Open"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return float(col.iloc[0])


class IpoPopAnalyzer:
    """Computes and plots the first-day price pop for each IPO."""

    def __init__(self, tickers, ipo_dates, ipo_prices, fetcher: DataFetcher, colors: dict):
        self.tickers    = tickers
        self.ipo_dates  = ipo_dates
        self.ipo_prices = ipo_prices
        self.fetcher    = fetcher
        self.colors     = colors

    def compute(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Ticker":         self.tickers,
            "IPO_Date":       [self.ipo_dates[t]  for t in self.tickers],
            "Issue_Price":    [self.ipo_prices[t] for t in self.tickers],
        })
        df["Day1_Open"] = df.apply(
            lambda r: self.fetcher.day1_open(r["Ticker"], r["IPO_Date"]), axis=1)
        df["IPO_Pop_%"] = ((df["Day1_Open"] - df["Issue_Price"]) / df["Issue_Price"] * 100).round(2)
        return df

    def plot(self, df: pd.DataFrame) -> plt.Figure:
        x, w = np.arange(len(df)), 0.35
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - w/2, df["Issue_Price"], w, label="Issue Price ($)",  color="#1f77b4")
        ax.bar(x + w/2, df["Day1_Open"],   w, label="Day-1 Open ($)",   color="#2ca02c")
        for i, val in enumerate(df["IPO_Pop_%"]):
            if pd.notna(val):
                top  = max(df["Issue_Price"].iloc[i], df["Day1_Open"].iloc[i])
                text = f"+{val}%" if val > 0 else f"{val}%"
                ax.text(x[i], top * 1.05, text, ha="center", va="bottom",
                        fontweight="bold", color="green" if val > 0 else "red", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Ticker"], fontsize=10, rotation=45)
        ax.set_ylabel("Price ($)")
        ax.set_title("IPO Pop: Issue Price vs Day-1 Open", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        return fig


class PsRatioAnalyzer:
    """Calculates the quarterly Price-to-Sales ratio and plots multiple compression."""

    def __init__(self, prices: pd.DataFrame, revenues: pd.DataFrame, shares: dict, colors: dict):
        self.prices   = prices
        self.revenues = revenues
        self.shares   = shares
        self.colors   = colors

    def compute(self) -> pd.DataFrame:
        rows = []
        for ticker, group in self.revenues.groupby("Ticker"):
            if ticker not in self.prices.columns:
                continue
            px = self.prices[ticker].dropna()
            for _, row in group.iterrows():
                date, rev = pd.Timestamp(row["Quarter"]), row["Revenue"]
                if pd.isna(rev) or rev <= 0:
                    continue
                window = px.loc[
                    (px.index >= date - pd.Timedelta(days=45)) &
                    (px.index <= date + pd.Timedelta(days=45))
                ]
                if window.empty:
                    continue
                ann_rev_per_share = (rev * 4) / self.shares[ticker]
                rows.append({"Ticker": ticker, "Date": date,
                             "Avg_Price": round(window.mean(), 2),
                             "PS_Ratio":  round(window.mean() / ann_rev_per_share, 2)})
        return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def plot(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 7))
        for ticker, group in df.groupby("Ticker"):
            ax.plot(group["Date"], group["PS_Ratio"], label=ticker,
                    color=self.colors[ticker], linewidth=2, marker="o", markersize=3)
        ax.axhline(20, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="P/S 20x target")
        ax.set_title("Multiple Compression: P/S Ratio Over Time", fontweight="bold")
        ax.set_ylabel("P/S Ratio")
        ax.set_xlabel("Date")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig


class SharpeAnalyzer:
    """Computes annualised Sharpe ratio for each ticker and visualises results."""

    def __init__(self, prices: pd.DataFrame, risk_free: float, colors: dict):
        self.prices    = prices
        self.risk_free = risk_free
        self.colors    = colors

    def compute(self) -> pd.DataFrame:
        ret = self.prices.pct_change().dropna(how="all")
        rows = []
        for ticker in self.prices.columns:
            r = ret[ticker].dropna()
            if r.empty:
                continue
            ann_ret = r.mean() * 252
            ann_vol = r.std()  * 252 ** 0.5
            rows.append({
                "Ticker":          ticker,
                "Annual_Return_%": round(ann_ret * 100, 2),
                "Annual_Vol_%":    round(ann_vol * 100, 2),
                "Sharpe_Ratio":    round((ann_ret - self.risk_free) / ann_vol, 3),
            })
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame) -> plt.Figure:
        bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Sharpe_Ratio"]]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(df["Ticker"], df["Sharpe_Ratio"], color=bar_colors,
                      edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Sharpe = 1")
        for bar, val in zip(bars, df["Sharpe_Ratio"]):
            offset = 0.05 if val >= 0 else -0.15
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:.2f}", ha="center", fontweight="bold", fontsize=9)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio Post-IPO by Ticker", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig


class DrawdownAnalyzer:
    """Measures the maximum drawdown in the first N months after each IPO."""

    def __init__(self, prices: pd.DataFrame, ipo_dates: dict, months: int, colors: dict):
        self.prices    = prices
        self.ipo_dates = ipo_dates
        self.months    = months
        self.colors    = colors

    def compute(self, spacex_valuation_bn: float) -> pd.DataFrame:
        rows = []
        for ticker, ipo_date in self.ipo_dates.items():
            if ticker not in self.prices.columns:
                continue
            start = pd.Timestamp(ipo_date)
            end   = start + pd.DateOffset(months=self.months)
            px    = self.prices[ticker].dropna().loc[start:end]
            if px.empty:
                continue
            dd     = (px / px.cummax() - 1) * 100
            max_dd = dd.min()
            rows.append({
                "Ticker":          ticker,
                "Max_Drawdown_%":  round(max_dd, 2),
                "Peak_Date":       px.idxmax().date(),
                "Trough_Date":     dd.idxmin().date(),
                "SpaceX_Loss_bn":  round(spacex_valuation_bn * (max_dd / 100), 0),
            })
        return pd.DataFrame(rows)

    def plot(self, df_summary: pd.DataFrame) -> plt.Figure:
        tickers = df_summary["Ticker"].tolist()
        ncols   = 4
        nrows   = int(np.ceil(len(tickers) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3), sharey=False)
        axes = axes.flatten()

        for i, ticker in enumerate(tickers):
            start = pd.Timestamp(self.ipo_dates[ticker])
            end   = start + pd.DateOffset(months=self.months)
            px    = self.prices[ticker].dropna().loc[start:end]
            dd    = (px / px.cummax() - 1) * 100
            max_dd = dd.min()
            ax = axes[i]
            ax.fill_between(dd.index, dd.values, color=self.colors[ticker], alpha=0.35)
            ax.plot(dd.index, dd.values, color=self.colors[ticker], linewidth=1.8)
            ax.axhline(max_dd, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(dd.index[-1], max_dd - 2, f" {max_dd:.1f}%",
                    va="top", fontsize=8, fontweight="bold")
            ax.set_title(ticker, fontsize=11, fontweight="bold")
            ax.grid(linestyle="--", alpha=0.4)
            ax.tick_params(axis="x", rotation=45, labelsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Stress Test: Drawdown in First {self.months} Months Post-IPO",
                     fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig


class SpaceXModel:
    """Builds the SpaceX S-1 financial model, revenue projections and SOTP breakdown."""

    def __init__(self, params: dict):
        self.p   = params
        self.rev = params["revenue_2025"]
        self.total_rev = sum(self.rev.values())
        self.ev  = params["target_valuation_bn"] + params["debt_bn"] - params["cash_bn"]
        self.ps  = params["target_valuation_bn"] / self.total_rev
        self.ev_sales = self.ev / self.total_rev
        years = 5
        self.cagr_ai    = (params["projection_ai_2030_bn"]    / self.rev["AI (xAI/X)"]) ** (1/years) - 1
        self.cagr_total = (params["projection_total_2030_bn"] / self.total_rev)         ** (1/years) - 1

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Metric": ["Target Valuation", "Enterprise Value", "Cash", "Debt",
                       "Total Revenue 2025", "P/S 2025", "EV/Sales 2025", "AI CAGR Required"],
            "Value":  [f"${self.p['target_valuation_bn']} bn", f"${self.ev} bn",
                       f"${self.p['cash_bn']} bn",             f"${self.p['debt_bn']} bn",
                       f"${self.total_rev:.1f} bn",            f"{self.ps:.1f}x",
                       f"{self.ev_sales:.1f}x",                f"{self.cagr_ai*100:.1f}%"]
        })

    def plot_sotp(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(list(self.rev.values()), labels=list(self.rev.keys()),
               autopct="%1.1f%%", startangle=140,
               colors=["#3498db", "#95a5a6", "#9b59b6"],
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title(f"SpaceX 2025 Revenue Breakdown (Total: ${self.total_rev:.1f} bn)",
                     fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_projections(self) -> plt.Figure:
        years = np.arange(2025, 2031)
        ai_curve  = self.rev["AI (xAI/X)"] * (1 + self.cagr_ai)    ** (years - 2025)
        tot_curve = self.total_rev          * (1 + self.cagr_total) ** (years - 2025)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, tot_curve, marker="o", linewidth=2.5, color="#2c3e50", label="Total Revenue Estimate")
        ax.plot(years, ai_curve,  marker="s", linewidth=2.5, color="#9b59b6", linestyle="--", label="AI Segment")
        ax.annotate(f"${self.p['projection_ai_2030_bn']} bn",
                    xy=(2030, self.p["projection_ai_2030_bn"]),
                    xytext=(-40, 15), textcoords="offset points",
                    fontweight="bold", color="#9b59b6")
        ax.set_title("SpaceX Growth Projections 2025–2030 (Goldman Sachs)", fontweight="bold")
        ax.set_ylabel("Revenue ($ bn)")
        ax.set_xlabel("Year")
        ax.set_xticks(years)
        ax.legend()
        ax.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig


class RegressionModel:
    """Linear regression: P/S ratio + IPO pop → Max Drawdown prediction for SpaceX."""

    def __init__(self, df_drawdown, df_ps, df_ipo_pop, spacex_ps):
        ps_initial  = df_ps.groupby("Ticker")["PS_Ratio"].first().reset_index()
        self.df     = (df_drawdown
                       .merge(ps_initial, on="Ticker")
                       .merge(df_ipo_pop[["Ticker", "IPO_Pop_%"]], on="Ticker")
                       .dropna())
        self.spacex_ps = spacex_ps
        self.model     = None

    def fit(self) -> dict:
        X = self.df[["PS_Ratio", "IPO_Pop_%"]]
        y = self.df["Max_Drawdown_%"]
        self.model = LinearRegression().fit(X, y)
        pred = float(self.model.predict([[self.spacex_ps, 0.0]])[0])
        return {
            "alpha":        round(self.model.intercept_, 2),
            "beta_ps":      round(self.model.coef_[0], 2),
            "beta_pop":     round(self.model.coef_[1], 2),
            "spcx_drawdown_pct": round(pred, 1),
        }

    def plot(self, result: dict, colors: dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.df["PS_Ratio"], self.df["Max_Drawdown_%"],
                   color="#3498db", s=120, edgecolor="white", label="Historical Proxies")
        for _, row in self.df.iterrows():
            ax.annotate(row["Ticker"], (row["PS_Ratio"], row["Max_Drawdown_%"]),
                        xytext=(8, 4), textcoords="offset points", fontsize=9)
        ax.scatter([self.spacex_ps], [result["spcx_drawdown_pct"]],
                   color="#e74c3c", s=300, marker="*", edgecolor="black", label="SpaceX Forecast (SPCX)")
        ax.annotate("SPCX (model estimate)", (self.spacex_ps, result["spcx_drawdown_pct"]),
                    xytext=(12, -5), textcoords="offset points",
                    fontsize=11, fontweight="bold", color="#e74c3c")
        ax.set_title("Predictive Model: P/S Ratio vs Drawdown Risk", fontweight="bold")
        ax.set_xlabel("P/S Ratio at Listing")
        ax.set_ylabel("Max Drawdown Estimate (%)")
        ax.axhline(0, color="black", linewidth=1)
        ax.legend()
        ax.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig


class OrnsteinUhlenbeckSimulator:
    """Simulates P/S multiple mean-reversion using an Ornstein-Uhlenbeck process."""

    def __init__(self, p0: float, params: dict):
        self.p0     = p0
        self.mu     = params["mu"]
        self.theta  = params["theta"]
        self.sigma  = params["sigma"]
        self.steps  = params["years"] * params["days_per_year"]
        self.n_sim  = params["n_sim"]
        self.dt     = 1 / 252

    def run(self) -> np.ndarray:
        paths = np.zeros((self.n_sim, self.steps))
        paths[:, 0] = self.p0
        np.random.seed(42)
        for t in range(1, self.steps):
            dW             = np.random.normal(0, np.sqrt(self.dt), self.n_sim)
            paths[:, t]    = paths[:, t-1] + self.theta * (self.mu - paths[:, t-1]) * self.dt + self.sigma * dW
            paths[:, t]    = np.maximum(paths[:, t], 1.0)
        return paths

    def plot(self, paths: np.ndarray) -> Tuple[plt.Figure, dict]:
        mean = np.mean(paths, axis=0)
        p5   = np.percentile(paths, 5,  axis=0)
        p95  = np.percentile(paths, 95, axis=0)
        days = np.arange(self.steps)

        fig, ax = plt.subplots(figsize=(12, 7))
        for i in range(min(100, self.n_sim)):
            ax.plot(days, paths[i], color="gray", alpha=0.1, linewidth=0.8)
        ax.plot(days, mean, color="#e74c3c", linewidth=3,   label="Expected Mean Path")
        ax.plot(days, p95,  color="#3498db", linestyle="--", linewidth=2, label="Optimistic (95th pct)")
        ax.plot(days, p5,   color="#e67e22", linestyle="--", linewidth=2, label="Pessimistic (5th pct)")
        ax.axhline(self.mu,  color="black", linestyle="-.",  linewidth=1.5, label=f"Fundamental target ({self.mu}x)")
        ax.axhline(self.p0,  color="black", linestyle=":",   linewidth=1.5, alpha=0.5)
        ax.set_title("Hype Decay: Ornstein-Uhlenbeck Stochastic Simulation (24 months)",
                     fontweight="bold")
        ax.set_xlabel("Trading Days Post-IPO")
        ax.set_ylabel("P/S Ratio")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()

        stats = {
            "p0":       round(self.p0, 1),
            "mean_end": round(float(mean[-1]), 1),
            "contraction_pct": round(((float(mean[-1]) - self.p0) / self.p0) * 100, 1),
        }
        return fig, stats


class NormalizedPriceVisualizer:
    """Plots all tickers rebased to 100 on their IPO date."""

    def __init__(self, prices: pd.DataFrame, colors: dict):
        self.prices = prices
        self.colors = colors

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 8))
        for ticker in self.prices.columns:
            px = self.prices[ticker].dropna()
            if px.empty:
                continue
            norm = (px / px.iloc[0]) * 100
            ax.plot(np.arange(len(norm)), norm.values, label=ticker,
                    color=self.colors[ticker], linewidth=1.5)
        ax.axhline(100, color="black", linestyle="--", linewidth=2, alpha=0.8)
        ax.set_title("Post-IPO Price Trajectory (Base 100 = Day 0)", fontweight="bold")
        ax.set_xlabel("Trading Days Since Listing")
        ax.set_ylabel("Normalised Price")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig


class ScatterVisualizer:
    """Scatter plot: initial hype (IPO pop) vs long-run quality (Sharpe ratio)."""

    def __init__(self, df: pd.DataFrame, colors: dict):
        self.df     = df
        self.colors = colors

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(data=self.df, x="IPO_Pop_%", y="Sharpe_Ratio",
                        hue="Ticker", palette=self.colors,
                        s=250, edgecolor="white", linewidth=1.5, ax=ax, legend=False)
        for _, row in self.df.iterrows():
            ax.annotate(row["Ticker"], (row["IPO_Pop_%"], row["Sharpe_Ratio"]),
                        xytext=(10, 5), textcoords="offset points",
                        fontsize=10, fontweight="bold", color=self.colors[row["Ticker"]])
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.axvline(0, color="black", linewidth=1, alpha=0.5)
        ax.axhline(1, color="gray",  linestyle="--", linewidth=1.2, alpha=0.6, label="Sharpe = 1")
        ax.set_title("Directional Matrix: Initial Hype vs 3-Year Risk-Adjusted Return", fontweight="bold")
        ax.set_xlabel("IPO Pop (%)")
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = build_palette(TICKERS)

    fetcher = DataFetcher(TICKERS, IPO_DATES)

    print("Fetching historical prices...")
    df_prices = fetcher.prices()

    print("Fetching quarterly revenues...")
    df_revenues = fetcher.revenues()

    print("Fetching shares outstanding...")
    shares = fetcher.shares_outstanding()

    print("Computing IPO pop...")
    ipo_analyzer = IpoPopAnalyzer(TICKERS, IPO_DATES, IPO_PRICES, fetcher, colors)
    df_ipo = ipo_analyzer.compute()
    save(ipo_analyzer.plot(df_ipo), "ipo_pop.png")
    df_ipo.to_csv(os.path.join(OUTPUT_DIR, "ipo_pop.csv"), index=False)

    print("Computing P/S ratios...")
    ps_analyzer = PsRatioAnalyzer(df_prices, df_revenues, shares, colors)
    df_ps = ps_analyzer.compute()
    save(ps_analyzer.plot(df_ps), "ps_compression.png")
    df_ps.to_csv(os.path.join(OUTPUT_DIR, "ps_ratio.csv"), index=False)

    print("Computing Sharpe ratios...")
    sharpe_analyzer = SharpeAnalyzer(df_prices, RISK_FREE_RATE, colors)
    df_sharpe = sharpe_analyzer.compute()
    save(sharpe_analyzer.plot(df_sharpe), "sharpe_ratio.png")
    df_sharpe.to_csv(os.path.join(OUTPUT_DIR, "sharpe_ratio.csv"), index=False)

    print("Computing max drawdown...")
    dd_analyzer = DrawdownAnalyzer(df_prices, IPO_DATES, DRAWDOWN_MONTHS, colors)
    df_drawdown = dd_analyzer.compute(SPACEX["target_valuation_bn"])
    save(dd_analyzer.plot(df_drawdown), "max_drawdown.png")
    df_drawdown.to_csv(os.path.join(OUTPUT_DIR, "max_drawdown.csv"), index=False)

    print("Building SpaceX financial model...")
    sx_model = SpaceXModel(SPACEX)
    save(sx_model.plot_sotp(),        "spacex_sotp.png")
    save(sx_model.plot_projections(), "spacex_projections.png")
    sx_model.summary().to_csv(os.path.join(OUTPUT_DIR, "spacex_fundamentals.csv"), index=False)

    print("Running regression model...")
    reg = RegressionModel(df_drawdown, df_ps, df_ipo, sx_model.ps)
    result = reg.fit()
    print(f"  Equation: Drawdown = {result['alpha']} + ({result['beta_ps']} * P/S) + ({result['beta_pop']} * IPO_Pop)")
    print(f"  SpaceX forecast → Max Drawdown: {result['spcx_drawdown_pct']:.1f}%")
    save(reg.plot(result, colors), "regression_model.png")

    print("Running Ornstein-Uhlenbeck simulation...")
    ou = OrnsteinUhlenbeckSimulator(sx_model.ps, OU_PARAMS)
    paths = ou.run()
    ou_fig, ou_stats = ou.plot(paths)
    save(ou_fig, "ou_simulation.png")
    print(f"  P/S start: {ou_stats['p0']}x  |  P/S mean end: {ou_stats['mean_end']}x  |  Contraction: {ou_stats['contraction_pct']}%")

    print("Generating summary charts...")
    df_summary = (df_ipo[["Ticker", "IPO_Pop_%"]]
                  .merge(df_sharpe[["Ticker", "Sharpe_Ratio", "Annual_Vol_%"]], on="Ticker")
                  .merge(df_drawdown[["Ticker", "Max_Drawdown_%", "SpaceX_Loss_bn"]], on="Ticker"))
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    save(NormalizedPriceVisualizer(df_prices, colors).plot(), "normalised_prices.png")
    save(ScatterVisualizer(df_summary, colors).plot(),        "scatter_hype_vs_sharpe.png")

    print(f"\nAll outputs saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()