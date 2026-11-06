import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------
st.set_page_config(page_title="MVP vs HRP Portfolio Optimizer", layout="wide")
st.title("NIFTY 50 Portfolio Optimization: MVP vs HRP")
st.caption(
    "Close-price-only analysis | Train on historical data | "
    "Evaluate fixed weights on the final six months"
)

DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "TCS.NS", "SBIN.NS", "LT.NS", "ITC.NS", "AXISBANK.NS"
]

with st.sidebar:
    st.header("Portfolio settings")
    ticker_text = st.text_area(
        "10 NIFTY 50 ticker symbols (Yahoo Finance format)",
        value=", ".join(DEFAULT_TICKERS),
        height=110,
        help="Edit this list to match your chosen top-10 universe. Indian NSE tickers usually end in .NS."
    )
    risk_free_rate = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=30.0,
        value=6.0, step=0.25
    ) / 100
    start_date = st.date_input("Data start date", value=pd.Timestamp("2025-01-01"))
    end_date = st.date_input("Data end date", value=pd.Timestamp("2026-10-31"))
    run = st.button("Run portfolio analysis", type="primary", use_container_width=True)

st.info(
    "Default universe is an editable illustrative list of major NIFTY 50 stocks, "
    "not a dynamically verified market-cap ranking. Confirm NIFTY 50 membership "
    "and the top-10 market-cap ranking for your selected reference date."
)

# ------------------------------------------------------------
# Portfolio functions
# ------------------------------------------------------------
def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    """Annualized expected return, volatility and Sharpe ratio."""
    annual_return = float(np.dot(weights, mean_returns) * 252)
    annual_vol = float(np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252))
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else np.nan
    return annual_return, annual_vol, sharpe


def optimize_mvp(returns):
    """Long-only global minimum-variance portfolio, fully invested."""
    n = returns.shape[1]
    cov = returns.cov().values * 252

    def objective(w):
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        objective, np.repeat(1 / n, n), method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12}
    )
    if not result.success:
        st.warning(f"MVP optimizer warning: {result.message}. Using equal weights as fallback.")
        return np.repeat(1 / n, n)
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def get_cluster_variance(cov, cluster_items):
    """Variance of an inverse-variance-weighted cluster portfolio."""
    cluster_cov = cov.loc[cluster_items, cluster_items]
    ivp = 1 / np.diag(cluster_cov.values).clip(min=1e-12)
    ivp = ivp / ivp.sum()
    return float(ivp @ cluster_cov.values @ ivp)


def optimize_hrp(returns):
    """Hierarchical Risk Parity using correlation distance and recursive bisection."""
    cov = returns.cov()
    corr = returns.corr().clip(-1, 1)
    dist = np.sqrt((1 - corr) / 2).clip(0, 1)
    np.fill_diagonal(dist.values, 0)

    condensed = squareform(dist.values, checks=False)
    tree = linkage(condensed, method="single")
    ordered_indices = leaves_list(tree)
    ordered = [returns.columns[i] for i in ordered_indices]

    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_left = get_cluster_variance(cov, left)
            var_right = get_cluster_variance(cov, right)
            alpha = 1 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = weights.reindex(returns.columns).fillna(0)
    return (weights / weights.sum()).values


def performance_metrics(daily_returns, weights, rf, label):
    """Out-of-sample metrics for fixed buy-and-hold weights, daily rebalanced to weights."""
    portfolio_daily = daily_returns @ weights
    n = len(portfolio_daily)
    total_return = (1 + portfolio_daily).prod() - 1
    years = n / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else np.nan
    volatility = portfolio_daily.std(ddof=1) * np.sqrt(252)
    sharpe = ((portfolio_daily.mean() * 252) - rf) / volatility if volatility > 0 else np.nan
    max_drawdown = ((1 + portfolio_daily).cumprod() /
                    (1 + portfolio_daily).cumprod().cummax() - 1).min()
    return {
        "Strategy": label,
        "Total Return": total_return,
        "Annualized Return (CAGR)": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown
    }, portfolio_daily


def pct(x):
    return f"{x:.2%}"


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
if run:
    tickers = list(dict.fromkeys(
        t.strip().upper() for t in ticker_text.replace("\n", ",").split(",") if t.strip()
    ))
    if len(tickers) < 2:
        st.error("Please enter at least two valid ticker symbols.")
        st.stop()
    if start_date >= end_date:
        st.error("The start date must be earlier than the end date.")
        st.stop()

    with st.spinner("Downloading daily closing prices from Yahoo Finance..."):
        try:
            # yfinance end date is exclusive, so add one day to include selected end_date.
            raw = yf.download(
                tickers, start=str(start_date),
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False, progress=False, group_by="column", threads=True
            )
            if raw.empty:
                st.error("No data was downloaded. Check ticker symbols and internet access.")
                st.stop()

            # Explicitly select the unadjusted 'Close' field only.
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" not in raw.columns.get_level_values(0):
                    st.error("Yahoo Finance did not return a Close price field.")
                    st.stop()
                close = raw["Close"].copy()
            else:
                # Single ticker download
                if "Close" not in raw.columns:
                    st.error("Yahoo Finance did not return a Close price field.")
                    st.stop()
                close = raw[["Close"]].copy()
                close.columns = [tickers[0]]

            close = close.dropna(axis=1, how="all").sort_index()
            # Remove assets with insufficient observations; align dates across assets.
            close = close.dropna(axis=0, how="any")
            if close.shape[1] < 2 or len(close) < 100:
                st.error("Insufficient overlapping closing-price data. Check the tickers/date range.")
                st.stop()

            daily_returns = close.pct_change().dropna()
            last_date = daily_returns.index.max()
            test_start = last_date - pd.DateOffset(months=6) + pd.Timedelta(days=1)

            train_returns = daily_returns.loc[daily_returns.index < test_start]
            test_returns = daily_returns.loc[daily_returns.index >= test_start]

            if len(train_returns) < 60 or len(test_returns) < 20:
                st.error(
                    "Not enough training/testing observations. Use a longer date range "
                    "or verify that the selected stocks have complete data."
                )
                st.stop()

            # Estimate weights only on training data.
            train_mean = train_returns.mean().values
            train_cov = train_returns.cov().values
            mvp_weights = optimize_mvp(train_returns)
            hrp_weights = optimize_hrp(train_returns)

            mvp_train_stats = portfolio_stats(mvp_weights, train_mean, train_cov, risk_free_rate)
            hrp_train_stats = portfolio_stats(hrp_weights, train_mean, train_cov, risk_free_rate)

            # Test period: weights fixed at training-end allocations.
            mvp_result, mvp_daily = performance_metrics(
                test_returns, mvp_weights, risk_free_rate, "MVP"
            )
            hrp_result, hrp_daily = performance_metrics(
                test_returns, hrp_weights, risk_free_rate, "HRP"
            )
            equal_weights = np.repeat(1 / len(close.columns), len(close.columns))
            ew_result, ew_daily = performance_metrics(
                test_returns, equal_weights, risk_free_rate, "Equal Weight"
            )

        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.success("Analysis completed.")

    st.subheader("1. Data split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training period", f"{train_returns.index.min():%d %b %Y} – {train_returns.index.max():%d %b %Y}")
    c2.metric("Testing period", f"{test_returns.index.min():%d %b %Y} – {test_returns.index.max():%d %b %Y}")
    c3.metric("Stocks analyzed", str(close.shape[1]))

    st.caption(
        f"Downloaded close prices from {close.index.min():%d %b %Y} to "
        f"{close.index.max():%d %b %Y}. Actual split is based on the last available "
        "trading date in the downloaded data."
    )

    st.subheader("2. Out-of-sample performance comparison")
    metrics = pd.DataFrame([mvp_result, hrp_result, ew_result]).set_index("Strategy")
    display_metrics = metrics.copy()
    for col in ["Total Return", "Annualized Return (CAGR)", "Annualized Volatility", "Maximum Drawdown"]:
        display_metrics[col] = display_metrics[col].map(pct)
    display_metrics["Sharpe Ratio"] = display_metrics["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True)

    st.subheader("3. Cumulative return plot (testing period)")
    cumulative = pd.DataFrame({
        "MVP": (1 + mvp_daily).cumprod(),
        "HRP": (1 + hrp_daily).cumprod(),
        "Equal Weight": (1 + ew_daily).cumprod()
    })
    fig, ax = plt.subplots(figsize=(11, 5))
    cumulative.plot(ax=ax, linewidth=2)
    ax.set_ylabel("Growth of ₹1")
    ax.set_xlabel("Date")
    ax.set_title("Out-of-sample cumulative growth")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Risk plot: rolling 21-trading-day annualized volatility")
    rolling_risk = pd.DataFrame({
        "MVP": mvp_daily.rolling(21).std() * np.sqrt(252),
        "HRP": hrp_daily.rolling(21).std() * np.sqrt(252),
        "Equal Weight": ew_daily.rolling(21).std() * np.sqrt(252)
    })
    fig, ax = plt.subplots(figsize=(11, 5))
    rolling_risk.plot(ax=ax, linewidth=1.8)
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Date")
    ax.set_title("Rolling realized risk during the test period")
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("5. Risk-return comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, row in metrics.iterrows():
        ax.scatter(row["Annualized Volatility"], row["Annualized Return (CAGR)"], s=110)
        ax.annotate(name, (row["Annualized Volatility"], row["Annualized Return (CAGR)"]),
                    xytext=(7, 6), textcoords="offset points")
    ax.set_xlabel("Annualized volatility (risk)")
    ax.set_ylabel("Annualized return (CAGR)")
    ax.set_title("Out-of-sample risk vs return")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("6. Portfolio allocations")
    allocation = pd.DataFrame({
        "Stock": close.columns,
        "MVP Weight": mvp_weights,
        "HRP Weight": hrp_weights
    }).set_index("Stock")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**MVP allocation**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(mvp_weights, labels=close.columns, autopct="%1.1f%%", startangle=90)
        ax.set_title("Minimum Variance Portfolio")
        st.pyplot(fig)
        plt.close(fig)
    with p2:
        st.markdown("**HRP allocation**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(hrp_weights, labels=close.columns, autopct="%1.1f%%", startangle=90)
        ax.set_title("Hierarchical Risk Parity")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("7. Allocation table")
    allocation_display = allocation.copy()
    allocation_display["MVP Weight"] = allocation_display["MVP Weight"].map(pct)
    allocation_display["HRP Weight"] = allocation_display["HRP Weight"].map(pct)
    st.dataframe(allocation_display, use_container_width=True)

    st.subheader("8. Training-period portfolio estimates")
    train_summary = pd.DataFrame(
        [mvp_train_stats, hrp_train_stats],
        index=["MVP", "HRP"],
        columns=["Annualized Return", "Annualized Volatility", "Sharpe Ratio"]
    )
    train_display = train_summary.copy()
    train_display["Annualized Return"] = train_display["Annualized Return"].map(pct)
    train_display["Annualized Volatility"] = train_display["Annualized Volatility"].map(pct)
    train_display["Sharpe Ratio"] = train_display["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    st.dataframe(train_display, use_container_width=True)

    st.subheader("9. Download results")
    export = allocation.copy()
    export["MVP Weight"] = mvp_weights
    export["HRP Weight"] = hrp_weights
    csv = export.to_csv().encode("utf-8")
    st.download_button(
        "Download allocation table (CSV)", data=csv,
        file_name="mvp_hrp_allocations.csv", mime="text/csv"
    )

    with st.expander("Methodology and interpretation"):
        st.markdown("""
        - **Close prices:** The app explicitly selects Yahoo Finance's `Close` field;
          it does not use Open, High, Low, or Volume.
        - **MVP:** Minimizes annualized covariance-based portfolio variance subject
          to long-only weights between 0 and 100% and weights summing to 100%.
        - **HRP:** Converts correlations into distances, clusters stocks, orders the
          hierarchy, and recursively allocates capital using cluster variances.
        - **Testing:** Training-estimated weights are applied unchanged to the test
          period. Daily portfolio returns are calculated as the weighted sum of
          daily stock returns (constant-weight/daily-rebalanced interpretation).
        - **Sharpe ratio:** `(annualized portfolio return - annual risk-free rate) /
          annualized volatility`. The annual risk-free rate is editable in the sidebar.
        - **Limitations:** Results depend on the selected universe, data availability,
          transaction costs, taxes, liquidity, and the chosen risk-free rate. This is
          an educational research tool, not investment advice.
        """)
else:
    st.write("Choose your stock universe and settings in the sidebar, then click **Run portfolio analysis**.")
    st.markdown("**Included outputs:** performance table, Sharpe ratios, cumulative return chart, rolling risk chart, risk-return plot, two allocation pie charts, allocation table, and CSV export.")
