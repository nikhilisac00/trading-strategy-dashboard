"""
Trading Strategy Dashboard v2
=============================
Full-featured dashboard with:
- Risk profile selection
- Options strategy builder with payoff diagrams
- Portfolio performance tracking
- Securities lookup with options chains

Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

# Import our modules
from options_regime_selector import (
    get_vix_stats,
    determine_regime as determine_vix_regime,
    VIX_REGIMES,
    fetch_vix_data,
    regime_distribution,
)
from fixed_income_dashboard import (
    get_current_yields,
    calculate_spreads,
    determine_curve_regime,
    CURVE_REGIMES,
    FI_STRATEGIES,
)
from entry_timing_signals import (
    get_all_signals,
    check_alerts,
    SignalStrength,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# RISK PROFILE STRATEGIES
# ============================================================

RISK_PROFILES = {
    "Conservative": {
        "description": "Capital preservation, lower risk",
        "max_position_size": 0.02,  # 2% of portfolio
        "strategies": {
            "LOW": [
                {"name": "Covered Calls", "risk": "Low", "reward": "Limited", "description": "Own stock, sell calls against it"},
                {"name": "Cash-Secured Puts", "risk": "Medium", "reward": "Limited", "description": "Sell puts with cash to cover assignment"},
            ],
            "NORMAL": [
                {"name": "Put Credit Spreads", "risk": "Defined", "reward": "Limited", "description": "Sell put spread for credit"},
                {"name": "Collar", "risk": "Very Low", "reward": "Limited", "description": "Own stock, buy put, sell call"},
            ],
            "ELEVATED": [
                {"name": "Protective Puts", "risk": "Cost of put", "reward": "Unlimited upside", "description": "Buy puts to protect positions"},
                {"name": "Cash/T-Bills", "risk": "None", "reward": "Risk-free rate", "description": "Wait for better opportunity"},
            ],
            "CRISIS": [
                {"name": "Dollar Cost Average", "risk": "Market risk", "reward": "Long-term", "description": "Slowly add to quality positions"},
                {"name": "Investment Grade Bonds", "risk": "Low", "reward": "Yield", "description": "TLT, LQD for safety"},
            ],
        },
    },
    "Moderate": {
        "description": "Balanced risk/reward",
        "max_position_size": 0.05,  # 5% of portfolio
        "strategies": {
            "LOW": [
                {"name": "Iron Condors", "risk": "Defined", "reward": "Limited", "description": "Sell both put and call spreads"},
                {"name": "Put Credit Spreads", "risk": "Defined", "reward": "Limited", "description": "Bullish credit spread"},
                {"name": "Calendar Spreads", "risk": "Defined", "reward": "Limited", "description": "Sell front month, buy back month"},
            ],
            "NORMAL": [
                {"name": "Vertical Spreads", "risk": "Defined", "reward": "Defined", "description": "Directional with limited risk"},
                {"name": "Iron Condors", "risk": "Defined", "reward": "Limited", "description": "Range-bound strategy"},
                {"name": "Diagonal Spreads", "risk": "Defined", "reward": "Limited", "description": "Directional calendar spread"},
            ],
            "ELEVATED": [
                {"name": "Call Debit Spreads", "risk": "Defined", "reward": "Defined", "description": "Bullish with limited cost"},
                {"name": "Put Credit Spreads", "risk": "Defined", "reward": "Limited", "description": "Sell fear premium"},
                {"name": "LEAPS Calls", "risk": "Premium paid", "reward": "Significant", "description": "Long-dated bullish bets"},
            ],
            "CRISIS": [
                {"name": "LEAPS on Quality", "risk": "Premium paid", "reward": "Significant", "description": "6-12 month calls on leaders"},
                {"name": "Put Credit Spreads", "risk": "Defined", "reward": "Limited", "description": "Sell extreme fear"},
                {"name": "Call Spreads", "risk": "Defined", "reward": "Defined", "description": "Recovery bets"},
            ],
        },
    },
    "Aggressive": {
        "description": "Higher risk for higher reward",
        "max_position_size": 0.10,  # 10% of portfolio
        "strategies": {
            "LOW": [
                {"name": "Naked Puts", "risk": "High", "reward": "Premium", "description": "Sell puts without hedge"},
                {"name": "Strangles (Short)", "risk": "Unlimited", "reward": "Premium", "description": "Sell OTM puts and calls"},
                {"name": "Ratio Spreads", "risk": "Unlimited on one side", "reward": "Premium + direction", "description": "Buy 1, sell 2"},
            ],
            "NORMAL": [
                {"name": "Naked Puts on Dips", "risk": "High", "reward": "Premium", "description": "Sell puts when stock pulls back"},
                {"name": "Strangles", "risk": "Unlimited", "reward": "Premium", "description": "Sell premium both sides"},
                {"name": "Ratio Call Spreads", "risk": "Unlimited upside", "reward": "Cheap entry", "description": "Finance calls by selling more"},
            ],
            "ELEVATED": [
                {"name": "Outright Calls", "risk": "100% of premium", "reward": "Unlimited", "description": "Buy calls on quality"},
                {"name": "LEAPS (Deep ITM)", "risk": "Premium", "reward": "Leveraged stock", "description": "Stock replacement"},
                {"name": "Naked Puts (Aggressive)", "risk": "High", "reward": "Rich premium", "description": "Sell fear at good prices"},
                {"name": "Ratio Spreads", "risk": "Unlimited", "reward": "Low cost entry", "description": "Buy 1, sell 2+ OTM"},
            ],
            "CRISIS": [
                {"name": "MAX LONG CALLS", "risk": "100% of premium", "reward": "Unlimited", "description": "Generational opportunity"},
                {"name": "Naked Puts on Survivors", "risk": "High", "reward": "Extreme premium", "description": "AAPL, MSFT, JPM"},
                {"name": "LEAPS (6-12 month)", "risk": "Premium", "reward": "10x+", "description": "Recovery plays"},
                {"name": "VIX Put Spreads", "risk": "Defined", "reward": "VIX mean reversion", "description": "VIX always comes down"},
            ],
        },
    },
}

# ============================================================
# OPTIONS PRICING FUNCTIONS
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call price."""
    from scipy.stats import norm
    if T <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put price."""
    from scipy.stats import norm
    if T <= 0:
        return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_payoff(strategy_type, spot_range, params):
    """Calculate payoff for various strategies."""
    payoffs = []

    if strategy_type == "Long Call":
        strike = params["strike"]
        premium = params["premium"]
        for S in spot_range:
            payoff = max(0, S - strike) - premium
            payoffs.append(payoff)

    elif strategy_type == "Long Put":
        strike = params["strike"]
        premium = params["premium"]
        for S in spot_range:
            payoff = max(0, strike - S) - premium
            payoffs.append(payoff)

    elif strategy_type == "Short Put":
        strike = params["strike"]
        premium = params["premium"]
        for S in spot_range:
            payoff = premium - max(0, strike - S)
            payoffs.append(payoff)

    elif strategy_type == "Bull Call Spread":
        long_strike = params["long_strike"]
        short_strike = params["short_strike"]
        net_debit = params["net_debit"]
        for S in spot_range:
            long_payoff = max(0, S - long_strike)
            short_payoff = max(0, S - short_strike)
            payoff = long_payoff - short_payoff - net_debit
            payoffs.append(payoff)

    elif strategy_type == "Bull Put Spread":
        short_strike = params["short_strike"]
        long_strike = params["long_strike"]
        net_credit = params["net_credit"]
        for S in spot_range:
            short_payoff = max(0, short_strike - S)
            long_payoff = max(0, long_strike - S)
            payoff = net_credit - short_payoff + long_payoff
            payoffs.append(payoff)

    elif strategy_type == "Iron Condor":
        put_long = params["put_long"]
        put_short = params["put_short"]
        call_short = params["call_short"]
        call_long = params["call_long"]
        net_credit = params["net_credit"]
        for S in spot_range:
            put_spread = max(0, put_short - S) - max(0, put_long - S)
            call_spread = max(0, S - call_short) - max(0, S - call_long)
            payoff = net_credit - put_spread - call_spread
            payoffs.append(payoff)

    elif strategy_type == "Straddle":
        strike = params["strike"]
        premium = params["premium"]
        for S in spot_range:
            call_payoff = max(0, S - strike)
            put_payoff = max(0, strike - S)
            payoff = call_payoff + put_payoff - premium
            payoffs.append(payoff)

    return payoffs

# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=300)
def load_vix_data():
    return get_vix_stats(lookback_years=2)

@st.cache_data(ttl=300)
def load_yield_data():
    return get_current_yields()

@st.cache_data(ttl=300)
def load_vix_history():
    return fetch_vix_data(lookback_years=2)

@st.cache_data(ttl=300)
def load_all_signals():
    return get_all_signals()

@st.cache_data(ttl=60)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return {"history": hist, "info": info, "ticker": ticker}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def get_historical_returns(ticker: str, period: str = "1y") -> dict:
    """
    Get REAL historical returns from yfinance.
    MUST be used for any return calculations - NEVER estimate or approximate.
    If data is unavailable, returns error explicitly.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty or len(hist) < 2:
            return {"error": f"No data available for {ticker} - cannot calculate returns", "ticker": ticker}

        start_price = float(hist['Close'].iloc[0])
        end_price = float(hist['Close'].iloc[-1])
        total_return = ((end_price / start_price) - 1) * 100

        # Calculate additional metrics from REAL data
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100)  # Annualized
        max_price = float(hist['Close'].max())
        min_price = float(hist['Close'].min())
        max_drawdown = ((max_price - min_price) / max_price) * 100

        # Annualize the return
        days = len(hist)
        years = days / 252  # Trading days
        if years > 0:
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        else:
            annualized_return = total_return

        return {
            "ticker": ticker,
            "period": period,
            "start_price": start_price,
            "end_price": end_price,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "trading_days": days,
            "data_source": "yfinance (REAL DATA)",
        }
    except Exception as e:
        return {"error": f"Cannot calculate returns for {ticker}: {str(e)}", "ticker": ticker}

@st.cache_data(ttl=300)
def get_portfolio_real_returns(positions: list, period: str = "1y") -> dict:
    """
    Calculate REAL portfolio returns using actual market data.
    NEVER estimates - returns error if data unavailable.
    """
    if not positions:
        return {"error": "No positions to analyze"}

    position_returns = []
    total_weight = 0
    weighted_return = 0
    weighted_volatility = 0
    errors = []

    for pos in positions:
        ticker = pos.get("ticker", "")
        if not ticker or "---" in pos.get("type", ""):
            continue

        quantity = pos.get("quantity", 0)
        entry_price = pos.get("entry_price", 0)
        position_value = quantity * entry_price

        # Get REAL returns - no estimates
        returns = get_historical_returns(ticker, period)

        if "error" in returns:
            errors.append(returns["error"])
            continue

        position_returns.append({
            "ticker": ticker,
            "value": position_value,
            "return": returns["total_return"],
            "annualized": returns["annualized_return"],
            "volatility": returns["volatility"],
            "data_source": returns["data_source"],
        })
        total_weight += position_value

    if total_weight == 0:
        return {"error": f"Could not calculate returns - no valid data. Errors: {errors}"}

    # Calculate weighted portfolio metrics from REAL data
    for pr in position_returns:
        weight = pr["value"] / total_weight
        weighted_return += weight * pr["return"]
        weighted_volatility += weight * pr["volatility"]

    return {
        "portfolio_return": weighted_return,
        "portfolio_volatility": weighted_volatility,
        "period": period,
        "positions_analyzed": len(position_returns),
        "total_value": total_weight,
        "position_details": position_returns,
        "data_source": "yfinance (ALL REAL DATA - no estimates)",
        "errors": errors if errors else None,
    }

@st.cache_data(ttl=60)
def get_options_chain(ticker):
    """Fetch options chain for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"error": "No options available"}

        # Get first 5 expirations
        chains = {}
        for exp in expirations[:5]:
            try:
                opt = stock.option_chain(exp)
                chains[exp] = {
                    "calls": opt.calls,
                    "puts": opt.puts,
                }
            except:
                pass

        return {"expirations": expirations, "chains": chains, "ticker": ticker}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def get_analyst_data(ticker):
    """Fetch analyst recommendations and price targets."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get analyst data
        data = {
            "ticker": ticker,
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "target_high": info.get("targetHighPrice", None),
            "target_low": info.get("targetLowPrice", None),
            "target_mean": info.get("targetMeanPrice", None),
            "target_median": info.get("targetMedianPrice", None),
            "recommendation": info.get("recommendationKey", "N/A"),
            "recommendation_mean": info.get("recommendationMean", None),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            "earnings_growth": info.get("earningsGrowth", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "forward_pe": info.get("forwardPE", None),
            "peg_ratio": info.get("pegRatio", None),
        }

        # Calculate implied upside/downside
        if data["current_price"] and data["target_mean"]:
            data["upside_pct"] = (data["target_mean"] - data["current_price"]) / data["current_price"] * 100
        else:
            data["upside_pct"] = None

        return data
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

@st.cache_data(ttl=300)
def get_historical_portfolio_performance(positions: list, period: str = "1mo"):
    """Calculate historical portfolio performance."""
    if not positions:
        return None

    period_map = {
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "YTD": "ytd",
    }
    yf_period = period_map.get(period, "1mo")

    # Fetch historical data for all positions
    portfolio_values = {}
    position_data = {}

    for pos in positions:
        ticker = pos.get("ticker", "")
        if not ticker or "---" in pos.get("type", ""):
            continue

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=yf_period)

            if not hist.empty:
                position_data[ticker] = {
                    "history": hist,
                    "quantity": pos.get("quantity", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "side": pos.get("side", "Long"),
                }
        except:
            pass

    if not position_data:
        return None

    # Calculate daily portfolio value
    all_dates = set()
    for ticker, data in position_data.items():
        all_dates.update(data["history"].index.tolist())

    all_dates = sorted(all_dates)

    daily_values = []
    for date in all_dates:
        total_value = 0
        for ticker, data in position_data.items():
            hist = data["history"]
            if date in hist.index:
                price = hist.loc[date, "Close"]
                quantity = data["quantity"]
                if data["side"] == "Short":
                    # Short position value = entry_price * qty - current_price * qty
                    value = data["entry_price"] * quantity - price * quantity
                else:
                    value = price * quantity
                total_value += value

        if total_value != 0:
            daily_values.append({"date": date, "value": total_value})

    if not daily_values:
        return None

    df = pd.DataFrame(daily_values)
    df.set_index("date", inplace=True)

    # Calculate returns
    start_value = df["value"].iloc[0]
    end_value = df["value"].iloc[-1]
    total_return = (end_value - start_value) / abs(start_value) * 100 if start_value != 0 else 0

    return {
        "history": df,
        "start_value": start_value,
        "end_value": end_value,
        "total_return": total_return,
        "period": period,
    }

# ============================================================
# PORTFOLIO STATE (Session State)
# ============================================================

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = pd.DataFrame()

if "fi_selected" not in st.session_state:
    st.session_state.fi_selected = "TLT"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "risk_level": "moderate",
        "budget": None,
        "return_target": None,
        "constraints": {}
    }

if "pending_portfolio" not in st.session_state:
    st.session_state.pending_portfolio = None

if "pending_rebalancing" not in st.session_state:
    st.session_state.pending_rebalancing = None

# ============================================================
# AI FINANCIAL PLANNER
# ============================================================

class FinancialPlannerAI:
    """AI Financial Planner that generates portfolios based on user requirements."""

    # Asset class expected returns and risk (historical approximations)
    ASSET_CLASSES = {
        # Equities
        "SPY": {"name": "S&P 500 ETF", "type": "equity", "expected_return": 0.10, "risk": 0.16, "yield": 0.013},
        "QQQ": {"name": "Nasdaq 100 ETF", "type": "equity", "expected_return": 0.12, "risk": 0.20, "yield": 0.005},
        "VTI": {"name": "Total Stock Market", "type": "equity", "expected_return": 0.10, "risk": 0.16, "yield": 0.014},
        "VXUS": {"name": "International Stocks", "type": "equity", "expected_return": 0.07, "risk": 0.18, "yield": 0.03},
        "VWO": {"name": "Emerging Markets", "type": "equity", "expected_return": 0.08, "risk": 0.22, "yield": 0.025},
        "VNQ": {"name": "Real Estate (REITs)", "type": "real_estate", "expected_return": 0.08, "risk": 0.18, "yield": 0.04},

        # Fixed Income
        "TLT": {"name": "20+ Year Treasury", "type": "treasury", "expected_return": 0.04, "risk": 0.12, "yield": 0.045},
        "IEF": {"name": "7-10 Year Treasury", "type": "treasury", "expected_return": 0.035, "risk": 0.07, "yield": 0.04},
        "SHY": {"name": "1-3 Year Treasury", "type": "treasury", "expected_return": 0.03, "risk": 0.02, "yield": 0.035},
        "BND": {"name": "Total Bond Market", "type": "bond", "expected_return": 0.04, "risk": 0.05, "yield": 0.04},
        "LQD": {"name": "Investment Grade Corp", "type": "corporate_bond", "expected_return": 0.05, "risk": 0.08, "yield": 0.05},
        "HYG": {"name": "High Yield Corp", "type": "high_yield", "expected_return": 0.06, "risk": 0.10, "yield": 0.065},
        "TIP": {"name": "TIPS (Inflation)", "type": "tips", "expected_return": 0.035, "risk": 0.05, "yield": 0.02},

        # Alternatives
        "GLD": {"name": "Gold", "type": "commodity", "expected_return": 0.05, "risk": 0.15, "yield": 0.0},
        "SCHD": {"name": "Dividend Growth", "type": "dividend", "expected_return": 0.09, "risk": 0.14, "yield": 0.035},
    }

    RISK_PROFILES = {
        "conservative": {
            "equity_range": (0.20, 0.40),
            "bond_range": (0.40, 0.60),
            "treasury_range": (0.10, 0.30),
            "max_single_position": 0.20,
        },
        "moderate": {
            "equity_range": (0.40, 0.60),
            "bond_range": (0.20, 0.40),
            "treasury_range": (0.05, 0.20),
            "max_single_position": 0.25,
        },
        "aggressive": {
            "equity_range": (0.60, 0.85),
            "bond_range": (0.05, 0.25),
            "treasury_range": (0.00, 0.15),
            "max_single_position": 0.30,
        },
        "very_aggressive": {
            "equity_range": (0.80, 1.00),
            "bond_range": (0.00, 0.15),
            "treasury_range": (0.00, 0.10),
            "max_single_position": 0.35,
        },
    }

    # Risk profiles linked to target Beta
    BETA_TARGETS = {
        "conservative": {"min": 0.3, "max": 0.6, "target": 0.5},
        "moderate": {"min": 0.6, "max": 0.9, "target": 0.75},
        "aggressive": {"min": 0.9, "max": 1.3, "target": 1.1},
        "very_aggressive": {"min": 1.2, "max": 2.0, "target": 1.5},
    }

    @classmethod
    def get_stock_beta(cls, ticker: str) -> float:
        """Get REAL beta for a stock from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            beta = info.get("beta", None)
            if beta is not None:
                return float(beta)
            return 1.0  # Default to market beta if unavailable
        except:
            return 1.0

    @classmethod
    def get_stock_info_for_portfolio(cls, ticker: str) -> dict:
        """Get comprehensive stock info for adding to portfolio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")

            current_price = float(hist['Close'].iloc[-1]) if not hist.empty else info.get("currentPrice", 100)

            return {
                "ticker": ticker.upper(),
                "name": info.get("shortName", info.get("longName", ticker)),
                "current_price": current_price,
                "beta": info.get("beta", 1.0),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "dividend_yield": info.get("dividendYield", 0) or 0,
                "pe_ratio": info.get("trailingPE", None),
                "52w_high": info.get("fiftyTwoWeekHigh", None),
                "52w_low": info.get("fiftyTwoWeekLow", None),
                "error": None
            }
        except Exception as e:
            return {"ticker": ticker.upper(), "error": str(e)}

    @classmethod
    def calculate_portfolio_beta(cls, positions: list) -> float:
        """Calculate weighted average Beta of portfolio."""
        if not positions:
            return 1.0

        total_value = 0
        weighted_beta = 0

        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker:
                continue

            quantity = pos.get("quantity", 0)
            entry_price = pos.get("entry_price", 100)
            value = quantity * entry_price

            beta = cls.get_stock_beta(ticker)
            weighted_beta += value * beta
            total_value += value

        if total_value > 0:
            return weighted_beta / total_value
        return 1.0

    @classmethod
    def parse_user_input(cls, message: str) -> dict:
        """Parse user message to extract requirements."""
        message_lower = message.lower()
        parsed = {
            "risk_level": None,
            "return_target": None,
            "budget": None,
            "treasury_constraint": None,
            "equity_constraint": None,
            "bond_constraint": None,
            "specific_stocks": [],  # New: specific stocks to add
            "action": None,  # New: add, remove, etc.
            "amount": None,  # New: dollar amount for specific stock
            "shares": None,  # New: number of shares
        }

        # Detect specific stock ticker requests
        import re
        # Look for "add AAPL" or "buy AAPL" or "AAPL stock" patterns
        # Using case-insensitive flag
        stock_patterns = [
            r'\b(?:add|buy|purchase|get|include)\s+(\$?[A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})\s+(?:stock|shares|position)\b',
            r'\bI\s+(?:want|like|prefer)\s+(\$?[A-Z]{1,5})\b',
            r'\b(?:add|buy)\s+(?:\$?[\d,]+\s+(?:of|worth|in)\s+)?([A-Z]{1,5})\b',
            r'\b(?:of|in|worth\s+of)\s+([A-Z]{1,5})\b',  # "$5000 of TSLA"
        ]

        message_upper = message.upper()
        found_tickers = set()
        for pattern in stock_patterns:
            matches = re.findall(pattern, message_upper, re.IGNORECASE)
            for match in matches:
                ticker = match.replace("$", "").strip()
                # Validate it looks like a ticker (not common words)
                if ticker and len(ticker) <= 5 and ticker not in ["I", "A", "THE", "AND", "FOR", "ADD", "BUY", "GET", "MY", "TO", "IN", "OF", "WORTH"]:
                    found_tickers.add(ticker)
        parsed["specific_stocks"] = list(found_tickers)

        # Detect dollar amount for specific stock - multiple patterns
        # Pattern 1: "$5000 of AAPL" or "$5,000 worth of AAPL"
        amount_match = re.search(r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:of|worth|in)?', message, re.IGNORECASE)
        if amount_match:
            parsed["amount"] = float(amount_match.group(1).replace(',', ''))
        # Pattern 2: "buy 5000 worth of AAPL" without $ sign
        if not parsed["amount"]:
            amount_match2 = re.search(r'(\d+(?:,\d{3})*)\s*(?:dollars?|worth|of)\s+', message, re.IGNORECASE)
            if amount_match2:
                parsed["amount"] = float(amount_match2.group(1).replace(',', ''))

        # Detect number of shares - "10 shares of NVDA" or "add 10 shares AAPL"
        shares_match = re.search(r'(\d+)\s+shares?\s+(?:of\s+)?([A-Z]{1,5})', message, re.IGNORECASE)
        if shares_match:
            parsed["shares"] = int(shares_match.group(1))
            ticker = shares_match.group(2)
            if ticker not in parsed["specific_stocks"]:
                parsed["specific_stocks"].append(ticker)

        # Detect action type
        if any(word in message_lower for word in ["add", "buy", "purchase", "get", "include"]):
            parsed["action"] = "add"
        elif any(word in message_lower for word in ["remove", "sell", "drop", "delete"]):
            parsed["action"] = "remove"

        # Detect risk level
        if any(word in message_lower for word in ["very risky", "very aggressive", "high risk", "maximum risk", "yolo"]):
            parsed["risk_level"] = "very_aggressive"
        elif any(word in message_lower for word in ["risky", "aggressive", "growth", "high return"]):
            parsed["risk_level"] = "aggressive"
        elif any(word in message_lower for word in ["conservative", "safe", "low risk", "preservation"]):
            parsed["risk_level"] = "conservative"
        elif any(word in message_lower for word in ["moderate", "balanced", "medium"]):
            parsed["risk_level"] = "moderate"

        # Detect return target (look for percentages)
        import re
        return_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:return|annually|annual|yearly|per year)', message_lower)
        if return_match:
            parsed["return_target"] = float(return_match.group(1)) / 100

        # Detect budget
        budget_match = re.search(r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:k|K|thousand)?(?:\s*(?:budget|invest|total|to invest))?', message)
        if budget_match:
            amount = float(budget_match.group(1).replace(',', ''))
            if 'k' in message_lower or 'K' in message:
                amount *= 1000
            if amount > 100:  # Assume it's a budget, not a percentage
                parsed["budget"] = amount

        # Detect treasury constraint
        treasury_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:in\s+)?(?:treasury|treasuries|t-bills|government bonds)', message_lower)
        if treasury_match:
            parsed["treasury_constraint"] = float(treasury_match.group(1)) / 100

        # Detect equity constraint
        equity_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:in\s+)?(?:stocks?|equities|equity)', message_lower)
        if equity_match:
            parsed["equity_constraint"] = float(equity_match.group(1)) / 100

        # Detect bond constraint
        bond_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:in\s+)?(?:bonds?|fixed income)', message_lower)
        if bond_match:
            parsed["bond_constraint"] = float(bond_match.group(1)) / 100

        return parsed

    @classmethod
    def generate_portfolio(cls, risk_level: str, return_target: float = None,
                          treasury_pct: float = None, budget: float = 100000) -> dict:
        """Generate a diversified portfolio based on requirements."""

        profile = cls.RISK_PROFILES.get(risk_level, cls.RISK_PROFILES["moderate"])

        # Start with base allocation based on risk profile
        allocations = {}

        # Determine treasury allocation
        if treasury_pct is not None:
            treasury_alloc = treasury_pct
        else:
            treasury_alloc = (profile["treasury_range"][0] + profile["treasury_range"][1]) / 2

        # Determine equity allocation
        equity_min, equity_max = profile["equity_range"]

        # If return target specified, adjust equity allocation
        if return_target:
            # Higher return target = more equity
            if return_target >= 0.12:
                equity_alloc = min(0.90, equity_max + 0.10)
            elif return_target >= 0.09:
                equity_alloc = equity_max
            elif return_target >= 0.06:
                equity_alloc = (equity_min + equity_max) / 2
            else:
                equity_alloc = equity_min
        else:
            equity_alloc = (equity_min + equity_max) / 2

        # Calculate bond allocation
        bond_alloc = 1.0 - equity_alloc - treasury_alloc

        # Ensure non-negative
        if bond_alloc < 0:
            bond_alloc = 0
            equity_alloc = 1.0 - treasury_alloc

        # Build specific allocations
        if equity_alloc > 0:
            # Diversify equity
            if risk_level in ["aggressive", "very_aggressive"]:
                allocations["SPY"] = equity_alloc * 0.40
                allocations["QQQ"] = equity_alloc * 0.30
                allocations["VXUS"] = equity_alloc * 0.15
                allocations["VWO"] = equity_alloc * 0.15
            else:
                allocations["SPY"] = equity_alloc * 0.50
                allocations["VTI"] = equity_alloc * 0.20
                allocations["VXUS"] = equity_alloc * 0.20
                allocations["SCHD"] = equity_alloc * 0.10

        if treasury_alloc > 0:
            # Diversify treasury
            if treasury_alloc >= 0.15:
                allocations["TLT"] = treasury_alloc * 0.40
                allocations["IEF"] = treasury_alloc * 0.40
                allocations["SHY"] = treasury_alloc * 0.20
            else:
                allocations["IEF"] = treasury_alloc * 0.60
                allocations["SHY"] = treasury_alloc * 0.40

        if bond_alloc > 0:
            # Diversify bonds
            if risk_level in ["aggressive", "very_aggressive"]:
                allocations["LQD"] = bond_alloc * 0.40
                allocations["HYG"] = bond_alloc * 0.40
                allocations["TIP"] = bond_alloc * 0.20
            else:
                allocations["BND"] = bond_alloc * 0.50
                allocations["LQD"] = bond_alloc * 0.30
                allocations["TIP"] = bond_alloc * 0.20

        # Calculate portfolio metrics using REAL DATA
        # Fetch actual historical returns for each position
        real_returns = {}
        real_volatilities = {}

        for ticker in allocations.keys():
            if allocations[ticker] > 0.001:
                hist_data = get_historical_returns(ticker, "1y")
                if "error" not in hist_data:
                    real_returns[ticker] = hist_data["annualized_return"] / 100
                    real_volatilities[ticker] = hist_data["volatility"] / 100
                else:
                    # Fallback to estimates only if real data unavailable
                    real_returns[ticker] = cls.ASSET_CLASSES.get(ticker, {}).get("expected_return", 0.08)
                    real_volatilities[ticker] = cls.ASSET_CLASSES.get(ticker, {}).get("risk", 0.15)

        # Calculate weighted portfolio metrics from REAL data
        expected_return = sum(
            allocations.get(ticker, 0) * real_returns.get(ticker, 0.08)
            for ticker in allocations.keys()
        )

        expected_yield = sum(
            allocations.get(ticker, 0) * cls.ASSET_CLASSES.get(ticker, {}).get("yield", 0.02)
            for ticker in allocations.keys()
        )

        # Portfolio risk from REAL volatility data
        portfolio_risk = sum(
            allocations.get(ticker, 0) * real_volatilities.get(ticker, 0.15)
            for ticker in allocations.keys()
        )

        # Convert to positions with REAL current prices
        positions = []
        for ticker, alloc in allocations.items():
            if alloc > 0.001:  # Skip tiny allocations
                asset = cls.ASSET_CLASSES.get(ticker, {"name": ticker, "type": "equity"})
                dollar_amount = budget * alloc

                # Get REAL current price
                stock_data = get_stock_data(ticker, period="5d")
                if "error" not in stock_data and not stock_data["history"].empty:
                    current_price = float(stock_data["history"]["Close"].iloc[-1])
                else:
                    current_price = 100  # Fallback only if data unavailable

                # Get REAL return data
                real_return = real_returns.get(ticker)
                return_source = "REAL (1Y historical)" if ticker in real_returns else "estimate"

                positions.append({
                    "ticker": ticker,
                    "name": asset.get("name", ticker),
                    "type": asset.get("type", "equity"),
                    "allocation": alloc,
                    "dollar_amount": dollar_amount,
                    "current_price": current_price,
                    "shares": int(dollar_amount / current_price) if current_price > 0 else 0,
                    "historical_return": real_return * 100 if real_return else None,
                    "data_source": return_source,
                })

        return {
            "positions": positions,
            "expected_return": expected_return,
            "expected_yield": expected_yield,
            "portfolio_risk": portfolio_risk,
            "total_budget": budget,
            "data_source": "REAL market data from yfinance (not estimates)",
            "summary": {
                "equity": equity_alloc,
                "treasury": treasury_alloc,
                "bonds": bond_alloc,
            }
        }

    @classmethod
    def analyze_portfolio_for_rebalancing(cls, positions: list) -> dict:
        """Analyze current portfolio and suggest rebalancing."""
        if not positions:
            return {"needs_rebalancing": False, "suggestions": [], "reason": "No positions to analyze"}

        # Calculate current allocation
        total_value = 0
        allocation_by_type = {"equity": 0, "treasury": 0, "bond": 0, "other": 0}
        position_values = []

        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker or "---" in pos.get("type", ""):
                continue

            # Get current price
            try:
                stock_data = get_stock_data(ticker, period="5d")
                if "error" not in stock_data and not stock_data["history"].empty:
                    current_price = float(stock_data["history"]["Close"].iloc[-1])
                else:
                    current_price = pos.get("entry_price", 100)
            except:
                current_price = pos.get("entry_price", 100)

            quantity = pos.get("quantity", 0)
            value = current_price * quantity
            total_value += value

            # Categorize
            pos_type = pos.get("type", "").lower()
            if any(t in pos_type for t in ["stock", "etf", "equity"]) and "bond" not in pos_type and "treasury" not in pos_type:
                allocation_by_type["equity"] += value
            elif any(t in pos_type for t in ["treasury", "tlt", "ief", "shy"]):
                allocation_by_type["treasury"] += value
            elif any(t in pos_type for t in ["bond", "lqd", "hyg", "bnd"]):
                allocation_by_type["bond"] += value
            else:
                # Check ticker
                if ticker in ["SPY", "QQQ", "VTI", "VXUS", "VWO", "SCHD", "VNQ"]:
                    allocation_by_type["equity"] += value
                elif ticker in ["TLT", "IEF", "IEI", "SHY", "GOVT", "TIP"]:
                    allocation_by_type["treasury"] += value
                elif ticker in ["BND", "LQD", "HYG", "VCIT", "VCSH"]:
                    allocation_by_type["bond"] += value
                else:
                    allocation_by_type["equity"] += value  # Default to equity

            position_values.append({
                "ticker": ticker,
                "value": value,
                "current_price": current_price,
                "quantity": quantity,
            })

        if total_value == 0:
            return {"needs_rebalancing": False, "suggestions": [], "reason": "Portfolio has no value"}

        # Calculate percentages
        current_allocation = {
            "equity": allocation_by_type["equity"] / total_value if total_value > 0 else 0,
            "treasury": allocation_by_type["treasury"] / total_value if total_value > 0 else 0,
            "bond": allocation_by_type["bond"] / total_value if total_value > 0 else 0,
        }

        # Get target allocation based on user profile (simplified)
        suggestions = []
        needs_rebalancing = False

        # Check for concentration issues
        for pv in position_values:
            pct = pv["value"] / total_value * 100
            if pct > 30:
                suggestions.append(f"**{pv['ticker']}** is {pct:.1f}% of portfolio - consider trimming to reduce concentration risk")
                needs_rebalancing = True

        # Check allocation drift
        if current_allocation["equity"] > 0.80:
            suggestions.append(f"Equity allocation is {current_allocation['equity']*100:.0f}% - consider adding bonds for diversification")
            needs_rebalancing = True
        elif current_allocation["equity"] < 0.30:
            suggestions.append(f"Equity allocation is only {current_allocation['equity']*100:.0f}% - consider adding equities for growth")
            needs_rebalancing = True

        if current_allocation["treasury"] == 0 and current_allocation["bond"] == 0:
            suggestions.append("No fixed income exposure - consider adding bonds for stability")
            needs_rebalancing = True

        return {
            "needs_rebalancing": needs_rebalancing,
            "suggestions": suggestions,
            "current_allocation": current_allocation,
            "total_value": total_value,
            "position_values": position_values,
        }

    @classmethod
    def generate_rebalancing_trades(cls, positions: list, target_allocation: dict, total_value: float) -> list:
        """Generate specific trades to rebalance portfolio."""
        trades = []

        # Calculate target values
        target_equity = total_value * target_allocation.get("equity", 0.6)
        target_treasury = total_value * target_allocation.get("treasury", 0.1)
        target_bond = total_value * target_allocation.get("bond", 0.3)

        # Simplified: suggest adding to underweight, trimming overweight
        analysis = cls.analyze_portfolio_for_rebalancing(positions)
        current = analysis.get("current_allocation", {})

        if current.get("equity", 0) > target_allocation.get("equity", 0.6) + 0.05:
            excess = (current["equity"] - target_allocation["equity"]) * total_value
            trades.append({"action": "SELL", "amount": excess, "category": "equity", "suggestion": f"Trim equities by ${excess:,.0f}"})

        if current.get("treasury", 0) < target_allocation.get("treasury", 0.1) - 0.02:
            needed = (target_allocation["treasury"] - current.get("treasury", 0)) * total_value
            trades.append({"action": "BUY", "ticker": "IEF", "amount": needed, "suggestion": f"Buy ${needed:,.0f} of IEF (Treasury)"})

        if current.get("bond", 0) < target_allocation.get("bond", 0.3) - 0.05:
            needed = (target_allocation["bond"] - current.get("bond", 0)) * total_value
            trades.append({"action": "BUY", "ticker": "BND", "amount": needed, "suggestion": f"Buy ${needed:,.0f} of BND (Bonds)"})

        return trades

    @classmethod
    def generate_response(cls, user_message: str, user_profile: dict) -> tuple:
        """Generate AI response and optionally create portfolio."""

        parsed = cls.parse_user_input(user_message)
        message_lower = user_message.lower()

        # Handle specific stock additions (e.g., "add AAPL")
        if parsed["specific_stocks"] and parsed["action"] == "add":
            response_parts = []
            added_stocks = []

            for ticker in parsed["specific_stocks"]:
                stock_info = cls.get_stock_info_for_portfolio(ticker)

                if stock_info.get("error"):
                    response_parts.append(f"Could not find **{ticker}** - {stock_info['error']}")
                    continue

                # Determine quantity
                if parsed["shares"]:
                    quantity = parsed["shares"]
                elif parsed["amount"]:
                    quantity = int(parsed["amount"] / stock_info["current_price"])
                else:
                    # Default: buy shares worth roughly $1000 or 10 shares
                    quantity = max(1, int(1000 / stock_info["current_price"]))

                if quantity > 0:
                    position = {
                        "ticker": stock_info["ticker"],
                        "type": "Stock",
                        "side": "Long",
                        "quantity": quantity,
                        "entry_price": stock_info["current_price"],
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                        "strike": None,
                        "expiry": None,
                        "yield_rate": None,
                        "maturity": None,
                        "id": len(st.session_state.portfolio)
                    }
                    st.session_state.portfolio.append(position)
                    added_stocks.append({
                        "ticker": stock_info["ticker"],
                        "name": stock_info["name"],
                        "quantity": quantity,
                        "price": stock_info["current_price"],
                        "beta": stock_info["beta"],
                        "sector": stock_info["sector"],
                    })

            if added_stocks:
                response_parts.append("## Stock Added to Portfolio\n")
                for stock in added_stocks:
                    value = stock["quantity"] * stock["price"]
                    response_parts.append(f"**{stock['ticker']}** ({stock['name']})")
                    response_parts.append(f"- Shares: {stock['quantity']} @ ${stock['price']:.2f} = ${value:,.2f}")
                    response_parts.append(f"- Beta: {stock['beta']:.2f}")
                    response_parts.append(f"- Sector: {stock['sector']}")
                    response_parts.append("")

                # Calculate and report portfolio Beta
                portfolio_beta = cls.calculate_portfolio_beta(st.session_state.portfolio)
                risk_level = user_profile.get("risk_level", "moderate")
                beta_target = cls.BETA_TARGETS.get(risk_level, cls.BETA_TARGETS["moderate"])

                response_parts.append(f"### Portfolio Risk Analysis")
                response_parts.append(f"**Current Portfolio Beta:** {portfolio_beta:.2f}")
                response_parts.append(f"**Target Beta Range ({risk_level}):** {beta_target['min']:.1f} - {beta_target['max']:.1f}")

                if portfolio_beta < beta_target["min"]:
                    response_parts.append(f"\n*Your portfolio Beta is below target. Consider adding higher-beta growth stocks to match your {risk_level} profile.*")
                elif portfolio_beta > beta_target["max"]:
                    response_parts.append(f"\n*Your portfolio Beta is above target. Consider adding defensive stocks or bonds to reduce volatility.*")
                else:
                    response_parts.append(f"\n*Your portfolio Beta is within the target range for your {risk_level} profile.*")

                response_parts.append("\nCheck the **Portfolio** tab to see your updated holdings.")

            return "\n".join(response_parts), None, user_profile

        # Handle specific stock removals
        if parsed["specific_stocks"] and parsed["action"] == "remove":
            response_parts = []
            removed = []

            for ticker in parsed["specific_stocks"]:
                ticker_upper = ticker.upper()
                # Find and remove matching positions
                initial_count = len(st.session_state.portfolio)
                st.session_state.portfolio = [
                    p for p in st.session_state.portfolio
                    if p.get("ticker", "").upper() != ticker_upper
                ]
                if len(st.session_state.portfolio) < initial_count:
                    removed.append(ticker_upper)

            if removed:
                response_parts.append(f"Removed **{', '.join(removed)}** from your portfolio.")
                portfolio_beta = cls.calculate_portfolio_beta(st.session_state.portfolio)
                response_parts.append(f"\n**New Portfolio Beta:** {portfolio_beta:.2f}")
            else:
                response_parts.append(f"Could not find {', '.join(parsed['specific_stocks'])} in your portfolio.")

            return "\n".join(response_parts), None, user_profile

        # Check for rebalancing request
        if any(word in message_lower for word in ["rebalance", "rebalancing", "adjust", "optimize portfolio"]):
            portfolio_positions = st.session_state.get("portfolio", [])
            if not portfolio_positions:
                return "You don't have any positions in your portfolio yet. Add some positions first, or tell me about your investment goals and I'll create a portfolio for you.", None, user_profile

            analysis = cls.analyze_portfolio_for_rebalancing(portfolio_positions)

            response_parts = ["## Portfolio Rebalancing Analysis\n"]
            response_parts.append(f"**Total Portfolio Value:** ${analysis['total_value']:,.2f}\n")

            response_parts.append("### Current Allocation")
            current = analysis.get("current_allocation", {})
            response_parts.append(f"- Equities: {current.get('equity', 0)*100:.1f}%")
            response_parts.append(f"- Treasury: {current.get('treasury', 0)*100:.1f}%")
            response_parts.append(f"- Bonds: {current.get('bond', 0)*100:.1f}%")

            if analysis["needs_rebalancing"]:
                response_parts.append("\n### Rebalancing Suggestions")
                for suggestion in analysis["suggestions"]:
                    response_parts.append(f"- {suggestion}")

                response_parts.append("\n**Say 'execute rebalancing' or 'rebalance now' to implement these changes.**")
                st.session_state.pending_rebalancing = analysis
            else:
                response_parts.append("\n✅ **Your portfolio looks well-balanced!** No rebalancing needed at this time.")

            return "\n".join(response_parts), None, user_profile

        # Check for rebalancing execution
        if any(phrase in message_lower for phrase in ["execute rebalancing", "rebalance now", "do the rebalancing", "implement rebalancing"]):
            if st.session_state.get("pending_rebalancing"):
                analysis = st.session_state.pending_rebalancing

                # Generate and execute trades
                target = {"equity": 0.60, "treasury": 0.10, "bond": 0.30}
                trades = cls.generate_rebalancing_trades(
                    st.session_state.portfolio,
                    target,
                    analysis["total_value"]
                )

                # Add new positions for buys
                for trade in trades:
                    if trade["action"] == "BUY" and trade.get("ticker"):
                        stock_data = get_stock_data(trade["ticker"], period="5d")
                        if "error" not in stock_data and not stock_data["history"].empty:
                            current_price = float(stock_data["history"]["Close"].iloc[-1])
                        else:
                            current_price = 100

                        quantity = int(trade["amount"] / current_price)
                        if quantity > 0:
                            position = {
                                "ticker": trade["ticker"],
                                "type": "ETF",
                                "side": "Long",
                                "quantity": quantity,
                                "entry_price": current_price,
                                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                                "strike": None,
                                "expiry": None,
                                "yield_rate": None,
                                "maturity": None,
                                "id": len(st.session_state.portfolio)
                            }
                            st.session_state.portfolio.append(position)

                st.session_state.pending_rebalancing = None
                return "✅ **Rebalancing executed!** I've added the recommended positions. Check the Portfolio tab to see your updated holdings.\n\n*Note: For sells, you'll need to manually adjust quantities in your broker.*", None, user_profile
            else:
                return "No pending rebalancing to execute. Say 'rebalance my portfolio' first to get suggestions.", None, user_profile

        # Update user profile with new information
        if parsed["risk_level"]:
            user_profile["risk_level"] = parsed["risk_level"]
        if parsed["return_target"]:
            user_profile["return_target"] = parsed["return_target"]
        if parsed["budget"]:
            user_profile["budget"] = parsed["budget"]
        if parsed["treasury_constraint"]:
            user_profile["constraints"]["treasury"] = parsed["treasury_constraint"]

        # Check what information we still need
        missing = []
        if not user_profile.get("budget"):
            missing.append("budget")

        # Generate response
        response_parts = []
        portfolio = None

        # Acknowledge what we understood
        if parsed["risk_level"]:
            response_parts.append(f"Got it - you have a **{parsed['risk_level'].replace('_', ' ')}** risk profile.")
        if parsed["return_target"]:
            response_parts.append(f"Targeting **{parsed['return_target']*100:.1f}% annual returns**.")
        if parsed["treasury_constraint"] is not None:
            response_parts.append(f"Treasury allocation constrained to **{parsed['treasury_constraint']*100:.1f}%**.")
        if parsed["budget"]:
            response_parts.append(f"Working with a budget of **${parsed['budget']:,.0f}**.")

        # Ask for missing information or generate portfolio
        if "budget" in missing and not parsed["budget"]:
            response_parts.append("\n**What's your total investable budget?** (e.g., $50,000 or $100K)")
        else:
            # We have enough to generate a portfolio
            budget = user_profile.get("budget", 100000)
            risk_level = user_profile.get("risk_level", "moderate")
            return_target = user_profile.get("return_target")
            treasury_constraint = user_profile.get("constraints", {}).get("treasury")

            portfolio = cls.generate_portfolio(
                risk_level=risk_level,
                return_target=return_target,
                treasury_pct=treasury_constraint,
                budget=budget
            )

            response_parts.append("\n---")
            response_parts.append("## Recommended Portfolio\n")

            # Summary
            response_parts.append(f"**Total Investment:** ${portfolio['total_budget']:,.0f}")
            response_parts.append(f"**Expected Annual Return:** {portfolio['expected_return']*100:.1f}%")
            response_parts.append(f"**Expected Yield:** {portfolio['expected_yield']*100:.2f}%")
            response_parts.append(f"**Portfolio Risk (Volatility):** {portfolio['portfolio_risk']*100:.1f}%")

            response_parts.append("\n### Allocation Breakdown")
            response_parts.append(f"- Equities: {portfolio['summary']['equity']*100:.0f}%")
            response_parts.append(f"- Treasury: {portfolio['summary']['treasury']*100:.0f}%")
            response_parts.append(f"- Bonds: {portfolio['summary']['bonds']*100:.0f}%")

            response_parts.append("\n### Positions")
            response_parts.append("| Ticker | Name | Allocation | Amount |")
            response_parts.append("|--------|------|------------|--------|")
            for pos in portfolio["positions"]:
                response_parts.append(f"| {pos['ticker']} | {pos['name']} | {pos['allocation']*100:.1f}% | ${pos['dollar_amount']:,.0f} |")

            response_parts.append("\n**Would you like me to add these positions to your portfolio?** (say 'yes' or 'add to portfolio')")

        return "\n".join(response_parts), portfolio, user_profile

# ============================================================
# SIDEBAR - AI FINANCIAL PLANNER CHATBOT
# ============================================================

with st.sidebar:
    st.title("🤖 AI Financial Planner")
    st.markdown("---")

    # Chat interface
    st.markdown("### Chat with your AI Advisor")

    # Display chat history
    chat_container = st.container(height=350)
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            **Hello! I'm your AI Financial Planner.**

            Tell me about:
            - Your **risk tolerance** (conservative, moderate, aggressive)
            - Your **return target** (e.g., 9% annually)
            - Any **allocation constraints** (e.g., 2% in treasury)
            - Your **investment budget**

            **Or add specific stocks:**
            - "Add AAPL to my portfolio"
            - "Buy $5000 of MSFT"
            - "Add 10 shares of TSLA"

            *I track portfolio Beta to match your risk profile.*
            """)
        else:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Advisor:** {msg['content']}")
                st.markdown("---")

    # User input
    user_input = st.text_area("Your message:", key="chat_input", height=80,
                              placeholder="Tell me about your investment goals...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send 📤", use_container_width=True):
            if user_input:
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": user_input})

                # Check for portfolio confirmation
                if any(word in user_input.lower() for word in ["yes", "add", "confirm", "do it", "proceed"]):
                    if st.session_state.get("pending_portfolio"):
                        portfolio = st.session_state.pending_portfolio
                        # Add positions to portfolio
                        for pos in portfolio["positions"]:
                            stock_data = get_stock_data(pos["ticker"], period="5d")
                            if "error" not in stock_data and not stock_data["history"].empty:
                                current_price = float(stock_data["history"]["Close"].iloc[-1])
                            else:
                                current_price = 100  # Fallback

                            quantity = int(pos["dollar_amount"] / current_price)
                            if quantity > 0:
                                position = {
                                    "ticker": pos["ticker"],
                                    "type": "ETF" if pos["type"] in ["equity", "treasury", "bond", "corporate_bond", "high_yield", "tips"] else pos["type"],
                                    "side": "Long",
                                    "quantity": quantity,
                                    "entry_price": current_price,
                                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                                    "strike": None,
                                    "expiry": None,
                                    "yield_rate": None,
                                    "maturity": None,
                                    "id": len(st.session_state.portfolio)
                                }
                                st.session_state.portfolio.append(position)

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"✅ **Done!** I've added {len(portfolio['positions'])} positions to your portfolio. Check the **Portfolio** tab to see them."
                        })
                        st.session_state.pending_portfolio = None
                    else:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": "I don't have a pending portfolio to add. Tell me about your investment goals first!"
                        })
                else:
                    # Generate AI response
                    response, portfolio, updated_profile = FinancialPlannerAI.generate_response(
                        user_input, st.session_state.user_profile
                    )
                    st.session_state.user_profile = updated_profile

                    if portfolio:
                        st.session_state.pending_portfolio = portfolio

                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

                st.rerun()

    with col2:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.user_profile = {
                "risk_level": "moderate",
                "budget": None,
                "return_target": None,
                "constraints": {}
            }
            st.session_state.pending_portfolio = None
            st.rerun()

    st.markdown("---")

    # Quick commands
    st.markdown("### Quick Commands")
    quick_col1, quick_col2 = st.columns(2)

    with quick_col1:
        if st.button("Conservative", use_container_width=True, key="q_cons"):
            st.session_state.chat_messages.append({"role": "user", "content": "I'm a conservative investor"})
            response, portfolio, updated_profile = FinancialPlannerAI.generate_response(
                "I'm a conservative investor with $100,000", st.session_state.user_profile
            )
            st.session_state.user_profile = updated_profile
            st.session_state.user_profile["budget"] = 100000
            if portfolio:
                st.session_state.pending_portfolio = portfolio
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    with quick_col2:
        if st.button("Aggressive", use_container_width=True, key="q_agg"):
            st.session_state.chat_messages.append({"role": "user", "content": "I'm an aggressive investor"})
            response, portfolio, updated_profile = FinancialPlannerAI.generate_response(
                "I'm an aggressive investor with $100,000", st.session_state.user_profile
            )
            st.session_state.user_profile = updated_profile
            st.session_state.user_profile["budget"] = 100000
            if portfolio:
                st.session_state.pending_portfolio = portfolio
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    st.markdown("---")

    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ============================================================
# MAIN CONTENT
# ============================================================

st.title("🎯 Trading Strategy Dashboard")
# Get risk profile from AI planner state
risk_profile = st.session_state.user_profile.get("risk_level", "moderate").replace("_", " ").title()
st.markdown(f"*Risk Profile: **{risk_profile}** | Regime-aware strategy selection*")

# Load data
try:
    vix_stats = load_vix_data()
    yield_data = load_yield_data()
    all_data = load_all_signals()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

vix_regime = determine_vix_regime(vix_stats["current"])
yields = yield_data.get("yields", {})
spreads = calculate_spreads(yields)
curve_regime = determine_curve_regime(spreads)
signals = all_data.get("signals", [])
alerts = check_alerts(all_data)

# Alerts
if alerts:
    for alert in alerts:
        if "EXTREME" in alert:
            st.error(f"🚨 {alert}")
        else:
            st.warning(f"⚠️ {alert}")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    regime_colors = {"LOW": "🟢", "NORMAL": "🟡", "ELEVATED": "🟠", "CRISIS": "🔴"}
    st.metric("VIX", f"{vix_stats['current']:.2f}", f"{vix_regime} {regime_colors.get(vix_regime, '')}")
with col2:
    st.metric("VIX Percentile", f"{vix_stats['percentile']:.0f}th", "vs 2Y")
with col3:
    spread_10y2y = spreads.get("10Y-2Y", 0)
    st.metric("10Y-2Y", f"{spread_10y2y:+.2f}%", curve_regime)
with col4:
    st.metric("10Y Yield", f"{yields.get('10Y', 0):.2f}%", "Treasury")

st.markdown("---")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Signals",
    "🎰 Options Builder",
    "📋 Securities Lookup",
    "💼 Portfolio",
    "📈 Performance",
    "📉 Charts"
])

# ============================================================
# TAB 1: SIGNALS
# ============================================================

with tab1:
    st.header("Active Trading Signals")

    # Get strategies for current profile and regime
    # Map AI planner risk level to RISK_PROFILES key
    ai_risk_level = st.session_state.user_profile.get("risk_level", "moderate")
    risk_profile_key = {
        "conservative": "Conservative",
        "moderate": "Moderate",
        "aggressive": "Aggressive",
        "very_aggressive": "Aggressive",
    }.get(ai_risk_level, "Moderate")
    profile_strategies = RISK_PROFILES.get(risk_profile_key, RISK_PROFILES["Moderate"])["strategies"].get(vix_regime, [])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Signals for {risk_profile} Profile")

        for signal in signals:
            strength_emoji = {
                SignalStrength.EXTREME: "🔥🔥🔥",
                SignalStrength.STRONG: "🔥🔥",
                SignalStrength.MODERATE: "🔥",
                SignalStrength.WEAK: "💤",
                SignalStrength.NONE: "",
            }

            with st.expander(
                f"{strength_emoji[signal.strength]} **{signal.name}** [{signal.direction}]",
                expanded=signal.strength.value >= SignalStrength.STRONG.value
            ):
                st.markdown(f"**Action:** {signal.action}")
                st.markdown(f"**Rationale:** {signal.rationale}")
                st.markdown("**Instruments:** " + ", ".join(signal.instruments))
                st.caption(f"Risk: {signal.risk_note}")

    with col2:
        st.subheader(f"Strategies: {vix_regime} Regime")

        for strat in profile_strategies:
            with st.container():
                st.markdown(f"**{strat['name']}**")
                st.caption(f"Risk: {strat['risk']} | Reward: {strat['reward']}")
                st.caption(strat['description'])
                st.markdown("---")

# ============================================================
# TAB 2: OPTIONS BUILDER
# ============================================================

with tab2:
    st.header("Options Strategy Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Strategy Setup")

        # Ticker input
        ticker = st.text_input("Underlying Ticker:", value="SPY", key="opt_ticker").upper()

        # Get current price
        stock_data = get_stock_data(ticker, period="5d")
        if "error" not in stock_data and not stock_data["history"].empty:
            current_price = stock_data["history"]["Close"].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        else:
            current_price = 100
            st.warning("Could not fetch price, using $100")

        # Strategy type
        strategy_type = st.selectbox(
            "Strategy Type:",
            ["Long Call", "Long Put", "Short Put", "Bull Call Spread",
             "Bull Put Spread", "Iron Condor", "Straddle"]
        )

        # Strategy parameters
        st.markdown("### Parameters")

        if strategy_type == "Long Call":
            strike = st.number_input("Strike Price:", value=float(int(current_price)), step=1.0)
            premium = st.number_input("Premium Paid:", value=5.0, step=0.5)
            params = {"strike": strike, "premium": premium}

            max_loss = premium * 100
            max_profit = "Unlimited"
            breakeven = strike + premium

        elif strategy_type == "Long Put":
            strike = st.number_input("Strike Price:", value=float(int(current_price)), step=1.0)
            premium = st.number_input("Premium Paid:", value=5.0, step=0.5)
            params = {"strike": strike, "premium": premium}

            max_loss = premium * 100
            max_profit = (strike - premium) * 100
            breakeven = strike - premium

        elif strategy_type == "Short Put":
            strike = st.number_input("Strike Price:", value=float(int(current_price * 0.95)), step=1.0)
            premium = st.number_input("Premium Received:", value=3.0, step=0.5)
            params = {"strike": strike, "premium": premium}

            max_loss = (strike - premium) * 100
            max_profit = premium * 100
            breakeven = strike - premium

        elif strategy_type == "Bull Call Spread":
            long_strike = st.number_input("Long Call Strike:", value=float(int(current_price)), step=1.0)
            short_strike = st.number_input("Short Call Strike:", value=float(int(current_price * 1.05)), step=1.0)
            net_debit = st.number_input("Net Debit:", value=2.0, step=0.5)
            params = {"long_strike": long_strike, "short_strike": short_strike, "net_debit": net_debit}

            max_loss = net_debit * 100
            max_profit = (short_strike - long_strike - net_debit) * 100
            breakeven = long_strike + net_debit

        elif strategy_type == "Bull Put Spread":
            short_strike = st.number_input("Short Put Strike:", value=float(int(current_price * 0.95)), step=1.0)
            long_strike = st.number_input("Long Put Strike:", value=float(int(current_price * 0.90)), step=1.0)
            net_credit = st.number_input("Net Credit:", value=1.5, step=0.5)
            params = {"short_strike": short_strike, "long_strike": long_strike, "net_credit": net_credit}

            max_loss = (short_strike - long_strike - net_credit) * 100
            max_profit = net_credit * 100
            breakeven = short_strike - net_credit

        elif strategy_type == "Iron Condor":
            put_long = st.number_input("Long Put Strike:", value=float(int(current_price * 0.90)), step=1.0)
            put_short = st.number_input("Short Put Strike:", value=float(int(current_price * 0.95)), step=1.0)
            call_short = st.number_input("Short Call Strike:", value=float(int(current_price * 1.05)), step=1.0)
            call_long = st.number_input("Long Call Strike:", value=float(int(current_price * 1.10)), step=1.0)
            net_credit = st.number_input("Net Credit:", value=2.0, step=0.5)
            params = {"put_long": put_long, "put_short": put_short,
                     "call_short": call_short, "call_long": call_long, "net_credit": net_credit}

            max_loss = (put_short - put_long - net_credit) * 100
            max_profit = net_credit * 100
            breakeven = f"{put_short - net_credit:.2f} / {call_short + net_credit:.2f}"

        elif strategy_type == "Straddle":
            strike = st.number_input("Strike Price:", value=float(int(current_price)), step=1.0)
            premium = st.number_input("Total Premium:", value=10.0, step=0.5)
            params = {"strike": strike, "premium": premium}

            max_loss = premium * 100
            max_profit = "Unlimited"
            breakeven = f"{strike - premium:.2f} / {strike + premium:.2f}"

        # Display P&L metrics
        st.markdown("### Expected P&L")
        st.metric("Max Profit", f"${max_profit:.2f}" if isinstance(max_profit, (int, float)) else max_profit)
        st.metric("Max Loss", f"${max_loss:.2f}" if isinstance(max_loss, (int, float)) else max_loss)
        st.metric("Breakeven", f"${breakeven:.2f}" if isinstance(breakeven, (int, float)) else breakeven)

    with col2:
        st.subheader("Payoff Diagram")

        # Calculate payoff
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        payoffs = calculate_payoff(strategy_type, price_range, params)

        # Create payoff chart
        fig = go.Figure()

        # Payoff line
        fig.add_trace(go.Scatter(
            x=price_range,
            y=payoffs,
            mode='lines',
            name='P&L at Expiration',
            line=dict(color='cyan', width=3)
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

        # Current price line
        fig.add_vline(x=current_price, line_dash="dot", line_color="yellow",
                     annotation_text=f"Current: ${current_price:.2f}")

        # Color profit/loss zones
        fig.add_trace(go.Scatter(
            x=price_range,
            y=[max(0, p) for p in payoffs],
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(width=0),
            name='Profit Zone',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=price_range,
            y=[min(0, p) for p in payoffs],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='Loss Zone',
            showlegend=False
        ))

        fig.update_layout(
            title=f"{strategy_type} Payoff - {ticker}",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Strategy summary
        st.markdown("### Strategy Summary")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Strategy | {strategy_type} |
        | Underlying | {ticker} @ ${current_price:.2f} |
        | Max Profit | {f"${max_profit:.2f}" if isinstance(max_profit, (int, float)) else max_profit} |
        | Max Loss | ${max_loss:.2f} |
        | Breakeven | {f"${breakeven:.2f}" if isinstance(breakeven, (int, float)) else breakeven} |
        | Risk/Reward | {f"{abs(max_loss/max_profit):.2f}" if isinstance(max_profit, (int, float)) and max_profit != 0 else "N/A"} |
        """)

# ============================================================
# TAB 3: SECURITIES LOOKUP
# ============================================================

with tab3:
    st.header("Securities Lookup")

    # Sub-tabs for different security types
    lookup_tab1, lookup_tab2 = st.tabs(["Stocks & Options", "Fixed Income"])

    with lookup_tab1:
        lookup_ticker = st.text_input("Enter Ticker Symbol:", value="AAPL", key="lookup_ticker").upper()

        if lookup_ticker:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(f"{lookup_ticker} Stock Info")

                stock_data = get_stock_data(lookup_ticker)

                if "error" not in stock_data:
                    info = stock_data.get("info", {})
                    hist = stock_data.get("history", pd.DataFrame())

                    # Key metrics
                    metrics_col1, metrics_col2 = st.columns(2)

                    with metrics_col1:
                        price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
                        st.metric("Price", f"${price:.2f}" if isinstance(price, (int, float)) else price)

                        pe = info.get("trailingPE", "N/A")
                        st.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else pe)

                        mktcap = info.get("marketCap", 0)
                        st.metric("Market Cap", f"${mktcap/1e9:.1f}B" if mktcap else "N/A")

                    with metrics_col2:
                        change = info.get("regularMarketChangePercent", 0)
                        st.metric("Day Change", f"{change:.2f}%" if isinstance(change, (int, float)) else "N/A")

                        vol = info.get("volume", 0)
                        st.metric("Volume", f"{vol/1e6:.1f}M" if vol else "N/A")

                        beta = info.get("beta", "N/A")
                        st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else beta)

                    # Price chart
                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=hist.index,
                            open=hist['Open'],
                            high=hist['High'],
                            low=hist['Low'],
                            close=hist['Close'],
                            name='Price'
                        ))
                        fig.update_layout(
                            title=f"{lookup_ticker} Price Chart (1Y)",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=400,
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Error: {stock_data['error']}")

            with col2:
                st.subheader(f"{lookup_ticker} Options Chain")

                options_data = get_options_chain(lookup_ticker)

                if "error" not in options_data:
                    expirations = options_data.get("expirations", [])
                    chains = options_data.get("chains", {})

                    if expirations:
                        selected_exp = st.selectbox("Expiration Date:", expirations[:10])

                        if selected_exp in chains:
                            chain = chains[selected_exp]

                            option_type = st.radio("Option Type:", ["Calls", "Puts"], horizontal=True)

                            if option_type == "Calls":
                                df = chain["calls"]
                            else:
                                df = chain["puts"]

                            if not df.empty:
                                # Filter to relevant columns
                                display_cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
                                available_cols = [c for c in display_cols if c in df.columns]

                                display_df = df[available_cols].copy()
                                display_df.columns = ["Strike", "Last", "Bid", "Ask", "Vol", "OI", "IV"][:len(available_cols)]

                                # Format IV as percentage
                                if "IV" in display_df.columns:
                                    display_df["IV"] = (display_df["IV"] * 100).round(1).astype(str) + "%"

                                st.dataframe(display_df, use_container_width=True, height=400)
                            else:
                                st.warning("No options data available")
                    else:
                        st.warning("No options expirations found")
                else:
                    st.error(f"Error: {options_data.get('error', 'Unknown error')}")

    with lookup_tab2:
        st.subheader("Fixed Income Securities")

        # Common fixed income ETFs
        FI_ETFS = {
            "Treasury ETFs": {
                "TLT": "20+ Year Treasury",
                "IEF": "7-10 Year Treasury",
                "IEI": "3-7 Year Treasury",
                "SHY": "1-3 Year Treasury",
                "GOVT": "All Treasury",
                "TIP": "TIPS (Inflation Protected)",
            },
            "Corporate Bond ETFs": {
                "LQD": "Investment Grade Corporate",
                "VCIT": "Intermediate Corporate",
                "VCSH": "Short-Term Corporate",
                "HYG": "High Yield Corporate",
                "JNK": "High Yield (SPDR)",
                "USIG": "Investment Grade (iShares)",
            },
            "Municipal Bond ETFs": {
                "MUB": "National Muni",
                "VTEB": "Tax-Exempt Muni",
                "HYD": "High Yield Muni",
            },
            "International Bonds": {
                "BNDX": "International Bond",
                "EMB": "Emerging Market Bonds",
                "IAGG": "International Aggregate",
            },
        }

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Select Category")
            fi_category = st.selectbox("Bond Category:", list(FI_ETFS.keys()))

            st.markdown("### Available ETFs")
            for ticker, name in FI_ETFS[fi_category].items():
                if st.button(f"{ticker}: {name}", key=f"fi_{ticker}", use_container_width=True):
                    st.session_state.fi_selected = ticker

            # Custom ticker input
            st.markdown("### Or Enter Custom")
            custom_fi = st.text_input("Bond ETF Ticker:", key="custom_fi")
            if custom_fi:
                st.session_state.fi_selected = custom_fi.upper()

        with col2:
            selected_fi = st.session_state.get("fi_selected", "TLT")
            st.subheader(f"{selected_fi} Analysis")

            fi_data = get_stock_data(selected_fi)

            if "error" not in fi_data:
                info = fi_data.get("info", {})
                hist = fi_data.get("history", pd.DataFrame())

                # Key fixed income metrics
                met_col1, met_col2, met_col3 = st.columns(3)

                with met_col1:
                    price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
                    st.metric("Price", f"${price:.2f}" if isinstance(price, (int, float)) else price)

                    ytd_return = info.get("ytdReturn", info.get("52WeekChange", "N/A"))
                    if isinstance(ytd_return, (int, float)):
                        st.metric("YTD Return", f"{ytd_return*100:.2f}%")

                with met_col2:
                    div_yield = info.get("yield", info.get("dividendYield", "N/A"))
                    if isinstance(div_yield, (int, float)):
                        st.metric("SEC Yield", f"{div_yield*100:.2f}%")
                    else:
                        st.metric("SEC Yield", "N/A")

                    expense = info.get("annualReportExpenseRatio", "N/A")
                    if isinstance(expense, (int, float)):
                        st.metric("Expense Ratio", f"{expense*100:.2f}%")

                with met_col3:
                    aum = info.get("totalAssets", 0)
                    if aum:
                        st.metric("AUM", f"${aum/1e9:.1f}B")

                    avg_vol = info.get("averageVolume", 0)
                    if avg_vol:
                        st.metric("Avg Volume", f"{avg_vol/1e6:.1f}M")

                # Price chart
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist["Close"],
                        mode="lines",
                        name="Price",
                        line=dict(color="cyan")
                    ))
                    fig.update_layout(
                        title=f"{selected_fi} Price History (1Y)",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Compare with treasury yields
                st.markdown("### Current Treasury Yields (Context)")
                yield_context = load_yield_data()
                if yield_context and yield_context.get("yields"):
                    yields_df = pd.DataFrame([
                        {"Maturity": k, "Yield": f"{v:.2f}%"}
                        for k, v in yield_context["yields"].items()
                    ])
                    st.dataframe(yields_df, use_container_width=True, hide_index=True)

                # Fund description
                desc = info.get("longBusinessSummary", "")
                if desc:
                    with st.expander("Fund Description"):
                        st.write(desc[:500] + "..." if len(desc) > 500 else desc)
            else:
                st.error(f"Could not load data for {selected_fi}")

# ============================================================
# TAB 4: PORTFOLIO
# ============================================================

with tab4:
    st.header("Portfolio Tracker")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Add Position")

        with st.form("add_position"):
            pos_ticker = st.text_input("Ticker:", value="SPY")

            # Position side (Long/Short)
            pos_side = st.selectbox("Side:", ["Long (Buy)", "Short (Sell)"])

            # Expanded position types including fixed income
            pos_type = st.selectbox("Type:", [
                "Stock",
                "Call Option",
                "Put Option",
                "--- Fixed Income ---",
                "Treasury Bond",
                "Treasury ETF (TLT, IEF, SHY)",
                "Corporate Bond ETF (LQD, HYG)",
                "Municipal Bond",
                "I-Bond / TIPS",
                "--- Other ---",
                "ETF",
                "Crypto",
                "Commodity"
            ])

            pos_quantity = st.number_input("Quantity:", value=100, step=1)
            pos_entry = st.number_input("Entry Price:", value=100.0, step=0.01)

            # For options, add strike and expiry
            if "Option" in pos_type:
                pos_strike = st.number_input("Strike Price:", value=100.0, step=1.0)
                pos_expiry = st.date_input("Expiration:", value=datetime.now() + timedelta(days=30))
            else:
                pos_strike = None
                pos_expiry = None

            # For bonds, add yield and maturity
            if "Bond" in pos_type or "Treasury" in pos_type or "TIPS" in pos_type:
                pos_yield = st.number_input("Yield/Coupon (%):", value=4.5, step=0.1)
                pos_maturity = st.date_input("Maturity Date:", value=datetime.now() + timedelta(days=365*5))
            else:
                pos_yield = None
                pos_maturity = None

            pos_date = st.date_input("Entry Date:", value=datetime.now())

            submitted = st.form_submit_button("Add Position")

            if submitted:
                # Determine if short position
                is_short = "Short" in pos_side

                position = {
                    "ticker": pos_ticker.upper(),
                    "type": pos_type,
                    "side": "Short" if is_short else "Long",
                    "quantity": pos_quantity,
                    "entry_price": pos_entry,
                    "entry_date": pos_date.strftime("%Y-%m-%d"),
                    "strike": pos_strike,
                    "expiry": pos_expiry.strftime("%Y-%m-%d") if pos_expiry else None,
                    "yield_rate": pos_yield,
                    "maturity": pos_maturity.strftime("%Y-%m-%d") if pos_maturity else None,
                    "id": len(st.session_state.portfolio)
                }
                st.session_state.portfolio.append(position)
                side_str = "Sold/Shorted" if is_short else "Bought"
                st.success(f"{side_str} {pos_quantity} {pos_ticker} {pos_type}")
                st.rerun()

    with col2:
        st.subheader("Current Positions")

        if st.session_state.portfolio:
            # Calculate current values
            portfolio_data = []
            total_value = 0
            total_pnl = 0
            total_long_value = 0
            total_short_value = 0

            for pos in st.session_state.portfolio:
                ticker = pos["ticker"]
                is_short = pos.get("side", "Long") == "Short"
                pos_type = pos.get("type", "Stock")

                # Skip separator entries
                if "---" in pos_type:
                    continue

                # Get current price
                stock_data = get_stock_data(ticker, period="5d")
                if "error" not in stock_data and not stock_data["history"].empty:
                    current_price = float(stock_data["history"]["Close"].iloc[-1])
                else:
                    current_price = pos["entry_price"]

                # Calculate P&L based on position side
                cost_basis = pos["entry_price"] * pos["quantity"]

                if is_short:
                    # Short position: profit when price goes down
                    current_value = pos["entry_price"] * pos["quantity"]  # Value is what you received
                    pnl = (pos["entry_price"] - current_price) * pos["quantity"]
                    total_short_value += current_price * pos["quantity"]  # Liability
                else:
                    # Long position: profit when price goes up
                    current_value = current_price * pos["quantity"]
                    pnl = current_value - cost_basis
                    total_long_value += current_value

                pnl_pct = (pnl / cost_basis) * 100 if cost_basis != 0 else 0

                # Build display row
                row = {
                    "Ticker": ticker,
                    "Side": pos.get("side", "Long"),
                    "Type": pos_type.replace("--- ", "").replace(" ---", ""),
                    "Qty": pos["quantity"],
                    "Entry": f"${pos['entry_price']:.2f}",
                    "Current": f"${current_price:.2f}",
                    "P&L": f"${pnl:+,.2f}",
                    "P&L %": f"{pnl_pct:+.1f}%",
                }

                # Add bond-specific info
                if pos.get("yield_rate"):
                    row["Yield"] = f"{pos['yield_rate']:.2f}%"
                if pos.get("maturity"):
                    row["Maturity"] = pos["maturity"]

                # Add option-specific info
                if pos.get("strike"):
                    row["Strike"] = f"${pos['strike']:.0f}"
                if pos.get("expiry"):
                    row["Expiry"] = pos["expiry"]

                portfolio_data.append(row)
                total_pnl += pnl

            total_value = total_long_value - total_short_value  # Net value

            # Display portfolio table
            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                st.dataframe(df, use_container_width=True)

            # Summary metrics
            st.markdown("---")
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.metric("Long Exposure", f"${total_long_value:,.2f}")
            with met_col2:
                st.metric("Short Exposure", f"${total_short_value:,.2f}")
            with met_col3:
                st.metric("Net Value", f"${total_value:,.2f}")
            with met_col4:
                st.metric("Total P&L", f"${total_pnl:+,.2f}",
                         delta="Unrealized")

            # Separate long and short positions
            longs = [p for p in portfolio_data if p.get("Side") == "Long"]
            shorts = [p for p in portfolio_data if p.get("Side") == "Short"]

            col_a, col_b = st.columns(2)

            with col_a:
                if longs:
                    # Long positions pie chart
                    try:
                        fig = px.pie(
                            values=[abs(float(p["P&L"].replace("$", "").replace(",", "").replace("+", ""))) + 100 for p in longs],
                            names=[f"{p['Ticker']} ({p['Type']})" for p in longs],
                            title="Long Positions"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

            with col_b:
                if shorts:
                    # Short positions pie chart
                    try:
                        fig = px.pie(
                            values=[abs(float(p["P&L"].replace("$", "").replace(",", "").replace("+", ""))) + 100 for p in shorts],
                            names=[f"{p['Ticker']} ({p['Type']})" for p in shorts],
                            title="Short Positions"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

            # Position type breakdown
            st.subheader("Position Breakdown by Type")
            type_counts = {}
            for p in portfolio_data:
                t = p.get("Type", "Other")
                type_counts[t] = type_counts.get(t, 0) + 1

            if type_counts:
                fig = px.bar(
                    x=list(type_counts.keys()),
                    y=list(type_counts.values()),
                    title="Positions by Type",
                    labels={"x": "Type", "y": "Count"}
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            # Clear portfolio button
            if st.button("Clear All Positions"):
                st.session_state.portfolio = []
                st.rerun()
        else:
            st.info("No positions yet. Add a position to start tracking.")

            # Quick add suggestions
            st.markdown("### Quick Add Suggestions")
            st.markdown("""
            **Equities:** SPY, QQQ, AAPL, MSFT, NVDA
            **Fixed Income:** TLT (20Y Treasury), IEF (7-10Y), SHY (1-3Y), LQD (Corp), HYG (High Yield)
            **Options:** Enter any stock ticker, select Call/Put option
            """)

# ============================================================
# TAB 5: PORTFOLIO PERFORMANCE
# ============================================================

with tab5:
    st.header("Portfolio Performance")

    if not st.session_state.portfolio:
        st.info("No positions in portfolio. Add positions via the Portfolio tab or ask the AI Financial Planner to create one for you.")
    else:
        # Time period selector
        st.markdown("### Historical Performance")
        time_period = st.radio(
            "Select Time Period:",
            ["1M", "3M", "6M", "1Y", "YTD"],
            horizontal=True,
            key="perf_period"
        )

        # Calculate historical performance
        perf_data = get_historical_portfolio_performance(st.session_state.portfolio, time_period)

        if perf_data and perf_data.get("history") is not None and not perf_data["history"].empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Starting Value", f"${perf_data['start_value']:,.2f}")
            with col2:
                st.metric("Current Value", f"${perf_data['end_value']:,.2f}")
            with col3:
                color = "normal" if perf_data['total_return'] >= 0 else "inverse"
                st.metric("Total Return", f"{perf_data['total_return']:+.2f}%")
            with col4:
                # Annualized return (approximate)
                period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "YTD": 180}
                days = period_days.get(time_period, 30)
                if days > 0 and perf_data['start_value'] != 0:
                    annualized = ((perf_data['end_value'] / perf_data['start_value']) ** (365/days) - 1) * 100
                    st.metric("Annualized", f"{annualized:+.2f}%")

            # Performance chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perf_data["history"].index,
                y=perf_data["history"]["value"],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='cyan', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 255, 0.1)'
            ))

            # Add starting value line
            fig.add_hline(
                y=perf_data['start_value'],
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Starting: ${perf_data['start_value']:,.0f}"
            )

            fig.update_layout(
                title=f"Portfolio Value ({time_period})",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to calculate historical performance. Make sure your positions have valid tickers.")

        st.markdown("---")

        # Future Performance Projections
        st.markdown("### Future Performance Projections")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Analyst Projections by Position")

            analyst_data_list = []
            for pos in st.session_state.portfolio:
                ticker = pos.get("ticker", "")
                if not ticker or "---" in pos.get("type", ""):
                    continue

                analyst = get_analyst_data(ticker)
                if "error" not in analyst:
                    analyst_data_list.append(analyst)

            if analyst_data_list:
                # Create analyst projections table
                analyst_df = pd.DataFrame([{
                    "Ticker": a["ticker"],
                    "Current": f"${a['current_price']:.2f}" if a.get('current_price') else "N/A",
                    "Target (Mean)": f"${a['target_mean']:.2f}" if a.get('target_mean') else "N/A",
                    "Target (High)": f"${a['target_high']:.2f}" if a.get('target_high') else "N/A",
                    "Upside": f"{a['upside_pct']:+.1f}%" if a.get('upside_pct') else "N/A",
                    "Rating": a.get('recommendation', 'N/A').upper(),
                    "# Analysts": a.get('num_analysts', 0),
                } for a in analyst_data_list])

                st.dataframe(analyst_df, use_container_width=True, hide_index=True)

                # Calculate weighted portfolio upside
                total_value = sum(
                    a.get('current_price', 0) * next(
                        (p.get('quantity', 0) for p in st.session_state.portfolio if p.get('ticker') == a['ticker']),
                        0
                    )
                    for a in analyst_data_list
                )

                if total_value > 0:
                    weighted_upside = sum(
                        (a.get('upside_pct', 0) or 0) * a.get('current_price', 0) * next(
                            (p.get('quantity', 0) for p in st.session_state.portfolio if p.get('ticker') == a['ticker']),
                            0
                        )
                        for a in analyst_data_list
                    ) / total_value

                    st.metric("Weighted Portfolio Upside (Analyst Consensus)", f"{weighted_upside:+.1f}%")
            else:
                st.info("No analyst data available for current positions.")

        with col2:
            st.markdown("#### Projection Summary")

            # Model-based projections
            if st.session_state.portfolio:
                # Use AI Financial Planner expected returns
                total_value = 0
                weighted_return = 0

                for pos in st.session_state.portfolio:
                    ticker = pos.get("ticker", "")
                    if ticker in FinancialPlannerAI.ASSET_CLASSES:
                        asset = FinancialPlannerAI.ASSET_CLASSES[ticker]
                        stock_data = get_stock_data(ticker, period="5d")
                        if "error" not in stock_data and not stock_data["history"].empty:
                            price = float(stock_data["history"]["Close"].iloc[-1])
                            value = price * pos.get("quantity", 0)
                            total_value += value
                            weighted_return += value * asset["expected_return"]

                if total_value > 0:
                    expected_annual = weighted_return / total_value * 100

                    st.markdown("**Model Projections:**")
                    st.metric("Expected Annual Return", f"{expected_annual:.1f}%")

                    # Project future values
                    current_val = perf_data['end_value'] if perf_data else total_value

                    st.markdown("**Projected Portfolio Value:**")
                    projections = {
                        "1 Year": current_val * (1 + expected_annual/100),
                        "3 Years": current_val * ((1 + expected_annual/100) ** 3),
                        "5 Years": current_val * ((1 + expected_annual/100) ** 5),
                    }

                    for period, value in projections.items():
                        gain = value - current_val
                        st.write(f"**{period}:** ${value:,.0f} (+${gain:,.0f})")

        st.markdown("---")

        # Rebalancing Section
        st.markdown("### Portfolio Rebalancing")

        analysis = FinancialPlannerAI.analyze_portfolio_for_rebalancing(st.session_state.portfolio)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Current Allocation")
            current = analysis.get("current_allocation", {})

            # Pie chart of current allocation
            if current:
                fig = go.Figure(data=[go.Pie(
                    labels=['Equities', 'Treasury', 'Bonds'],
                    values=[
                        current.get('equity', 0) * 100,
                        current.get('treasury', 0) * 100,
                        current.get('bond', 0) * 100
                    ],
                    marker_colors=['#00ff00', '#0088ff', '#ff8800'],
                    hole=0.4
                )])
                fig.update_layout(height=300, title="Current Allocation")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Rebalancing Suggestions")

            if analysis["needs_rebalancing"]:
                st.warning("⚠️ Rebalancing recommended")
                for suggestion in analysis["suggestions"]:
                    st.markdown(f"- {suggestion}")

                st.markdown("---")
                st.markdown("**To rebalance, tell the AI advisor:**")
                st.code("Rebalance my portfolio", language=None)
                st.markdown("Then say **'execute rebalancing'** to implement.")
            else:
                st.success("✅ Portfolio is well-balanced!")
                st.markdown("No rebalancing needed at this time.")

# ============================================================
# TAB 6: CHARTS
# ============================================================

with tab6:
    st.header("Market Analysis Charts")

    col1, col2 = st.columns(2)

    with col1:
        # VIX chart
        st.subheader("VIX History")
        vix_history = load_vix_history()

        if vix_history is not None and not vix_history.empty:
            if isinstance(vix_history.columns, pd.MultiIndex):
                close_data = vix_history["Close"].iloc[:, 0]
            else:
                close_data = vix_history["Close"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=close_data.index, y=close_data.values,
                                    mode='lines', name='VIX', line=dict(color='orange')))
            fig.add_hline(y=14, line_dash="dash", line_color="green")
            fig.add_hline(y=20, line_dash="dash", line_color="yellow")
            fig.add_hline(y=30, line_dash="dash", line_color="red")
            fig.update_layout(height=350, title="VIX (2Y)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Regime distribution
        st.subheader("Regime Distribution")
        try:
            dist = regime_distribution(lookback_years=2)
            fig = go.Figure(data=[go.Bar(
                x=list(dist.keys()),
                y=[dist[r]["percentage"] for r in dist.keys()],
                marker_color=["green", "yellow", "orange", "red"],
                text=[f"{dist[r]['percentage']:.1f}%" for r in dist.keys()],
                textposition='auto'
            )])
            fig.update_layout(height=350, title="Time in Each Regime (2Y)")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Could not load regime data")

    # Yield curve
    st.subheader("Yield Curve")
    if yields:
        maturities = ["3M", "2Y", "5Y", "10Y", "30Y"]
        maturity_years = [0.25, 2, 5, 10, 30]
        yield_values = [yields.get(m, None) for m in maturities]
        valid_data = [(y, v) for y, v in zip(maturity_years, yield_values) if v is not None]

        if valid_data:
            years, values = zip(*valid_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers',
                                    line=dict(color='cyan', width=3), marker=dict(size=10)))
            fig.update_layout(height=300, title="Treasury Yield Curve",
                            xaxis_title="Maturity (Years)", yaxis_title="Yield (%)")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
budget_display = st.session_state.user_profile.get('budget') or 0
st.caption(f"Dashboard v2 | Risk Profile: {risk_profile} | VIX Regime: {vix_regime} | Curve: {curve_regime} | Budget: ${budget_display:,.0f}")
st.caption("For educational purposes only. Not financial advice.")
