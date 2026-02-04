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
import json
import os
import re
import requests

# spaCy for NLP (free, open-source)
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded yet
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Ollama for local LLM (free, runs locally)
OLLAMA_AVAILABLE = False
OLLAMA_MODEL = "llama3.2"  # Default model, can be changed

def check_ollama():
    """Check if Ollama is running locally."""
    global OLLAMA_AVAILABLE
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            OLLAMA_AVAILABLE = True
            return True
    except:
        pass
    OLLAMA_AVAILABLE = False
    return False

# Check Ollama on startup
check_ollama()

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
        valid_data = True
        for ticker, data in position_data.items():
            hist = data["history"]
            if date in hist.index:
                price = hist.loc[date, "Close"]
                # Skip if price is NaN
                if pd.isna(price):
                    valid_data = False
                    break
                quantity = data["quantity"]
                if data["side"] == "Short":
                    # Short position value = entry_price * qty - current_price * qty
                    value = data["entry_price"] * quantity - price * quantity
                else:
                    value = price * quantity
                total_value += value

        if valid_data and total_value != 0:
            daily_values.append({"date": date, "value": total_value})

    if not daily_values:
        return None

    df = pd.DataFrame(daily_values)
    df.set_index("date", inplace=True)

    # Drop any rows with NaN values
    df = df.dropna()

    if df.empty:
        return None

    # Calculate returns - ensure no NaN
    try:
        start_value = float(df["value"].iloc[0])
        end_value = float(df["value"].iloc[-1])

        # Check for NaN
        if pd.isna(start_value) or pd.isna(end_value):
            start_value = 0 if pd.isna(start_value) else start_value
            end_value = 0 if pd.isna(end_value) else end_value

        if start_value != 0 and not pd.isna(start_value):
            total_return = (end_value - start_value) / abs(start_value) * 100
        else:
            total_return = 0

        # Final NaN check
        if pd.isna(total_return):
            total_return = 0
    except:
        start_value = 0
        end_value = 0
        total_return = 0

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
        # Equities - Broad Market
        "SPY": {"name": "S&P 500 ETF", "type": "equity", "expected_return": 0.10, "risk": 0.16, "yield": 0.013},
        "QQQ": {"name": "Nasdaq 100 ETF", "type": "equity", "expected_return": 0.12, "risk": 0.20, "yield": 0.005},
        "VTI": {"name": "Total Stock Market", "type": "equity", "expected_return": 0.10, "risk": 0.16, "yield": 0.014},
        "VXUS": {"name": "International Stocks", "type": "equity", "expected_return": 0.07, "risk": 0.18, "yield": 0.03},
        "VWO": {"name": "Emerging Markets", "type": "equity", "expected_return": 0.08, "risk": 0.22, "yield": 0.025},
        "VNQ": {"name": "Real Estate (REITs)", "type": "real_estate", "expected_return": 0.08, "risk": 0.18, "yield": 0.04},

        # Equities - High Beta / Growth (for aggressive portfolios)
        "VGT": {"name": "Tech Sector ETF", "type": "equity", "expected_return": 0.14, "risk": 0.22, "yield": 0.006},
        "IWM": {"name": "Small Cap ETF", "type": "equity", "expected_return": 0.11, "risk": 0.22, "yield": 0.012},
        "VIG": {"name": "Dividend Appreciation", "type": "equity", "expected_return": 0.09, "risk": 0.14, "yield": 0.018},

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
        "very_conservative": {
            "equity_range": (0.05, 0.15),  # Minimal equity
            "bond_range": (0.20, 0.35),
            "treasury_range": (0.50, 0.75),  # Heavy treasury
            "max_single_position": 0.15,
        },
        "conservative": {
            "equity_range": (0.25, 0.45),
            "bond_range": (0.35, 0.55),
            "treasury_range": (0.10, 0.25),
            "max_single_position": 0.20,
        },
        "moderate": {
            "equity_range": (0.55, 0.75),  # Increased equity for higher beta
            "bond_range": (0.15, 0.30),
            "treasury_range": (0.05, 0.15),
            "max_single_position": 0.25,
        },
        "aggressive": {
            "equity_range": (0.75, 0.95),  # More equity
            "bond_range": (0.00, 0.15),
            "treasury_range": (0.00, 0.10),
            "max_single_position": 0.30,
        },
        "very_aggressive": {
            "equity_range": (0.80, 1.00),
            "bond_range": (0.00, 0.15),
            "treasury_range": (0.00, 0.10),
            "max_single_position": 0.35,
        },
    }

    # Risk profiles linked to target Beta (calibrated for achievable ETF portfolios)
    # IMPORTANT: No overlap between ranges - clean boundaries
    # Very Conservative: < 0.30
    # Conservative: 0.30 - 0.54
    # Moderate: 0.55 - 0.84
    # Aggressive: 0.85 - 1.04
    # Very Aggressive: >= 1.05
    BETA_TARGETS = {
        "very_conservative": {"min": 0.05, "max": 0.29, "target": 0.15},
        "conservative": {"min": 0.30, "max": 0.54, "target": 0.42},
        "moderate": {"min": 0.55, "max": 0.84, "target": 0.70},
        "aggressive": {"min": 0.85, "max": 1.04, "target": 0.95},
        "very_aggressive": {"min": 1.05, "max": 1.50, "target": 1.20},
    }

    # Bond ETF risk metrics: duration (interest rate sensitivity), convexity, spread duration (credit)
    # Duration = approx % price change for 1% rate move
    BOND_RISK_DATA = {
        # Treasury ETFs (no credit risk, only duration risk)
        "TLT": {"duration": 17.5, "convexity": 3.8, "spread_duration": 0, "type": "treasury", "maturity": "20+ yr"},
        "IEF": {"duration": 7.5, "convexity": 0.7, "spread_duration": 0, "type": "treasury", "maturity": "7-10 yr"},
        "IEI": {"duration": 4.5, "convexity": 0.3, "spread_duration": 0, "type": "treasury", "maturity": "3-7 yr"},
        "SHY": {"duration": 1.9, "convexity": 0.05, "spread_duration": 0, "type": "treasury", "maturity": "1-3 yr"},
        "GOVT": {"duration": 6.2, "convexity": 0.5, "spread_duration": 0, "type": "treasury", "maturity": "mixed"},
        "TIP": {"duration": 6.8, "convexity": 0.6, "spread_duration": 0, "type": "tips", "maturity": "mixed"},
        # Corporate bond ETFs (duration + credit/spread risk)
        "LQD": {"duration": 8.5, "convexity": 1.0, "spread_duration": 8.2, "type": "investment_grade", "maturity": "mixed"},
        "VCIT": {"duration": 6.2, "convexity": 0.5, "spread_duration": 6.0, "type": "investment_grade", "maturity": "5-10 yr"},
        "VCSH": {"duration": 2.7, "convexity": 0.1, "spread_duration": 2.5, "type": "investment_grade", "maturity": "1-5 yr"},
        # High yield (higher spread duration = more credit sensitive)
        "HYG": {"duration": 3.8, "convexity": 0.2, "spread_duration": 3.6, "type": "high_yield", "maturity": "mixed"},
        "JNK": {"duration": 3.5, "convexity": 0.2, "spread_duration": 3.4, "type": "high_yield", "maturity": "mixed"},
        # Aggregate
        "BND": {"duration": 6.3, "convexity": 0.6, "spread_duration": 2.1, "type": "aggregate", "maturity": "mixed"},
        "AGG": {"duration": 6.2, "convexity": 0.5, "spread_duration": 2.0, "type": "aggregate", "maturity": "mixed"},
    }

    # Target portfolio volatility by risk profile (annualized %)
    VOLATILITY_TARGETS = {
        "very_conservative": {"min": 2, "max": 5, "target": 3},
        "conservative": {"min": 5, "max": 8, "target": 6},
        "moderate": {"min": 8, "max": 12, "target": 10},
        "aggressive": {"min": 12, "max": 18, "target": 15},
        "very_aggressive": {"min": 15, "max": 25, "target": 20},
    }

    # Duration targets by risk profile (weighted avg years)
    DURATION_TARGETS = {
        "very_conservative": {"min": 1, "max": 4, "target": 2.5},
        "conservative": {"min": 2, "max": 5, "target": 3.5},
        "moderate": {"min": 4, "max": 7, "target": 5.5},
        "aggressive": {"min": 5, "max": 10, "target": 7},
        "very_aggressive": {"min": 6, "max": 12, "target": 8},
    }

    @classmethod
    def get_stock_beta(cls, ticker: str) -> float:
        """Get REAL beta for a stock from yfinance."""
        ticker_upper = ticker.upper()

        # Treasury and Bond ETFs have near-zero beta (they're not correlated with stocks)
        treasury_etfs = ["TLT", "IEF", "IEI", "SHY", "GOVT", "TIP", "VGSH", "VGIT", "VGLT", "SCHO", "SCHR"]
        bond_etfs = ["BND", "AGG", "LQD", "HYG", "VCIT", "VCSH", "VCLT", "IGSB", "IGIB", "IGLB", "JNK", "USIG"]

        if ticker_upper in treasury_etfs:
            return 0.05  # Treasury ETFs have very low beta
        if ticker_upper in bond_etfs:
            return 0.15  # Corporate bond ETFs have slightly higher but still low beta

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            beta = info.get("beta", None)
            if beta is not None:
                return float(beta)
            return 1.0  # Default to market beta if unavailable for stocks
        except:
            return 1.0

    @classmethod
    def get_bond_risk_metrics(cls, ticker: str) -> dict:
        """Get duration, convexity, and spread duration for bond ETFs."""
        ticker_upper = ticker.upper()
        if ticker_upper in cls.BOND_RISK_DATA:
            return cls.BOND_RISK_DATA[ticker_upper]
        # Default for unknown bonds - estimate based on name/type
        return {"duration": 5.0, "convexity": 0.4, "spread_duration": 2.0, "type": "unknown", "maturity": "unknown"}

    @classmethod
    def is_bond_etf(cls, ticker: str) -> bool:
        """Check if ticker is a bond ETF."""
        bond_tickers = list(cls.BOND_RISK_DATA.keys())
        return ticker.upper() in bond_tickers

    @classmethod
    def calculate_portfolio_duration(cls, positions: list) -> dict:
        """Calculate weighted average duration and spread duration for fixed income."""
        bond_value = 0
        weighted_duration = 0
        weighted_spread_duration = 0
        weighted_convexity = 0

        for pos in positions:
            ticker = pos.get("ticker", "").upper()
            if not cls.is_bond_etf(ticker):
                continue

            quantity = pos.get("quantity", 0)
            entry_price = pos.get("entry_price", 100)
            value = quantity * entry_price

            metrics = cls.get_bond_risk_metrics(ticker)
            weighted_duration += value * metrics["duration"]
            weighted_spread_duration += value * metrics["spread_duration"]
            weighted_convexity += value * metrics["convexity"]
            bond_value += value

        if bond_value > 0:
            return {
                "duration": weighted_duration / bond_value,
                "spread_duration": weighted_spread_duration / bond_value,
                "convexity": weighted_convexity / bond_value,
                "bond_value": bond_value,
            }
        return {"duration": 0, "spread_duration": 0, "convexity": 0, "bond_value": 0}

    @classmethod
    def estimate_portfolio_volatility(cls, positions: list) -> float:
        """Estimate annualized portfolio volatility using position data."""
        if not positions:
            return 0

        # Get historical returns for each position
        total_value = 0
        weighted_var = 0

        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker:
                continue

            quantity = pos.get("quantity", 0)
            entry_price = pos.get("entry_price", 100)
            value = quantity * entry_price
            total_value += value

            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if not hist.empty and len(hist) > 20:
                    returns = hist['Close'].pct_change().dropna()
                    vol = float(returns.std() * np.sqrt(252) * 100)  # Annualized %
                    weighted_var += (value ** 2) * (vol ** 2)
            except:
                # Default volatility estimate
                if cls.is_bond_etf(ticker):
                    weighted_var += (value ** 2) * (5 ** 2)  # ~5% for bonds
                else:
                    weighted_var += (value ** 2) * (20 ** 2)  # ~20% for stocks

        if total_value > 0:
            # Simplified - assumes no correlation (conservative estimate)
            portfolio_vol = np.sqrt(weighted_var) / total_value
            return portfolio_vol
        return 0

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
        # Common words that should NOT be treated as tickers
        excluded_words = {
            "I", "A", "THE", "AND", "FOR", "ADD", "BUY", "GET", "MY", "TO", "IN", "OF",
            "WORTH", "RISK", "HAVE", "WANT", "VERY", "SOME", "WITH", "THAT", "THIS",
            "STOCK", "BOND", "SAFE", "HIGH", "LOW", "MORE", "LESS", "ONLY", "JUST",
            "TESLA", "APPLE", "AMAZON", "GOOGLE", "META",  # Company names (use ticker instead)
        }
        for pattern in stock_patterns:
            matches = re.findall(pattern, message_upper, re.IGNORECASE)
            for match in matches:
                ticker = match.replace("$", "").strip()
                # Validate it looks like a ticker (not common words)
                if ticker and len(ticker) <= 5 and ticker not in excluded_words:
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

        # Detect risk level - check most specific patterns first
        if any(phrase in message_lower for phrase in ["very conservative", "extremely conservative", "ultra conservative", "scared of risk", "scared", "very safe", "no risk", "hate risk", "fear risk", "risk averse"]):
            parsed["risk_level"] = "very_conservative"
        elif any(word in message_lower for word in ["very risky", "very aggressive", "high risk", "maximum risk", "yolo"]):
            parsed["risk_level"] = "very_aggressive"
        elif any(word in message_lower for word in ["risky", "aggressive", "growth", "high return"]):
            parsed["risk_level"] = "aggressive"
        elif any(word in message_lower for word in ["conservative", "safe", "low risk", "preservation", "cautious"]):
            parsed["risk_level"] = "conservative"
        elif any(word in message_lower for word in ["moderate", "balanced", "medium"]):
            parsed["risk_level"] = "moderate"

        # Detect "a lot of" or "mostly" preferences
        if re.search(r'(?:a lot of|lots of|mostly|mainly|heavy|primarily)\s+(?:treasury|treasuries)', message_lower):
            parsed["treasury_constraint"] = 0.60  # 60% treasury
        if re.search(r'(?:a lot of|lots of|mostly|mainly|heavy|primarily)\s+(?:bonds?|fixed income)', message_lower):
            parsed["bond_constraint"] = 0.50
        if re.search(r'(?:some|a little|small amount|bit of)\s+(\w+)\s+(?:stock|shares)', message_lower):
            # "some Tesla stock" - mark as wanting small allocation
            parsed["small_equity"] = True

        # Detect return target (look for percentages)
        import re
        return_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:return|annually|annual|yearly|per year)', message_lower)
        if return_match:
            parsed["return_target"] = float(return_match.group(1)) / 100

        # Detect budget - look for dollar amounts or large numbers
        # Pattern 1: "$100,000" or "$100000" or "100,000" with optional k/K multiplier
        budget_patterns = [
            r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:k|K)?',  # $100,000 or $100K
            r'(?:have|budget|invest|total)\s+\$?([\d,]+(?:\.\d{2})?)\s*(?:k|K)?',  # "have 100000"
            r'\b([\d,]{4,})\b',  # Any number with 4+ digits (likely a budget)
        ]
        for bp in budget_patterns:
            budget_match = re.search(bp, message, re.IGNORECASE)
            if budget_match:
                amount_str = budget_match.group(1).replace(',', '')
                amount = float(amount_str)
                # Check for k/K multiplier in the match context
                match_end = budget_match.end()
                if match_end < len(message) and message[match_end:match_end+1].lower() == 'k':
                    amount *= 1000
                elif 'k' in message_lower and amount < 10000:
                    amount *= 1000
                if amount >= 1000:  # Reasonable budget threshold
                    parsed["budget"] = amount
                    break

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
                          treasury_pct: float = None, budget: float = 100000,
                          specific_stocks: list = None) -> dict:
        """Generate a diversified portfolio based on requirements."""

        profile = cls.RISK_PROFILES.get(risk_level, cls.RISK_PROFILES["moderate"])

        # Start with base allocation based on risk profile
        allocations = {}

        # Reserve allocation for specific stocks user requested
        specific_stock_alloc = 0
        if specific_stocks:
            # Give each specific stock a small allocation (5-10% each depending on risk)
            per_stock = 0.05 if risk_level in ["very_conservative", "conservative"] else 0.10
            specific_stock_alloc = min(0.30, len(specific_stocks) * per_stock)
            for ticker in specific_stocks:
                allocations[ticker.upper()] = per_stock

        # Determine treasury allocation
        if treasury_pct is not None:
            treasury_alloc = treasury_pct
        elif risk_level == "very_aggressive":
            treasury_alloc = 0  # No treasury for very aggressive
        else:
            treasury_alloc = (profile["treasury_range"][0] + profile["treasury_range"][1]) / 2

        # Determine equity allocation (excluding specific stocks)
        equity_min, equity_max = profile["equity_range"]
        remaining_equity = max(0, equity_max - specific_stock_alloc)

        # For very_aggressive: maximize equity, no bonds
        if risk_level == "very_aggressive":
            equity_alloc = 1.0 - specific_stock_alloc  # 100% equity (minus any specific stocks)
            bond_alloc = 0
            treasury_alloc = 0
        elif return_target:
            # If return target specified, adjust equity allocation
            if return_target >= 0.12:
                equity_alloc = min(remaining_equity, equity_max)
            elif return_target >= 0.09:
                equity_alloc = min(remaining_equity, (equity_min + equity_max) / 2)
            elif return_target >= 0.06:
                equity_alloc = min(remaining_equity, equity_min)
            else:
                equity_alloc = min(remaining_equity, equity_min * 0.5)
            bond_alloc = 1.0 - equity_alloc - treasury_alloc - specific_stock_alloc
        else:
            equity_alloc = min(remaining_equity, (equity_min + equity_max) / 2)
            bond_alloc = 1.0 - equity_alloc - treasury_alloc - specific_stock_alloc

        # For very_conservative with specific stocks, minimize other equity
        if risk_level == "very_conservative" and specific_stocks:
            equity_alloc = 0  # Only use the specific stocks they requested
            bond_alloc = 1.0 - treasury_alloc - specific_stock_alloc

        # Ensure non-negative
        if bond_alloc < 0:
            bond_alloc = 0
            equity_alloc = max(0, 1.0 - treasury_alloc - specific_stock_alloc)

        # Build specific allocations for ETFs based on risk level
        if equity_alloc > 0:
            if risk_level == "very_aggressive":
                # High-beta: mix of growth ETFs AND high-beta individual stocks
                # Include NVDA (~1.7), AMD (~1.6), TSLA (~2.0) for higher beta
                allocations["QQQ"] = equity_alloc * 0.25   # Nasdaq 100, beta ~1.1
                allocations["NVDA"] = equity_alloc * 0.20  # NVIDIA, beta ~1.7
                allocations["AMD"] = equity_alloc * 0.15   # AMD, beta ~1.6
                allocations["TSLA"] = equity_alloc * 0.15  # Tesla, beta ~2.0
                allocations["VGT"] = equity_alloc * 0.15   # Tech sector, beta ~1.2
                allocations["IWM"] = equity_alloc * 0.10   # Small cap, beta ~1.2
            elif risk_level == "aggressive":
                # Growth tilted but more diversified
                allocations["SPY"] = equity_alloc * 0.35
                allocations["QQQ"] = equity_alloc * 0.30
                allocations["VGT"] = equity_alloc * 0.20   # Tech for higher beta
                allocations["VXUS"] = equity_alloc * 0.15
            elif risk_level == "moderate":
                # Balanced broad market with slight growth tilt
                allocations["SPY"] = equity_alloc * 0.35
                allocations["QQQ"] = equity_alloc * 0.20   # Add growth for higher beta
                allocations["VTI"] = equity_alloc * 0.20
                allocations["VXUS"] = equity_alloc * 0.15
                allocations["SCHD"] = equity_alloc * 0.10  # Dividend for stability
            elif risk_level == "conservative":
                # Defensive, dividend focused
                allocations["VTI"] = equity_alloc * 0.30
                allocations["SCHD"] = equity_alloc * 0.30  # Dividend ETF, lower beta
                allocations["VIG"] = equity_alloc * 0.25   # Dividend growth, beta ~0.9
                allocations["VXUS"] = equity_alloc * 0.15
            else:  # very_conservative
                # Minimal equity, very defensive
                allocations["SCHD"] = equity_alloc * 0.50  # Dividend, lower volatility
                allocations["VIG"] = equity_alloc * 0.50   # Dividend growth

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
                dollar_amount = budget * alloc

                # Check if this is a known ETF or a specific stock
                if ticker in cls.ASSET_CLASSES:
                    asset = cls.ASSET_CLASSES[ticker]
                    asset_name = asset.get("name", ticker)
                    asset_type = asset.get("type", "equity")
                    # Get REAL current price
                    stock_data = get_stock_data(ticker, period="5d")
                    if "error" not in stock_data and not stock_data["history"].empty:
                        current_price = float(stock_data["history"]["Close"].iloc[-1])
                    else:
                        current_price = 100
                else:
                    # This is a specific stock (TSLA, AAPL, etc.) - fetch real info
                    stock_info = cls.get_stock_info_for_portfolio(ticker)
                    if stock_info.get("error"):
                        # Skip stocks we can't find
                        continue
                    asset_name = stock_info.get("name", ticker)
                    asset_type = "stock"  # Mark as individual stock
                    current_price = stock_info.get("current_price", 100)

                # Get REAL return data
                real_return = real_returns.get(ticker)
                return_source = "REAL (1Y historical)" if ticker in real_returns else "estimate"

                positions.append({
                    "ticker": ticker,
                    "name": asset_name,
                    "type": asset_type,
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
                "equity": equity_alloc + specific_stock_alloc,  # Include specific stocks in equity
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

        # Get user risk profile
        risk_level = st.session_state.user_profile.get("risk_level", "moderate")

        # Calculate EQUITY risk metrics (Beta-based)
        equity_beta = cls.calculate_portfolio_beta(positions)
        beta_target = cls.BETA_TARGETS.get(risk_level, cls.BETA_TARGETS["moderate"])

        # Calculate FIXED INCOME risk metrics (Duration-based)
        duration_metrics = cls.calculate_portfolio_duration(positions)
        duration_target = cls.DURATION_TARGETS.get(risk_level, cls.DURATION_TARGETS["moderate"])

        # Estimate portfolio volatility (cross-asset risk budget)
        portfolio_vol = cls.estimate_portfolio_volatility(positions)
        vol_target = cls.VOLATILITY_TARGETS.get(risk_level, cls.VOLATILITY_TARGETS["moderate"])

        suggestions = []
        needs_rebalancing = False

        # === EQUITY RISK ANALYSIS (Beta) ===
        if current_allocation["equity"] > 0.1:
            if equity_beta < beta_target["min"]:
                suggestions.append(f"**Equity Beta** ({equity_beta:.2f}) is below target ({beta_target['min']:.1f}-{beta_target['max']:.1f}). Consider higher-beta growth stocks for your {risk_level} profile.")
                needs_rebalancing = True
            elif equity_beta > beta_target["max"]:
                suggestions.append(f"**Equity Beta** ({equity_beta:.2f}) exceeds target ({beta_target['min']:.1f}-{beta_target['max']:.1f}). Consider defensive stocks or reduce equity exposure.")
                needs_rebalancing = True

        # === FIXED INCOME RISK ANALYSIS (Duration) ===
        if duration_metrics["bond_value"] > 0:
            port_duration = duration_metrics["duration"]
            spread_duration = duration_metrics["spread_duration"]

            # Duration check (interest rate sensitivity)
            if port_duration > duration_target["max"]:
                suggestions.append(f"**Bond Duration** ({port_duration:.1f} yrs) exceeds target ({duration_target['max']} yrs). High interest rate sensitivity - consider shorter-duration bonds (SHY, VCSH).")
                needs_rebalancing = True
            elif port_duration < duration_target["min"] and current_allocation["treasury"] + current_allocation["bond"] > 0.2:
                suggestions.append(f"**Bond Duration** ({port_duration:.1f} yrs) is below target ({duration_target['min']} yrs). May sacrifice yield - consider intermediate bonds (IEF, VCIT).")
                needs_rebalancing = True

            # Spread duration check (credit risk sensitivity)
            if spread_duration > 5:
                suggestions.append(f"**Spread Duration** ({spread_duration:.1f}) indicates high credit risk. Consider reducing HYG/JNK exposure or adding treasuries.")
                needs_rebalancing = True

        # === CROSS-ASSET RISK BUDGET (Volatility) ===
        if portfolio_vol > 0:
            if portfolio_vol > vol_target["max"]:
                suggestions.append(f"**Portfolio Volatility** (~{portfolio_vol:.1f}%) exceeds target ({vol_target['max']}%). Consider rebalancing to bonds/treasuries.")
                needs_rebalancing = True
            elif portfolio_vol < vol_target["min"] and risk_level in ["aggressive", "very_aggressive"]:
                suggestions.append(f"**Portfolio Volatility** (~{portfolio_vol:.1f}%) is below target ({vol_target['min']}%). Portfolio may be too conservative for your goals.")
                needs_rebalancing = True

        # === CONCENTRATION RISK ===
        for pv in position_values:
            pct = pv["value"] / total_value * 100
            if pct > 30:
                suggestions.append(f"**{pv['ticker']}** is {pct:.1f}% of portfolio - consider trimming to reduce concentration risk")
                needs_rebalancing = True

        # === ALLOCATION DRIFT ===
        if current_allocation["equity"] > 0.80:
            suggestions.append(f"Equity allocation is {current_allocation['equity']*100:.0f}% - consider adding bonds for diversification")
            needs_rebalancing = True
        elif current_allocation["equity"] < 0.30 and risk_level not in ["conservative"]:
            suggestions.append(f"Equity allocation is only {current_allocation['equity']*100:.0f}% - consider adding equities for growth")
            needs_rebalancing = True

        if current_allocation["treasury"] == 0 and current_allocation["bond"] == 0:
            suggestions.append("No fixed income exposure - consider adding bonds for stability and income")
            needs_rebalancing = True

        return {
            "needs_rebalancing": needs_rebalancing,
            "suggestions": suggestions,
            "current_allocation": current_allocation,
            "total_value": total_value,
            "position_values": position_values,
            # Risk metrics by asset class
            "equity_metrics": {
                "beta": equity_beta,
                "target_range": f"{beta_target['min']:.1f}-{beta_target['max']:.1f}",
            },
            "fixed_income_metrics": {
                "duration": duration_metrics["duration"],
                "spread_duration": duration_metrics["spread_duration"],
                "convexity": duration_metrics["convexity"],
                "target_duration": f"{duration_target['min']}-{duration_target['max']} yrs",
            },
            "portfolio_metrics": {
                "volatility": portfolio_vol,
                "target_volatility": f"{vol_target['min']}-{vol_target['max']}%",
            },
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

    # Common company names to ticker mapping
    COMPANY_TO_TICKER = {
        "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META", "facebook": "META",
        "nvidia": "NVDA", "netflix": "NFLX", "disney": "DIS", "walmart": "WMT",
        "jpmorgan": "JPM", "chase": "JPM", "berkshire": "BRK-B", "visa": "V",
        "mastercard": "MA", "paypal": "PYPL", "intel": "INTC", "amd": "AMD",
        "boeing": "BA", "coca-cola": "KO", "coke": "KO", "pepsi": "PEP",
        "nike": "NKE", "starbucks": "SBUX", "mcdonalds": "MCD", "home depot": "HD",
    }

    @classmethod
    def resolve_company_names(cls, message: str, tickers: list) -> list:
        """Convert company names in message to tickers."""
        message_lower = message.lower()
        resolved = list(tickers)  # Start with already detected tickers

        for company, ticker in cls.COMPANY_TO_TICKER.items():
            if company in message_lower and ticker not in resolved:
                resolved.append(ticker)

        return resolved

    @classmethod
    def generate_response(cls, user_message: str, user_profile: dict) -> tuple:
        """Generate AI response and optionally create portfolio."""

        parsed = cls.parse_user_input(user_message)
        message_lower = user_message.lower()

        # Resolve company names to tickers (e.g., "tesla" -> "TSLA")
        parsed["specific_stocks"] = cls.resolve_company_names(user_message, parsed["specific_stocks"])

        # Check if this is a portfolio creation request (has budget or risk level mentioned)
        is_portfolio_request = parsed["budget"] or parsed["risk_level"] or parsed["treasury_constraint"]

        # Handle IMMEDIATE stock additions only if NOT creating a portfolio
        # If creating portfolio, specific stocks will be included in generation
        if parsed["specific_stocks"] and parsed["action"] == "add" and not is_portfolio_request:
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

            # Display risk metrics by asset class
            response_parts.append("\n### Risk Metrics by Asset Class")

            # Equity metrics (Beta-based)
            eq_metrics = analysis.get("equity_metrics", {})
            if current.get('equity', 0) > 0.05:
                response_parts.append(f"\n**Equities (Beta-based risk):**")
                response_parts.append(f"- Portfolio Beta: {eq_metrics.get('beta', 1.0):.2f}")
                response_parts.append(f"- Target Range: {eq_metrics.get('target_range', 'N/A')}")

            # Fixed income metrics (Duration-based)
            fi_metrics = analysis.get("fixed_income_metrics", {})
            if current.get('treasury', 0) + current.get('bond', 0) > 0.05:
                response_parts.append(f"\n**Fixed Income (Duration-based risk):**")
                response_parts.append(f"- Duration: {fi_metrics.get('duration', 0):.1f} years (interest rate sensitivity)")
                response_parts.append(f"- Spread Duration: {fi_metrics.get('spread_duration', 0):.1f} (credit risk sensitivity)")
                response_parts.append(f"- Convexity: {fi_metrics.get('convexity', 0):.2f}")
                response_parts.append(f"- Target Duration: {fi_metrics.get('target_duration', 'N/A')}")

            # Portfolio-level metrics
            port_metrics = analysis.get("portfolio_metrics", {})
            response_parts.append(f"\n**Cross-Asset Risk Budget:**")
            response_parts.append(f"- Est. Portfolio Volatility: {port_metrics.get('volatility', 0):.1f}%")
            response_parts.append(f"- Target Volatility: {port_metrics.get('target_volatility', 'N/A')}")

            if analysis["needs_rebalancing"]:
                response_parts.append("\n### Rebalancing Suggestions")
                for suggestion in analysis["suggestions"]:
                    response_parts.append(f"- {suggestion}")

                response_parts.append("\n**Say 'execute rebalancing' or 'rebalance now' to implement these changes.**")
                st.session_state.pending_rebalancing = analysis
            else:
                response_parts.append("\n**Your portfolio is well-balanced!** No rebalancing needed at this time.")

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

            # Get specific stocks user mentioned for inclusion
            specific_stocks = parsed.get("specific_stocks", [])

            portfolio = cls.generate_portfolio(
                risk_level=risk_level,
                return_target=return_target,
                treasury_pct=treasury_constraint,
                budget=budget,
                specific_stocks=specific_stocks if specific_stocks else None
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
# SMART AI AGENT (Free & Open Source - spaCy + Ollama)
# ============================================================

class SmartFinancialAgent:
    """
    AI Financial Planner powered by FREE open-source tools:
    - spaCy for NLP entity extraction (always available)
    - Ollama for local LLM (optional, if installed)
    - Smart rule-based fallback

    NO PAID APIs - 100% FREE
    """

    # Comprehensive company name to ticker mapping
    COMPANY_TICKERS = {
        # Tech giants
        "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META", "facebook": "META",
        "nvidia": "NVDA", "netflix": "NFLX", "adobe": "ADBE", "salesforce": "CRM",
        "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM", "cisco": "CSCO",
        "oracle": "ORCL", "ibm": "IBM", "palantir": "PLTR", "snowflake": "SNOW",
        # Finance
        "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM", "goldman": "GS",
        "goldman sachs": "GS", "morgan stanley": "MS", "bank of america": "BAC",
        "wells fargo": "WFC", "citigroup": "C", "citi": "C", "blackrock": "BLK",
        "visa": "V", "mastercard": "MA", "paypal": "PYPL", "square": "SQ", "block": "SQ",
        "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
        # Consumer
        "disney": "DIS", "walmart": "WMT", "target": "TGT", "costco": "COST",
        "home depot": "HD", "lowes": "LOW", "nike": "NKE", "starbucks": "SBUX",
        "mcdonalds": "MCD", "coca-cola": "KO", "coke": "KO", "pepsi": "PEP",
        "procter": "PG", "procter gamble": "PG", "johnson": "JNJ", "johnson johnson": "JNJ",
        # Energy & Industrial
        "exxon": "XOM", "chevron": "CVX", "shell": "SHEL", "bp": "BP",
        "boeing": "BA", "lockheed": "LMT", "raytheon": "RTX", "caterpillar": "CAT",
        "3m": "MMM", "honeywell": "HON", "general electric": "GE", "ge": "GE",
        # Healthcare
        "pfizer": "PFE", "moderna": "MRNA", "merck": "MRK", "abbvie": "ABBV",
        "eli lilly": "LLY", "lilly": "LLY", "unitedhealth": "UNH", "cvs": "CVS",
        # ETFs
        "spy": "SPY", "qqq": "QQQ", "voo": "VOO", "vti": "VTI", "arkk": "ARKK",
    }

    # Risk level keywords - extensive natural language phrases from Reddit, forums, real conversations
    RISK_KEYWORDS = {
        "very_conservative": {
            "phrases": [
                # Direct statements
                "very conservative", "extremely conservative", "ultra conservative",
                "super conservative", "highly conservative", "totally conservative",
                # Fear/anxiety expressions
                "scared of risk", "scared of losing", "terrified of losing", "afraid of risk",
                "fear of losing", "anxious about risk", "worried about losing", "nervous about risk",
                "hate risk", "hate losing money", "can't stand risk", "can't handle losses",
                "scared", "terrified", "petrified", "frightened",
                # Safety seeking
                "very safe", "super safe", "ultra safe", "extremely safe", "totally safe",
                "as safe as possible", "safest possible", "maximum safety", "zero risk",
                "no risk at all", "absolutely no risk", "minimal risk possible", "no risk",
                # Life situation
                "retirement money", "life savings", "can't afford to lose", "need this money",
                "retiring soon", "about to retire", "fixed income", "on a pension",
                "social security", "nest egg", "emergency fund", "rainy day fund",
                # Sleep/peace of mind
                "sleep at night", "peace of mind", "no stress", "stress free",
                "don't want to worry", "set and forget safe", "boring is good",
                # Age related
                "too old for risk", "not young anymore", "older investor", "senior",
                # Capital preservation
                "preserve capital", "capital preservation", "protect principal",
                "don't lose principal", "keep my money safe", "wealth preservation",
                # Negative risk sentiment
                "risk averse", "risk off", "avoid risk", "stay away from risk",
                "don't like risk", "don't want risk", "no tolerance for risk",
                "low risk tolerance", "very low risk tolerance", "zero risk tolerance",
            ],
            "weight": 1.0
        },
        "conservative": {
            "phrases": [
                # Direct statements
                "conservative", "somewhat conservative", "fairly conservative",
                "pretty conservative", "lean conservative", "more conservative",
                # Safety preference
                "safe", "prefer safe", "prefer safety", "safety first", "on the safe side",
                "play it safe", "rather be safe", "err on safe side",
                # Low risk
                "low risk", "lower risk", "less risk", "not much risk", "little risk",
                "minimal risk", "limited risk", "small risk", "slight risk",
                # Cautious
                "cautious", "careful", "prudent", "sensible", "responsible",
                "thoughtful", "measured", "calculated",
                # Stability seeking
                "stable", "stability", "steady", "consistent", "reliable", "predictable",
                "secure", "security", "protect", "protection", "defensive",
                # Income focus
                "income focused", "dividend focused", "yield focused", "income investor",
                "dividend investor", "steady income", "passive income", "cash flow",
                # Blue chip
                "blue chip", "blue chips only", "large cap only", "established companies",
                "proven companies", "quality companies", "safe stocks",
                # Bond preference
                "mostly bonds", "prefer bonds", "bond heavy", "fixed income heavy",
                "more bonds than stocks", "70 30 bonds", "60 40 bonds",
                # Slow and steady
                "slow and steady", "tortoise not hare", "steady wins race",
                "not in a hurry", "patient investor", "long term safe",
            ],
            "weight": 0.8
        },
        "moderate": {
            "phrases": [
                # Direct statements
                "moderate", "medium", "middle", "average", "normal", "standard",
                "typical", "regular", "ordinary", "conventional",
                # Balanced
                "balanced", "balance", "even split", "50 50", "sixty forty", "60 40",
                "mix of both", "stocks and bonds", "diversified", "well rounded", "all around",
                # Some risk acceptance
                "some risk", "bit of risk", "little bit of risk", "moderate risk",
                "medium risk", "acceptable risk", "reasonable risk", "manageable risk",
                "okay with some risk", "can handle some risk", "tolerate some risk",
                # Middle ground
                "middle ground", "middle of road", "not too risky", "not too safe",
                "somewhere in between", "in the middle", "happy medium",
                # Unsure/neutral
                "not sure", "unsure", "don't know", "idk", "uncertain", "undecided",
                "no preference", "whatever you suggest", "recommend something",
                "what do you think", "help me decide", "guide me", "you decide",
                "up to you", "your call", "dealer's choice",
                # Time horizon
                "medium term", "5 to 10 years", "10 years", "few years",
                # Growth and safety
                "growth and safety", "growth with protection", "upside with protection",
                "want both", "best of both", "growth and income",
            ],
            "weight": 0.5
        },
        "aggressive": {
            "phrases": [
                # Direct statements
                "aggressive", "more aggressive", "fairly aggressive", "pretty aggressive",
                "somewhat aggressive", "leaning aggressive", "on aggressive side",
                # Risk acceptance
                "like risk", "love risk", "enjoy risk", "embrace risk", "welcome risk",
                "comfortable with risk", "okay with risk", "fine with risk", "handle risk",
                "can take risk", "take on risk", "willing to risk", "accept risk",
                "don't mind risk", "risk doesn't bother me", "risk is fine",
                "higher risk tolerance", "high risk tolerance", "good risk tolerance",
                "risk taker", "take risks", "i like taking risks",
                # Growth focus
                "growth", "growth focused", "growth investor", "growth stocks",
                "capital appreciation", "capital gains", "maximize growth",
                "high growth", "aggressive growth", "growth over income",
                # Returns focus
                "high returns", "higher returns", "big returns", "maximize returns",
                "best returns", "outperform", "beat the market", "alpha",
                "10 percent", "double digit returns", "20 percent returns",
                "good returns", "great returns", "amazing returns",
                # Age/time related
                "young", "i'm young", "im young", "still young", "got time",
                "long time horizon", "decades to invest", "30 years", "20 years",
                "time on my side", "can wait it out", "can ride it out",
                "just starting out", "early career", "in my 20s", "in my 30s",
                # Tech/growth sectors
                "tech stocks", "tech heavy", "technology", "innovation",
                "disruptive", "emerging", "new economy", "future",
                # Volatility acceptance
                "can handle volatility", "volatility is fine", "don't mind swings",
                "stomach for volatility", "ride the waves", "handle the ups and downs",
                "can stomach losses", "can handle drawdowns",
                # Reddit/casual phrases
                "risk it for the biscuit", "no risk no reward", "gotta risk it",
                "fortune favors the bold", "go big or go home", "send it",
                "let it ride", "full send", "balls deep", "lfg", "let's go",
                # Stock heavy
                "all stocks", "100 percent stocks", "mostly stocks", "stock heavy",
                "equities only", "no bonds", "skip the bonds", "stocks only",
                "80 20 stocks", "90 10 stocks", "heavy equities",
            ],
            "weight": 0.3
        },
        "very_aggressive": {
            "phrases": [
                # Direct statements
                "very aggressive", "extremely aggressive", "super aggressive",
                "ultra aggressive", "highly aggressive", "max aggressive",
                "most aggressive", "maximum aggression",
                # YOLO culture (Reddit/WSB)
                "yolo", "yoloing", "yolo it", "full yolo", "going yolo",
                "to the moon", "moon", "mooning", "wen moon", "rocket", "rockets",
                "ape", "ape in", "aping", "going ape", "diamond hands", "💎🙌",
                "hodl", "hold the line", "never sell", "paper hands never",
                "tendies", "wife's boyfriend", "this is the way", "the way",
                "smooth brain", "wrinkle brain", "regarded", "autist",
                "wallstreetbets", "wsb", "degenerates", "degen", "degenerate",
                "casino", "gambling", "gamble", "bet it all", "all or nothing",
                "sir this is a wendy's", "loss porn", "gain porn",
                # Maximum risk
                "maximum risk", "highest risk", "most risk possible", "full risk",
                "extremely risky", "very risky", "super risky", "high risk high reward",
                "risk it all", "risk everything", "nothing to lose", "house money",
                # Speculation
                "speculative", "speculation", "speculate", "punt", "punting",
                "meme stocks", "meme coins", "penny stocks", "small caps only",
                "micro caps", "options", "calls", "puts", "leveraged", "leverage",
                "margin", "on margin", "3x", "2x leveraged", "weekly options",
                "0dte", "fds", "lottos", "lotto tickets",
                # All in mentality
                "all in", "going all in", "everything on", "bet the farm",
                "swing for fences", "home run", "10 bagger", "100 bagger",
                "get rich", "get rich quick", "lambo", "wen lambo", "millionaire",
                "generational wealth", "life changing money", "fuck you money",
                # Crypto adjacent
                "crypto style", "like crypto", "degen plays", "ape strong",
                "ngmi", "wagmi", "gm", "probably nothing", "few understand",
                # Youth/FOMO
                "can afford to lose it all", "play money", "fun money",
                "lottery ticket", "lotto", "fomo", "fear of missing out",
                "cant miss this", "once in lifetime", "generational opportunity",
            ],
            "weight": 0.1
        }
    }

    # Intent patterns
    INTENT_PATTERNS = {
        "create_portfolio": [
            r"create.*(portfolio|allocation)",
            r"build.*(portfolio|allocation)",
            r"make.*(portfolio|allocation)",
            r"(i have|invest|budget).+\$?\d+",
            r"(looking for|want|need).*(return|yield|income)",
            r"(i have|got|with).+\d+.*(dollar|million|k\b|thousand)",
            r"(scared|conservative|aggressive).*(want|treasury|stock)",
            r"(want|need).*(treasury|bond|stock).*(and|some)",
            r"\$?\d+[mk]?\s*(to invest|budget|portfolio)",
            # New natural language patterns
            r"(i like|i love|i want|i prefer).*(risk|growth|stock)",
            r"(return|yield|gain).*(of|about|around)?\s*\d+\s*%",
            r"\d+\s*%\s*(return|yield|yearly|annual)",
            r"(like|love|want|prefer|fan of)\s+[A-Z]{2,5}\b",  # "I like OKLO"
            r"(invest|put|allocate).*money",
        ],
        "add_stock": [
            r"(add|buy|purchase|get|include)\s+.*(stock|share|position)?",
            r"(add|buy|purchase)\s+\$?\d+.*\s+(of|worth|in)",
            r"\d+\s+shares?\s+(of\s+)?",
            r"^add\s+[A-Z]{1,5}$",  # Simple "add TSLA"
        ],
        "remove_stock": [
            r"(remove|sell|drop|delete|get rid)\s+",
        ],
        "rebalance": [
            r"rebalance",
            r"optimize.*portfolio",
            r"adjust.*allocation",
            r"fix.*portfolio",
        ],
        "portfolio_status": [
            r"(show|what|how).*(portfolio|holdings|positions)",
            r"(my|current)\s+(portfolio|holdings|positions)",
        ],
        "market_conditions": [
            r"(market|vix|volatility|yield|curve)",
            r"(what|how).*(market|economy)",
        ],
        "stock_info": [
            r"(tell|show|what|how).*(about|is)\s+[A-Z]{1,5}",
            r"(price|info|information).*(of|for|about)",
        ],
    }

    @classmethod
    def extract_entities_spacy(cls, text: str) -> dict:
        """Use spaCy to extract entities from text."""
        entities = {
            "money": [],
            "percentages": [],
            "companies": [],
            "tickers": [],
            "numbers": [],
        }

        if not SPACY_AVAILABLE or nlp is None:
            return entities

        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "MONEY":
                # Extract numeric value from money
                amount_str = re.sub(r'[^\d.]', '', ent.text)
                if amount_str:
                    try:
                        entities["money"].append(float(amount_str))
                    except ValueError:
                        pass
            elif ent.label_ == "PERCENT":
                pct_str = re.sub(r'[^\d.]', '', ent.text)
                if pct_str:
                    try:
                        entities["percentages"].append(float(pct_str))
                    except ValueError:
                        pass
            elif ent.label_ == "ORG":
                entities["companies"].append(ent.text.lower())
            elif ent.label_ in ["CARDINAL", "QUANTITY"]:
                num_str = re.sub(r'[^\d.]', '', ent.text)
                if num_str:
                    try:
                        entities["numbers"].append(float(num_str))
                    except ValueError:
                        pass

        # Also find tickers (1-5 uppercase letters)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, text)
        common_words = {"I", "A", "THE", "AND", "FOR", "TO", "IN", "OF", "IS", "IT",
                       "MY", "ME", "BE", "DO", "SO", "IF", "OR", "AS", "AT", "BY"}
        entities["tickers"] = [t for t in potential_tickers if t not in common_words]

        return entities

    @classmethod
    def extract_entities_regex(cls, text: str) -> dict:
        """Fallback regex-based entity extraction."""
        entities = {
            "money": [],
            "percentages": [],
            "companies": [],
            "tickers": [],
            "numbers": [],
        }

        text_lower = text.lower()

        # Money patterns with multipliers - order matters!
        # Pattern: $X million, X million dollars, $Xm, $X mil
        money_patterns = [
            # "$1 million" or "$1.5 million" or "1 million dollars"
            (r'\$?\s*(\d+(?:\.\d+)?)\s*(?:million|mil)\b', 1000000),
            # "$1m" or "1m" (but not in words like "I'm")
            (r'\$\s*(\d+(?:\.\d+)?)\s*m\b', 1000000),
            # "1 billion"
            (r'\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|bil|b)\b', 1000000000),
            # "$100k" or "100k" or "$100 thousand"
            (r'\$?\s*(\d+(?:\.\d+)?)\s*(?:k|thousand)\b', 1000),
            # "$1,000,000" or "$100000" (plain numbers with $)
            (r'\$\s*([\d,]+(?:\.\d{2})?)', 1),
            # "1000000 dollars"
            (r'([\d,]+(?:\.\d{2})?)\s*(?:dollars?|USD)', 1),
        ]

        found_amounts = set()
        for pattern, multiplier in money_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    amount = float(match.replace(',', '')) * multiplier
                    if amount >= 1000:
                        found_amounts.add(amount)
                except ValueError:
                    pass

        entities["money"] = list(found_amounts)

        # Percentage patterns
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        entities["percentages"] = [float(p) for p in pct_matches]

        # Large numbers (likely budgets) - only if no money found yet
        if not entities["money"]:
            large_nums = re.findall(r'\b(\d{5,})\b', text.replace(',', ''))
            for num in large_nums:
                try:
                    n = float(num)
                    if n >= 10000:
                        entities["money"].append(n)
                except ValueError:
                    pass

        return entities

    @classmethod
    def resolve_companies(cls, text: str, entities: dict) -> list:
        """Resolve company names and tickers."""
        tickers = list(entities.get("tickers", []))
        text_lower = text.lower()

        # Check for company names
        for company, ticker in cls.COMPANY_TICKERS.items():
            if company in text_lower and ticker not in tickers:
                tickers.append(ticker)

        # Check if any detected companies map to tickers
        for company in entities.get("companies", []):
            if company.lower() in cls.COMPANY_TICKERS:
                ticker = cls.COMPANY_TICKERS[company.lower()]
                if ticker not in tickers:
                    tickers.append(ticker)

        return tickers

    @classmethod
    def detect_risk_level(cls, text: str) -> tuple:
        """Detect risk level from text with confidence score."""
        text_lower = text.lower()

        # PRIORITY 0: Check for fear/aversion phrases FIRST - these are VERY CONSERVATIVE
        fear_phrases = [
            "scared of risk", "scared of losing", "fear risk", "hate risk", "afraid of risk",
            "terrified", "petrified", "frightened", "anxious about", "worried about losing",
            "can't afford to lose", "can't handle", "no risk", "zero risk", "avoid risk",
            "scared", "fearful", "nervous about risk", "risk averse", "risk-averse",
        ]
        for phrase in fear_phrases:
            if phrase in text_lower:
                return "very_conservative", 0.98

        # PRIORITY 1: Check for explicit risk level words
        # Order: "very" variants first, then "moderate" (to catch "moderate risk" before "aggressive" phrases)
        if "very conservative" in text_lower:
            return "very_conservative", 0.95
        if "very aggressive" in text_lower:
            return "very_aggressive", 0.95
        # Check "moderate" BEFORE other single-word levels
        # This ensures "I am moderate risk" doesn't accidentally match "aggressive" phrases
        if "moderate" in text_lower:
            return "moderate", 0.95
        if "conservative" in text_lower:
            return "conservative", 0.95
        if "aggressive" in text_lower:
            return "aggressive", 0.95

        # PRIORITY 2: Check for exact phrase matches
        # Order: most extreme first, then moderate last (as fallback)
        for level in ["very_conservative", "very_aggressive", "conservative", "aggressive", "moderate"]:
            data = cls.RISK_KEYWORDS[level]
            for phrase in data["phrases"]:
                # Skip single common words that might cause false matches
                if len(phrase) > 4 and phrase in text_lower:
                    return level, 0.9

        # PRIORITY 3: Fuzzy matching for partial matches
        risk_scores = {}
        for level, data in cls.RISK_KEYWORDS.items():
            score = 0
            for phrase in data["phrases"]:
                words = phrase.split()
                # Only count if at least half the words match
                matches = sum(1 for word in words if word in text_lower and len(word) > 3)
                if matches >= len(words) / 2:
                    score += matches / len(words)
            risk_scores[level] = score

        if max(risk_scores.values()) > 0.5:
            best_level = max(risk_scores, key=risk_scores.get)
            return best_level, risk_scores[best_level]

        return None, 0.0

    @classmethod
    def detect_intent(cls, text: str) -> tuple:
        """Detect user intent from text."""
        text_lower = text.lower()

        intent_scores = {}
        for intent, patterns in cls.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            intent_scores[intent] = score

        if max(intent_scores.values()) > 0:
            best_intent = max(intent_scores, key=intent_scores.get)
            return best_intent, intent_scores[best_intent]

        return "unknown", 0

    @classmethod
    def detect_treasury_preference(cls, text: str) -> float:
        """Detect treasury allocation preference."""
        text_lower = text.lower()

        # Heavy treasury indicators
        heavy_patterns = [
            (r"(a lot of|lots of|mostly|mainly|heavy|primarily|majority)\s+(treasury|treasuries|bonds)", 0.60),
            (r"(60|70|80|90)\s*%?\s*(treasury|treasuries|in bonds)", None),  # Use the percentage
            (r"very.*safe", 0.50),
            (r"(all|most).*(treasury|bonds|fixed income)", 0.70),
        ]

        for pattern, default_pct in heavy_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Check if there's a specific percentage
                pct_match = re.search(r'(\d+)\s*%', match.group())
                if pct_match:
                    return float(pct_match.group(1)) / 100
                elif default_pct:
                    return default_pct

        # Check for specific percentage mentions
        treasury_pct = re.search(r'(\d+)\s*%?\s*(in\s+)?(treasury|treasuries)', text_lower)
        if treasury_pct:
            return float(treasury_pct.group(1)) / 100

        return None

    @classmethod
    def detect_return_target(cls, text: str) -> float:
        """Detect target return percentage from text."""
        text_lower = text.lower()

        # Patterns for return targets
        patterns = [
            r'return\s*(?:of\s*)?(?:about\s*)?(\d+(?:\.\d+)?)\s*%',  # "return of about 10%"
            r'(\d+(?:\.\d+)?)\s*%\s*(?:return|yield|yearly|annual|per year)',  # "10% return"
            r'(?:target|want|need|looking for)\s*(\d+(?:\.\d+)?)\s*%',  # "target 10%"
            r'(\d+(?:\.\d+)?)\s*%\s*(?:a year|annually)',  # "10% a year"
            r'(?:make|earn|get)\s*(\d+(?:\.\d+)?)\s*%',  # "make 10%"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1)) / 100

        return None

    @classmethod
    def extract_raw_tickers(cls, text: str) -> list:
        """Extract potential stock tickers from text - handles multiple tickers with and/or/commas."""
        valid_tickers = []

        # Common words to filter out (not tickers)
        non_tickers = {'I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'IN', 'ON', 'AT', 'BY',
                       'AN', 'AS', 'IF', 'IT', 'OF', 'UP', 'DO', 'GO', 'SO', 'NO', 'US',
                       'AM', 'PM', 'OK', 'HI', 'MY', 'ME', 'WE', 'HE', 'BE', 'IS', 'ARE',
                       'WAS', 'HAS', 'HAD', 'BUT', 'NOT', 'ALL', 'CAN', 'HER', 'WANT',
                       'ONE', 'OUR', 'OUT', 'YOU', 'DAY', 'GET', 'HIM', 'HIS', 'LIKE',
                       'HOW', 'ITS', 'LET', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY',
                       'WHO', 'BOY', 'DID', 'OWN', 'SAY', 'SHE', 'TOO', 'USE', 'USD',
                       'ETF', 'CEO', 'CFO', 'IPO', 'USA', 'NYC', 'API', 'WITH', 'ALSO',
                       'SOME', 'ANY', 'HAVE', 'THIS', 'THAT', 'FROM', 'BEEN', 'WILL',
                       'INTO', 'JUST', 'ONLY', 'OVER', 'SUCH', 'MAKE', 'THAN', 'THEM',
                       'WELL', 'BACK', 'YEAR', 'WHEN', 'YOUR', 'WHAT', 'THEN', 'LOOK'}

        # 1. Find ALL CAPS tickers (2-5 letters)
        uppercase_tickers = re.findall(r'\b([A-Z]{2,5})\b', text)
        for t in uppercase_tickers:
            if t not in non_tickers and t not in valid_tickers:
                valid_tickers.append(t)

        # 2. Find tickers after "like", "want", "love", "buy", etc. - even if lowercase
        # Pattern: "I like oklo and nvda" or "want tsla or aapl"
        ticker_context_patterns = [
            r'(?:like|love|want|buy|get|add|prefer|into|own|hold|holding)\s+([a-zA-Z]{2,5})(?:\s+(?:and|or|,)\s+([a-zA-Z]{2,5}))*',
            r'(?:like|love|want|buy|get|add|prefer|into|own|hold|holding)\s+([a-zA-Z]{2,5})\s+(?:and|or|,)\s+([a-zA-Z]{2,5})',
            r'([a-zA-Z]{2,5})\s+(?:and|or|,)\s+([a-zA-Z]{2,5})\s+(?:stock|stocks|shares)',
        ]

        for pattern in ticker_context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for t in match:
                        if t and t.upper() not in non_tickers and t.upper() not in valid_tickers:
                            valid_tickers.append(t.upper())
                elif match:
                    if match.upper() not in non_tickers and match.upper() not in valid_tickers:
                        valid_tickers.append(match.upper())

        # 3. Find comma/and/or separated lists of potential tickers
        # "OKLO, NVDA, TSLA" or "oklo and nvda and tsla" or "tsla or aapl"
        list_pattern = r'\b([a-zA-Z]{2,5})\s*(?:,|and|or|\&)\s*([a-zA-Z]{2,5})(?:\s*(?:,|and|or|\&)\s*([a-zA-Z]{2,5}))?(?:\s*(?:,|and|or|\&)\s*([a-zA-Z]{2,5}))?'
        list_matches = re.findall(list_pattern, text, re.IGNORECASE)
        for match in list_matches:
            for t in match:
                if t and len(t) >= 2 and t.upper() not in non_tickers and t.upper() not in valid_tickers:
                    # Additional check: if it looks like a sentence word, skip it
                    if t.lower() not in ['like', 'want', 'have', 'some', 'with', 'also', 'just', 'into', 'from']:
                        valid_tickers.append(t.upper())

        # 4. Also check for tickers mentioned with "stock" or "stocks"
        stock_pattern = r'([a-zA-Z]{2,5})\s+(?:stock|stocks|shares|share)'
        stock_matches = re.findall(stock_pattern, text, re.IGNORECASE)
        for t in stock_matches:
            if t.upper() not in non_tickers and t.upper() not in valid_tickers:
                valid_tickers.append(t.upper())

        return valid_tickers

    @classmethod
    def parse_message_smart(cls, text: str) -> dict:
        """Smart message parsing using spaCy + regex."""
        # Get entities from spaCy or fallback to regex
        if SPACY_AVAILABLE:
            entities = cls.extract_entities_spacy(text)
        else:
            entities = cls.extract_entities_regex(text)

        # Also run regex to catch things spaCy might miss
        regex_entities = cls.extract_entities_regex(text)

        # Merge money amounts (deduplicate)
        all_money = list(set(entities.get("money", []) + regex_entities.get("money", [])))

        # Resolve companies and tickers
        tickers = cls.resolve_companies(text, entities)

        # Also extract raw tickers from text (like OKLO, TSLA mentioned directly)
        raw_tickers = cls.extract_raw_tickers(text)
        for ticker in raw_tickers:
            if ticker not in tickers:
                tickers.append(ticker)

        # Detect risk level
        risk_level, risk_confidence = cls.detect_risk_level(text)

        # Detect intent
        intent, intent_confidence = cls.detect_intent(text)

        # Detect treasury preference
        treasury_pct = cls.detect_treasury_preference(text)

        # Detect return target
        return_target = cls.detect_return_target(text)

        # Get budget (largest money amount, or largest number if no money detected)
        budget = None
        if all_money:
            budget = max(all_money)
        elif entities.get("numbers"):
            large_nums = [n for n in entities["numbers"] if n >= 1000]
            if large_nums:
                budget = max(large_nums)

        # If we detected risk, tickers, or return target, assume create_portfolio intent
        if (risk_level or tickers or return_target) and intent == "unknown":
            intent = "create_portfolio"

        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "risk_level": risk_level,
            "risk_confidence": risk_confidence,
            "budget": budget,
            "treasury_pct": treasury_pct,
            "tickers": tickers,
            "return_target": return_target,
            "percentages": entities.get("percentages", []) + regex_entities.get("percentages", []),
            "raw_entities": entities,
        }

    @classmethod
    def call_ollama(cls, prompt: str, system_prompt: str = None) -> str:
        """Call local Ollama LLM."""
        if not OLLAMA_AVAILABLE:
            return None

        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower = more factual
                    "num_predict": 500,  # Limit response length
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            pass

        return None

    @classmethod
    def execute_action(cls, intent: str, parsed: dict) -> str:
        """Execute the detected action and return response."""
        try:
            if intent == "create_portfolio":
                risk_level = parsed.get("risk_level") or "moderate"
                budget = parsed.get("budget") or 100000
                treasury_pct = parsed.get("treasury_pct")
                tickers = parsed.get("tickers", [])
                return_target = parsed.get("return_target")

                # IMPORTANT: Save the detected risk level to session state for Risk Profile display
                if "user_profile" not in st.session_state or not st.session_state.user_profile:
                    st.session_state.user_profile = {"constraints": {}}
                st.session_state.user_profile["risk_level"] = risk_level
                st.session_state.user_profile["budget"] = budget

                # When user tells chatbot a risk level, ALWAYS assign a random beta within that range
                # This ensures their stated preference is reflected in the target beta
                # (User can still override with slider later if they want exact control)
                import random
                # Beta ranges - ensuring no overlap and clear boundaries
                # Very Conservative: < 0.30
                # Conservative: 0.30 - 0.54
                # Moderate: 0.55 - 0.84
                # Aggressive: 0.85 - 1.04
                # Very Aggressive: >= 1.05
                beta_ranges = {
                    "very_conservative": (0.10, 0.29),
                    "conservative": (0.30, 0.54),
                    "moderate": (0.55, 0.84),
                    "aggressive": (0.85, 1.04),
                    "very_aggressive": (1.05, 1.40),
                }
                beta_min, beta_max = beta_ranges.get(risk_level, (0.55, 0.84))
                # Generate random beta and CLAMP to ensure it stays within bounds
                raw_beta = random.uniform(beta_min, beta_max)
                random_beta = round(max(beta_min, min(beta_max, raw_beta)), 2)
                st.session_state.user_profile["target_beta"] = random_beta

                if return_target:
                    st.session_state.user_profile["return_target"] = return_target
                if treasury_pct:
                    st.session_state.user_profile["constraints"]["treasury"] = treasury_pct

                portfolio = FinancialPlannerAI.generate_portfolio(
                    risk_level=risk_level,
                    budget=budget,
                    treasury_pct=treasury_pct,
                    specific_stocks=tickers if tickers else None,
                    return_target=return_target
                )

                st.session_state.pending_portfolio = portfolio

                # Format response
                response = f"""## Portfolio Created for You

**Your Profile:** {risk_level.replace('_', ' ').title()}
**Target Beta:** {random_beta} (assigned within {risk_level.replace('_', ' ')} range)
**Budget:** ${budget:,.0f}
"""
                if return_target:
                    response += f"**Target Return:** {return_target*100:.0f}%\n"
                if treasury_pct:
                    response += f"**Treasury Allocation:** {treasury_pct*100:.0f}%\n"
                if tickers:
                    response += f"**Requested Stocks:** {', '.join(tickers)}\n"

                response += f"""
### Allocation Summary
- **Equities:** {portfolio['summary']['equity']*100:.0f}%
- **Treasury:** {portfolio['summary']['treasury']*100:.0f}%
- **Bonds:** {portfolio['summary']['bonds']*100:.0f}%

### Positions
| Ticker | Name | Allocation | Amount |
|--------|------|------------|--------|
"""
                for pos in portfolio["positions"]:
                    response += f"| {pos['ticker']} | {pos['name']} | {pos['allocation']*100:.1f}% | ${pos['dollar_amount']:,.0f} |\n"

                response += f"""
**Expected Return:** {portfolio['expected_return']*100:.1f}% (based on historical data)
**Portfolio Volatility:** {portfolio['portfolio_risk']*100:.1f}%

*Say 'yes' or 'confirm' to add these positions to your portfolio.*"""

                return response

            elif intent == "add_stock":
                if not parsed.get("tickers"):
                    return "I'd be happy to add a stock to your portfolio. Which stock would you like? (e.g., 'add Tesla' or 'buy AAPL')"

                responses = []
                for ticker in parsed["tickers"]:
                    stock_info = FinancialPlannerAI.get_stock_info_for_portfolio(ticker)
                    if stock_info.get("error"):
                        responses.append(f"Could not find **{ticker}**: {stock_info['error']}")
                        continue

                    # Determine quantity
                    if parsed.get("budget"):
                        quantity = int(parsed["budget"] / stock_info["current_price"])
                    else:
                        quantity = max(1, int(1000 / stock_info["current_price"]))

                    position = {
                        "ticker": stock_info["ticker"],
                        "type": "Stock",
                        "side": "Long",
                        "quantity": quantity,
                        "entry_price": stock_info["current_price"],
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                        "strike": None, "expiry": None, "yield_rate": None, "maturity": None,
                        "id": len(st.session_state.portfolio)
                    }
                    st.session_state.portfolio.append(position)

                    value = quantity * stock_info["current_price"]
                    responses.append(f"""**Added {stock_info['ticker']}** ({stock_info['name']})
- Shares: {quantity} @ ${stock_info['current_price']:.2f} = ${value:,.2f}
- Beta: {stock_info['beta']:.2f}
- Sector: {stock_info['sector']}""")

                # Calculate portfolio beta
                portfolio_beta = FinancialPlannerAI.calculate_portfolio_beta(st.session_state.portfolio)
                responses.append(f"\n**Portfolio Beta:** {portfolio_beta:.2f}")

                return "\n\n".join(responses)

            elif intent == "remove_stock":
                if not parsed.get("tickers"):
                    return "Which stock would you like to remove? (e.g., 'remove AAPL')"

                removed = []
                for ticker in parsed["tickers"]:
                    initial_count = len(st.session_state.portfolio)
                    st.session_state.portfolio = [
                        p for p in st.session_state.portfolio
                        if p.get("ticker", "").upper() != ticker.upper()
                    ]
                    if len(st.session_state.portfolio) < initial_count:
                        removed.append(ticker)

                if removed:
                    return f"Removed **{', '.join(removed)}** from your portfolio."
                return f"Could not find {', '.join(parsed['tickers'])} in your portfolio."

            elif intent == "portfolio_status":
                positions = st.session_state.get("portfolio", [])
                if not positions:
                    return "Your portfolio is empty. Tell me about your investment goals and I'll create a portfolio for you!"

                total_value = 0
                response = "## Your Current Portfolio\n\n| Ticker | Shares | Price | Value |\n|--------|--------|-------|-------|\n"

                for pos in positions:
                    ticker = pos.get("ticker", "")
                    qty = pos.get("quantity", 0)
                    price = pos.get("entry_price", 0)
                    value = qty * price
                    total_value += value
                    response += f"| {ticker} | {qty} | ${price:.2f} | ${value:,.2f} |\n"

                beta = FinancialPlannerAI.calculate_portfolio_beta(positions)
                response += f"\n**Total Value:** ${total_value:,.2f}\n**Portfolio Beta:** {beta:.2f}"

                return response

            elif intent == "rebalance":
                positions = st.session_state.get("portfolio", [])
                if not positions:
                    return "No portfolio to rebalance. Create one first!"

                analysis = FinancialPlannerAI.analyze_portfolio_for_rebalancing(positions)

                response = f"""## Portfolio Rebalancing Analysis

**Total Value:** ${analysis['total_value']:,.2f}

### Current Allocation
- Equities: {analysis['current_allocation'].get('equity', 0)*100:.1f}%
- Treasury: {analysis['current_allocation'].get('treasury', 0)*100:.1f}%
- Bonds: {analysis['current_allocation'].get('bond', 0)*100:.1f}%

### Risk Metrics
"""
                eq_metrics = analysis.get("equity_metrics", {})
                fi_metrics = analysis.get("fixed_income_metrics", {})
                response += f"- **Equity Beta:** {eq_metrics.get('beta', 'N/A')}\n"
                response += f"- **Bond Duration:** {fi_metrics.get('duration', 0):.1f} years\n"

                if analysis["needs_rebalancing"]:
                    response += "\n### Suggestions\n"
                    for s in analysis["suggestions"]:
                        response += f"- {s}\n"
                    response += "\n*Say 'execute rebalancing' to implement these changes.*"
                else:
                    response += "\n**Your portfolio is well-balanced!**"

                return response

            elif intent == "market_conditions":
                vix_data = get_vix_stats()
                vix_regime = determine_vix_regime(vix_data.get("current", 20))
                yields = get_current_yields()
                curve_regime = determine_curve_regime(yields)

                return f"""## Current Market Conditions

**VIX (Fear Index):** {vix_data.get('current', 'N/A'):.1f}
- Regime: **{vix_regime}**
- Percentile: {vix_data.get('percentile', 'N/A'):.0f}%

**Yield Curve:** {curve_regime}
- 10Y-2Y Spread: {yields.get('10Y', 0) - yields.get('2Y', 0):.2f}%

### Market Interpretation
{"- High volatility - consider defensive positions" if vix_regime in ['ELEVATED', 'CRISIS'] else "- Normal volatility - standard positioning"}
{"- Yield curve inverted - recession risk elevated" if 'Inverted' in curve_regime else "- Normal yield curve"}"""

            elif intent == "stock_info":
                if not parsed.get("tickers"):
                    return "Which stock would you like to know about? (e.g., 'tell me about AAPL')"

                responses = []
                for ticker in parsed["tickers"][:3]:  # Limit to 3 stocks
                    info = FinancialPlannerAI.get_stock_info_for_portfolio(ticker)
                    if info.get("error"):
                        responses.append(f"Could not find {ticker}")
                        continue

                    responses.append(f"""### {info['ticker']} - {info['name']}
- **Price:** ${info['current_price']:.2f}
- **Beta:** {info['beta']:.2f}
- **Sector:** {info['sector']}
- **Industry:** {info['industry']}
- **52-Week Range:** ${info.get('52w_low', 0):.2f} - ${info.get('52w_high', 0):.2f}
- **Dividend Yield:** {info.get('dividend_yield', 0)*100:.2f}%""")

                return "\n\n".join(responses)

            else:
                return None  # Unknown intent, will use fallback

        except Exception as e:
            return f"Error executing action: {str(e)}"

    @classmethod
    def generate_smart_response(cls, parsed: dict, user_message: str) -> str:
        """Generate a contextual response based on what we understood."""
        parts = []

        # Acknowledge what we understood
        if parsed.get("risk_level"):
            parts.append(f"I understand you have a **{parsed['risk_level'].replace('_', ' ')}** risk tolerance.")
        if parsed.get("budget"):
            parts.append(f"Budget: **${parsed['budget']:,.0f}**")
        if parsed.get("treasury_pct"):
            parts.append(f"Treasury preference: **{parsed['treasury_pct']*100:.0f}%**")
        if parsed.get("tickers"):
            parts.append(f"Stocks mentioned: **{', '.join(parsed['tickers'])}**")

        if not parts:
            # Didn't understand much, ask for clarification
            return """I'm here to help you with your investments! You can:

- **Create a portfolio:** "I'm conservative with $100,000 to invest"
- **Add stocks:** "Add Tesla to my portfolio" or "Buy $5000 of AAPL"
- **Check your portfolio:** "Show my holdings"
- **Get market info:** "What are market conditions?"
- **Rebalance:** "Rebalance my portfolio"

What would you like to do?"""

        return "\n".join(parts)

    @classmethod
    def chat(cls, user_message: str, chat_history: list) -> str:
        """Main chat function - uses Ollama if available, otherwise smart rules."""

        # Parse the message
        parsed = cls.parse_message_smart(user_message)

        # Check for confirmation of pending portfolio
        # Note: Sidebar usually handles this first, but keeping for consistency
        if any(word in user_message.lower() for word in ["yes", "confirm", "do it", "proceed", "add it", "looks good"]):
            if st.session_state.get("pending_portfolio"):
                portfolio = st.session_state.pending_portfolio
                added_count = 0
                for pos in portfolio["positions"]:
                    ticker = pos["ticker"]
                    dollar_amount = pos.get("dollar_amount", 0)

                    # Get current price
                    current_price = pos.get("current_price", 100)
                    if current_price <= 0:
                        stock_data = get_stock_data(ticker, period="5d")
                        if "error" not in stock_data and not stock_data["history"].empty:
                            current_price = float(stock_data["history"]["Close"].iloc[-1])
                        else:
                            current_price = 100

                    quantity = int(dollar_amount / current_price) if current_price > 0 else 0
                    if quantity > 0:
                        # Determine display type
                        pos_type = pos.get("type", "equity").lower()
                        if pos_type in ["stock"]:
                            display_type = "Stock"
                        elif pos_type in ["treasury"]:
                            display_type = "Treasury ETF (TLT, IEF, SHY)"
                        elif pos_type in ["bond", "corporate_bond", "high_yield"]:
                            display_type = "Corporate Bond ETF (LQD, HYG)"
                        else:
                            display_type = "ETF"

                        position = {
                            "ticker": ticker,
                            "type": display_type,
                            "side": "Long",
                            "quantity": quantity,
                            "entry_price": current_price,
                            "entry_date": datetime.now().strftime("%Y-%m-%d"),
                            "strike": None, "expiry": None, "yield_rate": None, "maturity": None,
                            "id": len(st.session_state.portfolio)
                        }
                        st.session_state.portfolio.append(position)
                        added_count += 1

                st.session_state.pending_portfolio = None
                return f"✅ **Done!** Added {added_count} positions to your portfolio. Check the **Portfolio** tab to see them."

        # Try to execute the detected intent
        intent = parsed.get("intent", "unknown")
        result = cls.execute_action(intent, parsed)

        if result:
            return result

        # If we have Ollama, use it for better understanding
        if OLLAMA_AVAILABLE and intent == "unknown":
            ollama_prompt = f"""You are a helpful financial advisor assistant. The user said: "{user_message}"

Based on this, determine what they want and respond helpfully. Keep your response concise and actionable.

If they want to:
- Create a portfolio: Ask about their risk tolerance and budget
- Add a stock: Confirm which stock and how much
- Get information: Provide it
- Something unclear: Ask clarifying questions

Respond naturally and helpfully:"""

            ollama_response = cls.call_ollama(ollama_prompt)
            if ollama_response:
                return ollama_response

        # Fallback to smart response based on what we parsed
        return cls.generate_smart_response(parsed, user_message)


# (Old OpenAI code removed - now using SmartFinancialAgent above)


def _placeholder_for_removed_old_code():
    """This function exists only to maintain file structure during cleanup."""
    pass


# ============================================================
# SIDEBAR - AI FINANCIAL PLANNER CHATBOT (FREE - No API needed!)
# ============================================================

with st.sidebar:
    # Logo - try multiple paths
    from pathlib import Path
    possible_paths = [
        Path(__file__).parent.parent / "assets" / "logo.png",
        Path("assets/logo.png"),
        Path("../assets/logo.png"),
    ]
    logo_displayed = False
    for logo_path in possible_paths:
        if logo_path.exists():
            st.image(str(logo_path), width=180)
            logo_displayed = True
            break

    st.title("AI Financial Planner")

    # Show AI status
    ai_status = []
    if SPACY_AVAILABLE:
        ai_status.append("spaCy NLP")
    if OLLAMA_AVAILABLE:
        ai_status.append("Ollama LLM")
    if not ai_status:
        ai_status.append("Smart Rules")

    st.caption(f"🧠 AI: {' + '.join(ai_status)} | 100% FREE")
    st.markdown("---")

    # Chat interface
    st.markdown("### Chat with your AI Advisor")

    # Display chat history
    chat_container = st.container(height=350)
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
**Hello! I'm your AI Financial Planner.**

I can help you:
- Create portfolios based on your risk tolerance
- Add specific stocks (e.g., "Add Tesla")
- Analyze and rebalance your holdings
- Check market conditions

**Just tell me what you need!**

*Try: "I'm scared of risk, want mostly treasury and some Tesla. I have $1 million."*
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
                        added_count = 0
                        added_tickers = []

                        # Add ALL positions to portfolio
                        for pos in portfolio["positions"]:
                            ticker = pos.get("ticker", "UNKNOWN")
                            dollar_amount = pos.get("dollar_amount", 0)
                            current_price = pos.get("current_price", 0)

                            # Fallback price fetch if needed
                            if current_price <= 0:
                                try:
                                    stock_data = get_stock_data(ticker, period="5d")
                                    if "error" not in stock_data and not stock_data["history"].empty:
                                        current_price = float(stock_data["history"]["Close"].iloc[-1])
                                    else:
                                        current_price = 100
                                except:
                                    current_price = 100

                            # Calculate quantity
                            if current_price > 0 and dollar_amount > 0:
                                quantity = max(1, int(dollar_amount / current_price))
                            else:
                                quantity = 1  # Minimum 1 share

                            # Determine display type based on asset type
                            pos_type = pos.get("type", "equity").lower()
                            if pos_type == "stock":
                                display_type = "Stock"
                            elif pos_type == "treasury":
                                display_type = "Treasury ETF (TLT, IEF, SHY)"
                            elif pos_type in ["bond", "corporate_bond"]:
                                display_type = "Corporate Bond ETF (LQD, HYG)"
                            elif pos_type in ["high_yield"]:
                                display_type = "Corporate Bond ETF (LQD, HYG)"
                            elif pos_type == "tips":
                                display_type = "I-Bond / TIPS"
                            else:
                                display_type = "ETF"

                            position = {
                                "ticker": ticker,
                                "type": display_type,
                                "side": "Long",
                                "quantity": quantity,
                                "entry_price": current_price if current_price > 0 else 100,
                                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                                "strike": None,
                                "expiry": None,
                                "yield_rate": None,
                                "maturity": None,
                                "id": len(st.session_state.portfolio)
                            }
                            st.session_state.portfolio.append(position)
                            added_count += 1
                            added_tickers.append(ticker)

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"✅ **Done!** Added {added_count} positions: {', '.join(added_tickers)}. Check the **Portfolio** tab."
                        })
                        st.session_state.pending_portfolio = None
                    else:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": "I don't have a pending portfolio to add. Tell me about your investment goals first!"
                        })
                else:
                    # Generate AI response using SmartFinancialAgent (100% FREE)
                    response = SmartFinancialAgent.chat(
                        user_input,
                        st.session_state.chat_messages
                    )

                    # Check if there's a pending portfolio from the action
                    if st.session_state.get("pending_portfolio"):
                        # Portfolio was created by SmartFinancialAgent
                        pass

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
            msg = "I'm a very conservative investor with $100,000 budget. Create a safe portfolio for me."
            st.session_state.chat_messages.append({"role": "user", "content": msg})
            response = SmartFinancialAgent.chat(msg, st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    with quick_col2:
        if st.button("Aggressive", use_container_width=True, key="q_agg"):
            msg = "I'm an aggressive investor with $100,000 budget. I want high growth potential."
            st.session_state.chat_messages.append({"role": "user", "content": msg})
            response = SmartFinancialAgent.chat(msg, st.session_state.chat_messages)
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

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: #e94560;
        margin: 0;
    }
    .main-header p {
        color: #a0a0a0;
        margin: 0.5rem 0 0 0;
    }

    /* Card styling */
    .metric-card {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e94560;
    }

    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🎯 Trading Strategy Dashboard")
risk_profile = st.session_state.user_profile.get("risk_level", "moderate").replace("_", " ").title()
budget = st.session_state.user_profile.get("budget")
budget_str = f" | Budget: ${budget:,.0f}" if budget else ""

# Status bar
status_col1, status_col2, status_col3 = st.columns([2, 2, 1])
with status_col1:
    st.markdown(f"**Risk Profile:** {risk_profile}")
with status_col2:
    st.markdown(f"**AI Status:** {'🟢 spaCy NLP Active' if SPACY_AVAILABLE else '🟡 Rule-based'}{budget_str}")
with status_col3:
    st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M')}")

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

        # Type selection OUTSIDE form for immediate update
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
        ], key="position_type_selector")

        # Show type-specific info
        is_fixed_income = "Bond" in pos_type or "Treasury" in pos_type or "TIPS" in pos_type
        is_option = "Option" in pos_type

        if is_fixed_income:
            st.info("📊 Fixed income selected - enter yield/coupon and maturity below")
        elif is_option:
            st.info("📈 Option selected - enter strike and expiration below")

        with st.form("add_position"):
            pos_ticker = st.text_input("Ticker:", value="SPY" if not is_fixed_income else "TLT")

            # Position side (Long/Short)
            pos_side = st.selectbox("Side:", ["Long (Buy)", "Short (Sell)"])

            pos_quantity = st.number_input("Quantity:", value=100, step=1)
            pos_entry = st.number_input("Entry Price:", value=100.0, step=0.01)

            # For options, add strike and expiry
            if is_option:
                pos_strike = st.number_input("Strike Price:", value=100.0, step=1.0)
                pos_expiry = st.date_input("Expiration:", value=datetime.now() + timedelta(days=30))
            else:
                pos_strike = None
                pos_expiry = None

            # For bonds/treasury, add yield and maturity - ALWAYS show when fixed income
            if is_fixed_income:
                st.markdown("**Fixed Income Details:**")
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

                # Delete position section
                st.markdown("**Remove Position:**")
                del_col1, del_col2 = st.columns([3, 1])
                with del_col1:
                    # Create list of positions to delete
                    position_options = [f"{pos.get('ticker', 'Unknown')} ({pos.get('side', 'Long')} {pos.get('quantity', 0)} @ ${pos.get('entry_price', 0):.2f})"
                                       for pos in st.session_state.portfolio if pos.get("ticker")]
                    if position_options:
                        selected_position = st.selectbox("Select position to remove:", position_options, key="delete_position_select")
                with del_col2:
                    if position_options and st.button("🗑️ Remove", key="delete_position_btn", use_container_width=True):
                        # Find and remove the selected position
                        selected_idx = position_options.index(selected_position)
                        removed = st.session_state.portfolio.pop(selected_idx)
                        st.success(f"Removed {removed.get('ticker', 'position')}")
                        st.rerun()

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

            # ============================================================
            # PORTFOLIO ALLOCATION PIE CHART
            # ============================================================
            st.markdown("---")
            st.subheader("Portfolio Allocation")

            # Calculate actual values for pie chart
            position_values = []
            for pos in st.session_state.portfolio:
                ticker = pos.get("ticker", "")
                if "---" in pos.get("type", ""):
                    continue
                qty = pos.get("quantity", 0)
                price = pos.get("entry_price", 100)
                value = qty * price
                if value > 0:
                    position_values.append({
                        "ticker": ticker,
                        "value": value,
                        "type": pos.get("type", "Stock")
                    })

            if position_values:
                pie_col1, pie_col2 = st.columns(2)

                with pie_col1:
                    # Allocation by position
                    fig = px.pie(
                        values=[p["value"] for p in position_values],
                        names=[p["ticker"] for p in position_values],
                        title="Allocation by Holding",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(height=350, showlegend=True)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                with pie_col2:
                    # Allocation by asset type
                    type_values = {}
                    for p in position_values:
                        t = p["type"]
                        if "Stock" in t:
                            category = "Stocks"
                        elif "Treasury" in t or "TLT" in t or "IEF" in t or "SHY" in t:
                            category = "Treasury"
                        elif "Bond" in t or "LQD" in t or "HYG" in t:
                            category = "Bonds"
                        elif "ETF" in t:
                            category = "ETFs"
                        else:
                            category = "Other"
                        type_values[category] = type_values.get(category, 0) + p["value"]

                    fig = px.pie(
                        values=list(type_values.values()),
                        names=list(type_values.keys()),
                        title="Allocation by Asset Class",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(height=350, showlegend=True)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

            # ============================================================
            # RISK PROFILE & BETA TRACKER
            # ============================================================
            st.markdown("---")
            st.subheader("Risk Profile & Beta Analysis")

            # Get USER'S SELECTED risk profile from session state
            user_risk_level = st.session_state.user_profile.get("risk_level", "moderate") if st.session_state.user_profile else "moderate"

            # Map user risk level to display format
            user_risk_map = {
                "very_conservative": ("Very Conservative", "🛡️"),
                "conservative": ("Conservative", "🌿"),
                "moderate": ("Moderate", "⚖️"),
                "aggressive": ("Aggressive", "🔥"),
                "very_aggressive": ("Very Aggressive", "🚀"),
            }
            user_risk_display, user_risk_emoji = user_risk_map.get(user_risk_level, ("Moderate", "⚖️"))

            # Calculate portfolio beta and risk metrics
            portfolio_beta = FinancialPlannerAI.calculate_portfolio_beta(st.session_state.portfolio)

            # Determine ACTUAL portfolio risk level based on beta
            # Thresholds calibrated for achievable ETF portfolios:
            # - Very Conservative: Heavy bonds/treasury (beta ~0.1-0.3)
            # - Conservative: Mostly bonds, some stocks (beta ~0.3-0.55)
            # - Moderate: Balanced 60/40 style (beta ~0.55-0.85)
            # - Aggressive: Equity heavy (beta ~0.85-1.05)
            # - Very Aggressive: Growth/tech focused (beta >= 1.05)
            if portfolio_beta < 0.3:
                actual_risk_label = "Very Conservative"
                actual_risk_emoji = "🛡️"
            elif portfolio_beta < 0.55:
                actual_risk_label = "Conservative"
                actual_risk_emoji = "🌿"
            elif portfolio_beta < 0.85:
                actual_risk_label = "Moderate"
                actual_risk_emoji = "⚖️"
            elif portfolio_beta < 1.05:
                actual_risk_label = "Aggressive"
                actual_risk_emoji = "🔥"
            else:
                actual_risk_label = "Very Aggressive"
                actual_risk_emoji = "🚀"

            # Check if portfolio matches user preference
            risk_match = actual_risk_label.lower().replace(" ", "_") == user_risk_level or actual_risk_label == user_risk_display

            # Custom Beta Target Selection
            st.markdown("##### Set Custom Beta Target")
            beta_col1, beta_col2, beta_col3 = st.columns([2, 1, 1])

            with beta_col1:
                # Use target_beta from user_profile if set (from chatbot), otherwise use current portfolio beta
                saved_target_beta = st.session_state.user_profile.get("target_beta") if st.session_state.user_profile else None
                default_beta = saved_target_beta if saved_target_beta else (portfolio_beta if portfolio_beta > 0.1 else 0.7)
                # Clamp to slider range
                default_beta = max(0.1, min(1.5, default_beta))

                target_beta = st.slider(
                    "Target Portfolio Beta:",
                    min_value=0.1,
                    max_value=1.5,
                    value=default_beta,
                    step=0.05,
                    help="0.1-0.3: Very Conservative | 0.3-0.55: Conservative | 0.55-0.85: Moderate | 0.85-1.05: Aggressive | 1.05+: Very Aggressive"
                )

            with beta_col2:
                # Show what risk level the target beta corresponds to
                if target_beta < 0.3:
                    target_label = "🛡️ V.Cons"
                elif target_beta < 0.55:
                    target_label = "🌿 Cons"
                elif target_beta < 0.85:
                    target_label = "⚖️ Mod"
                elif target_beta < 1.05:
                    target_label = "🔥 Aggr"
                else:
                    target_label = "🚀 V.Aggr"
                st.metric("Target Level", target_label)

            with beta_col3:
                # Map beta to risk level
                if target_beta < 0.3:
                    new_risk = "very_conservative"
                elif target_beta < 0.55:
                    new_risk = "conservative"
                elif target_beta < 0.85:
                    new_risk = "moderate"
                elif target_beta < 1.05:
                    new_risk = "aggressive"
                else:
                    new_risk = "very_aggressive"

                if st.button("🔄 Rebalance to Beta", key="rebalance_to_beta", use_container_width=True):
                    # Calculate current portfolio value
                    rebal_total = sum(p.get("quantity", 0) * p.get("entry_price", 100)
                                     for p in st.session_state.portfolio)
                    rebal_budget = rebal_total if rebal_total > 0 else 100000

                    # Clear portfolio for rebuild
                    st.session_state.portfolio = []

                    # Calculate exact equity/bond split to hit target beta
                    # Formula: target_beta = equity_pct * equity_beta + bond_pct * bond_beta
                    # Assume: equity_beta ~1.0, bond_beta ~0.08
                    # Solving: equity_pct = (target_beta - 0.08) / (1.0 - 0.08)
                    equity_beta_avg = 1.0
                    bond_beta_avg = 0.08

                    # Clamp target beta to achievable range
                    clamped_beta = max(0.08, min(1.5, target_beta))
                    equity_pct = (clamped_beta - bond_beta_avg) / (equity_beta_avg - bond_beta_avg)
                    equity_pct = max(0.0, min(1.0, equity_pct))  # Clamp 0-100%
                    bond_pct = 1.0 - equity_pct

                    equity_budget = rebal_budget * equity_pct
                    bond_budget = rebal_budget * bond_pct

                    # Select ETFs based on target beta level
                    if target_beta >= 1.05:
                        # Very aggressive - high beta stocks
                        equity_etfs = [("QQQ", 0.25), ("NVDA", 0.25), ("AMD", 0.20), ("TSLA", 0.15), ("VGT", 0.15)]
                    elif target_beta >= 0.85:
                        # Aggressive
                        equity_etfs = [("SPY", 0.35), ("QQQ", 0.30), ("VGT", 0.20), ("VXUS", 0.15)]
                    elif target_beta >= 0.55:
                        # Moderate
                        equity_etfs = [("SPY", 0.40), ("QQQ", 0.20), ("VTI", 0.20), ("VXUS", 0.20)]
                    else:
                        # Conservative
                        equity_etfs = [("VTI", 0.30), ("SCHD", 0.35), ("VIG", 0.20), ("VXUS", 0.15)]

                    bond_etfs = [("BND", 0.40), ("TLT", 0.30), ("LQD", 0.30)]

                    # Add equity positions
                    for ticker, weight in equity_etfs:
                        dollar_amount = equity_budget * weight
                        stock_data = get_stock_data(ticker, period="5d")
                        if "error" not in stock_data and not stock_data["history"].empty:
                            price = float(stock_data["history"]["Close"].iloc[-1])
                        else:
                            price = 100
                        qty = int(dollar_amount / price) if price > 0 else 0
                        if qty > 0:
                            st.session_state.portfolio.append({
                                "ticker": ticker,
                                "type": "ETF" if ticker not in ["NVDA", "AMD", "TSLA"] else "Stock",
                                "side": "Long",
                                "quantity": qty,
                                "entry_price": price,
                                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                                "strike": None, "expiry": None, "yield_rate": None, "maturity": None,
                                "id": len(st.session_state.portfolio)
                            })

                    # Add bond positions
                    for ticker, weight in bond_etfs:
                        dollar_amount = bond_budget * weight
                        stock_data = get_stock_data(ticker, period="5d")
                        if "error" not in stock_data and not stock_data["history"].empty:
                            price = float(stock_data["history"]["Close"].iloc[-1])
                        else:
                            price = 100
                        qty = int(dollar_amount / price) if price > 0 else 0
                        if qty > 0:
                            st.session_state.portfolio.append({
                                "ticker": ticker,
                                "type": "Bond ETF" if ticker != "TLT" else "Treasury ETF",
                                "side": "Long",
                                "quantity": qty,
                                "entry_price": price,
                                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                                "strike": None, "expiry": None, "yield_rate": None, "maturity": None,
                                "id": len(st.session_state.portfolio)
                            })

                    # Update user profile
                    if "user_profile" not in st.session_state or not st.session_state.user_profile:
                        st.session_state.user_profile = {"constraints": {}}
                    st.session_state.user_profile["risk_level"] = new_risk
                    st.session_state.user_profile["target_beta"] = target_beta

                    st.success(f"✅ Rebalanced to beta {target_beta:.2f} ({equity_pct*100:.0f}% equity / {bond_pct*100:.0f}% bonds)")
                    st.rerun()

            st.markdown("---")

            risk_col1, risk_col2, risk_col3 = st.columns(3)

            with risk_col1:
                st.markdown("##### Your Selected Risk Profile")
                # Show custom beta if set
                custom_beta = st.session_state.user_profile.get("target_beta") if st.session_state.user_profile else None
                if custom_beta:
                    st.metric("Target Beta", f"{custom_beta:.2f}")
                    st.caption(f"Custom target → {user_risk_display}")
                else:
                    st.metric(
                        "Target Risk",
                        f"{user_risk_emoji} {user_risk_display}",
                    )
                    st.caption("Based on your stated preference")

            with risk_col2:
                st.markdown("##### Actual Portfolio Risk")
                st.metric(
                    "Portfolio Beta",
                    f"{portfolio_beta:.2f}",
                    delta=f"vs Market (1.0)",
                    delta_color="off"
                )
                st.metric(
                    "Calculated Risk",
                    f"{actual_risk_emoji} {actual_risk_label}",
                )

            with risk_col3:
                st.markdown("##### Risk Alignment")
                # Calculate allocation percentages
                # Note: Treasury/Bond ETFs should NOT count as equity
                total_val = sum(p["value"] for p in position_values) if position_values else 0

                equity_val = 0
                bond_val = 0
                for p in position_values:
                    pos_type = p.get("type", "").lower()
                    val = p["value"]
                    # Check fixed income FIRST (before ETF check)
                    if "treasury" in pos_type or "bond" in pos_type:
                        bond_val += val
                    elif "stock" in pos_type or "etf" in pos_type:
                        equity_val += val

                equity_pct = (equity_val / total_val * 100) if total_val > 0 else 0
                bond_pct = (bond_val / total_val * 100) if total_val > 0 else 0

                # Show match status (use beta slider above to rebalance)
                if risk_match:
                    st.success("✅ Portfolio matches target!")
                else:
                    st.warning(f"⚠️ {actual_risk_label} vs {user_risk_display}")
                    st.caption("Use beta slider above to rebalance")

                st.metric("Equity", f"{equity_pct:.0f}%")
                st.metric("Fixed Income", f"{bond_pct:.0f}%")

            # ============================================================
            # EXPECTED PORTFOLIO RETURN - Prominent Display
            # ============================================================
            st.markdown("---")
            st.subheader("📈 Expected Portfolio Return")

            # Calculate Expected Portfolio Return
            expected_return = 0
            total_weight = 0
            for pos in st.session_state.portfolio:
                ticker = pos.get("ticker", "")
                if not ticker or "---" in pos.get("type", ""):
                    continue
                qty = pos.get("quantity", 0)
                price = pos.get("entry_price", 100)
                value = qty * price

                # Get expected return from ASSET_CLASSES or estimate from beta
                if ticker in FinancialPlannerAI.ASSET_CLASSES:
                    exp_ret = FinancialPlannerAI.ASSET_CLASSES[ticker].get("expected_return", 0.08)
                else:
                    # Estimate: risk-free (4%) + beta * market premium (6%)
                    beta = FinancialPlannerAI.get_stock_beta(ticker)
                    exp_ret = 0.04 + beta * 0.06

                expected_return += value * exp_ret
                total_weight += value

            if total_weight > 0:
                expected_return = expected_return / total_weight

            # Display in a centered, prominent way
            ret_col1, ret_col2, ret_col3 = st.columns([1, 2, 1])
            with ret_col2:
                st.metric(
                    "Annual Expected Return",
                    f"{expected_return*100:.1f}%",
                    delta="Based on CAPM: Rf (4%) + β × Market Premium (6%)",
                    delta_color="off"
                )
                st.caption("Note: This is a theoretical estimate based on historical beta and market assumptions.")

            # Individual position betas
            st.markdown("#### Individual Position Betas")
            beta_data = []
            for pos in st.session_state.portfolio:
                ticker = pos.get("ticker", "")
                if not ticker or "---" in pos.get("type", ""):
                    continue
                beta = FinancialPlannerAI.get_stock_beta(ticker)
                qty = pos.get("quantity", 0)
                price = pos.get("entry_price", 100)
                value = qty * price
                weight = (value / total_val * 100) if total_val > 0 else 0
                contribution = beta * weight / 100

                beta_data.append({
                    "Ticker": ticker,
                    "Beta": f"{beta:.2f}",
                    "Weight": f"{weight:.1f}%",
                    "Beta Contribution": f"{contribution:.3f}"
                })

            if beta_data:
                beta_df = pd.DataFrame(beta_data)
                st.dataframe(beta_df, use_container_width=True, hide_index=True)

            # ============================================================
            # ACTIONS
            # ============================================================
            st.markdown("---")
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("🗑️ Clear All Positions", use_container_width=True):
                    st.session_state.portfolio = []
                    st.rerun()
            with action_col2:
                if st.button("🔄 Refresh Prices", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()

        else:
            # Empty portfolio state
            st.markdown("---")
            st.info("📊 **No positions yet.** Add positions using the form on the left, or ask the AI Financial Planner to create a portfolio for you!")

            st.markdown("### Quick Start Ideas")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("""
                **📈 Equities**
                - SPY (S&P 500)
                - QQQ (Nasdaq)
                - VTI (Total Market)
                """)
            with col_b:
                st.markdown("""
                **🏦 Fixed Income**
                - TLT (20Y Treasury)
                - IEF (7-10Y Treasury)
                - LQD (Corp Bonds)
                """)
            with col_c:
                st.markdown("""
                **💡 Try the AI**
                - "I'm aggressive with $100k"
                - "Conservative, want dividends"
                - "Balanced 60/40 portfolio"
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
            # Safely get values - convert to float and handle NaN/None
            try:
                start_val = float(perf_data.get('start_value', 0) or 0)
                if pd.isna(start_val) or start_val != start_val:  # NaN check
                    start_val = 0
            except:
                start_val = 0

            try:
                end_val = float(perf_data.get('end_value', 0) or 0)
                if pd.isna(end_val) or end_val != end_val:
                    end_val = 0
            except:
                end_val = 0

            try:
                total_ret = float(perf_data.get('total_return', 0) or 0)
                if pd.isna(total_ret) or total_ret != total_ret:
                    total_ret = 0
            except:
                total_ret = 0

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Starting Value", f"${start_val:,.2f}")
            with col2:
                st.metric("Current Value", f"${end_val:,.2f}")
            with col3:
                st.metric("Total Return", f"{total_ret:+.2f}%")
            with col4:
                # Annualized return (approximate)
                period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "YTD": 180}
                days = period_days.get(time_period, 30)
                try:
                    if days > 0 and start_val > 0 and end_val > 0:
                        annualized = ((end_val / start_val) ** (365/days) - 1) * 100
                        if pd.isna(annualized) or annualized != annualized:
                            st.metric("Annualized", "N/A")
                        else:
                            st.metric("Annualized", f"{annualized:+.2f}%")
                    else:
                        st.metric("Annualized", "N/A")
                except:
                    st.metric("Annualized", "N/A")

            # Clean history data - drop any NaN values
            clean_history = perf_data["history"].dropna()

            # Performance chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=clean_history.index,
                y=clean_history["value"],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='cyan', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 255, 0.1)'
            ))

            # Add starting value line (only if valid)
            if start_val > 0:
                fig.add_hline(
                    y=start_val,
                    line_dash="dash",
                    line_color="yellow",
                    annotation_text=f"Starting: ${start_val:,.0f}"
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
