"""
Options Regime Selector
=======================
VIX-based strategy selection for aggressive/high-risk traders.

Determines market regime from VIX levels and recommends options strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ============================================================
# VIX REGIME DEFINITIONS (Calibrated for aggressive trader)
# ============================================================

VIX_REGIMES = {
    "LOW": {
        "range": (0, 14),
        "description": "Complacency - Premium is cheap",
        "market_state": "Grinding higher, low fear",
    },
    "NORMAL": {
        "range": (14, 20),
        "description": "Typical conditions",
        "market_state": "Normal two-way market",
    },
    "ELEVATED": {
        "range": (20, 30),
        "description": "Fear emerging - Premium is rich",
        "market_state": "Pullback/correction mode",
    },
    "CRISIS": {
        "range": (30, 100),
        "description": "Panic - Extreme premium",
        "market_state": "Capitulation/crash",
    },
}

# ============================================================
# STRATEGY RECOMMENDATIONS (High Risk Appetite)
# ============================================================

AGGRESSIVE_STRATEGIES = {
    "LOW": {
        "primary": [
            {
                "name": "Sell Naked Puts",
                "ticker_focus": "High-quality stocks you want to own",
                "rationale": "Collect premium while VIX is low; get paid to wait for entries",
                "risk": "Unlimited downside if stock craters",
                "example": "Sell AAPL 5% OTM puts, 30-45 DTE",
                "edge": "IV often overstates realized vol in low VIX environments",
            },
            {
                "name": "Sell Strangles",
                "ticker_focus": "Range-bound stocks, ETFs like SPY/QQQ",
                "rationale": "Profit from theta decay on both sides",
                "risk": "Unlimited on both sides; need to manage",
                "example": "Sell SPY 5% OTM puts + 5% OTM calls, 30-45 DTE",
                "edge": "Low VIX = low IV = stocks likely stay in range",
            },
            {
                "name": "Short VIX Calls (via VXX/UVXY puts)",
                "ticker_focus": "VXX, UVXY",
                "rationale": "VIX products decay structurally; accelerates in low vol",
                "risk": "VIX spike can cause 50%+ loss quickly",
                "example": "Buy UVXY puts or put spreads",
                "edge": "Contango + decay = structural tailwind",
            },
        ],
        "avoid": ["Buying premium (theta kills you)", "Long straddles (IV too low)"],
        "position_sizing": "Can be more aggressive; vol expansion unlikely",
    },
    "NORMAL": {
        "primary": [
            {
                "name": "Put Credit Spreads",
                "ticker_focus": "Strong trend stocks, indices",
                "rationale": "Defined risk with positive theta",
                "risk": "Max loss = spread width - premium",
                "example": "Sell SPY put spread, 5-10% OTM, 30-45 DTE",
                "edge": "Probability favors sellers in neutral/bull markets",
            },
            {
                "name": "Iron Condors",
                "ticker_focus": "Range-bound names, indices",
                "rationale": "Profit if underlying stays in range",
                "risk": "Loss on breakout either direction",
                "example": "SPY iron condor, 10% wings, 30-45 DTE",
                "edge": "Collect premium on both sides in calm markets",
            },
            {
                "name": "Calendar Spreads",
                "ticker_focus": "Stocks with upcoming catalysts",
                "rationale": "Benefit from IV term structure",
                "risk": "Big moves kill you",
                "example": "Buy back-month, sell front-month at same strike",
                "edge": "Front-month IV often overpriced",
            },
        ],
        "avoid": ["Naked positions (risk/reward not optimal)", "Large directional bets"],
        "position_sizing": "Standard sizing; market can go either way",
    },
    "ELEVATED": {
        "primary": [
            {
                "name": "Buy Calls on Quality",
                "ticker_focus": "Beaten-down leaders (AAPL, MSFT, GOOGL)",
                "rationale": "IV is high but so is fear; reversals are violent",
                "risk": "Premium expensive; can lose 100% of position",
                "example": "Buy AAPL 3-6 month calls, ATM or slightly OTM",
                "edge": "Fear peaks before price bottoms; be early",
            },
            {
                "name": "LEAPS on Indices",
                "ticker_focus": "SPY, QQQ",
                "rationale": "Long-dated calls benefit from mean reversion",
                "risk": "Time decay if correction extends",
                "example": "Buy SPY 1-year calls, 5-10% OTM",
                "edge": "Markets recover; duration gives you time",
            },
            {
                "name": "Ratio Spreads (1x2)",
                "ticker_focus": "Indices, liquid names",
                "rationale": "Finance long calls by selling 2x OTM calls",
                "risk": "Unlimited upside risk if rally is huge",
                "example": "Buy 1 ATM call, sell 2 calls 10% OTM",
                "edge": "Reduces cost basis significantly",
            },
            {
                "name": "Sell Cash-Secured Puts on Dips",
                "ticker_focus": "Stocks you want to own at lower prices",
                "rationale": "Rich premium + lower strikes = great entries",
                "risk": "Stock keeps falling; you own at higher price",
                "example": "Sell NVDA puts 15% below current, 45 DTE",
                "edge": "Get paid to buy the dip",
            },
        ],
        "avoid": ["Short volatility (can spike further)", "Small position sizes (this is opportunity)"],
        "position_sizing": "INCREASE SIZE - This is where money is made",
    },
    "CRISIS": {
        "primary": [
            {
                "name": "Buy Cheap OTM Calls for Recovery",
                "ticker_focus": "Quality names down 30%+, indices",
                "rationale": "Lottery tickets with asymmetric payoff",
                "risk": "Can lose entire premium; position small",
                "example": "Buy SPY 6-month calls 15% OTM",
                "edge": "Crisis = overshoot; snap-back rallies are massive",
            },
            {
                "name": "Sell Puts on Companies That Survive",
                "ticker_focus": "AAPL, MSFT, JPM, BRK - fortress balance sheets",
                "rationale": "Extreme premium for taking crisis risk",
                "risk": "Company-specific risk; ensure they survive",
                "example": "Sell AAPL puts 20% OTM, collect 5%+ premium",
                "edge": "You're the insurance company when everyone panics",
            },
            {
                "name": "VIX Put Spreads",
                "ticker_focus": "VIX options directly",
                "rationale": "VIX mean-reverts; 30+ is unsustainable",
                "risk": "VIX can stay high longer than expected",
                "example": "Buy VIX put spread 30/20 strikes",
                "edge": "VIX always comes down eventually",
            },
            {
                "name": "Call Spreads (Debit)",
                "ticker_focus": "Indices, quality tech",
                "rationale": "Defined risk bullish bet at crisis lows",
                "risk": "Lose premium if no recovery",
                "example": "Buy SPY call spread ATM/+10%, 90 DTE",
                "edge": "Capped upside but also capped cost",
            },
        ],
        "avoid": ["Short puts without conviction (can gap lower)", "Leverage on leverage (TQQQ options)"],
        "position_sizing": "MAX AGGRESSION - Generational opportunities",
    },
}


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_vix_data(lookback_years: int = 5) -> pd.DataFrame:
    """Fetch VIX historical data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    vix = yf.download("^VIX", start=start_date.strftime("%Y-%m-%d"),
                      end=end_date.strftime("%Y-%m-%d"), progress=False)

    if vix.empty:
        raise ValueError("No VIX data returned")

    # Handle MultiIndex columns
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    return vix


def get_current_vix() -> float:
    """Get the most recent VIX close."""
    vix_data = fetch_vix_data(lookback_years=1)
    return float(vix_data["Close"].iloc[-1])


def calculate_vix_percentile(current_vix: float, lookback_years: int = 2) -> float:
    """Calculate where current VIX sits in historical distribution."""
    vix_data = fetch_vix_data(lookback_years=lookback_years)
    historical_vix = vix_data["Close"].dropna()
    percentile = (historical_vix < current_vix).mean() * 100
    return percentile


def get_vix_stats(lookback_years: int = 2) -> Dict:
    """Get comprehensive VIX statistics."""
    vix_data = fetch_vix_data(lookback_years=lookback_years)
    vix_close = vix_data["Close"].dropna()

    current = float(vix_close.iloc[-1])

    return {
        "current": current,
        "percentile": calculate_vix_percentile(current, lookback_years),
        "mean": float(vix_close.mean()),
        "median": float(vix_close.median()),
        "std": float(vix_close.std()),
        "min": float(vix_close.min()),
        "max": float(vix_close.max()),
        "5th_percentile": float(np.percentile(vix_close, 5)),
        "25th_percentile": float(np.percentile(vix_close, 25)),
        "75th_percentile": float(np.percentile(vix_close, 75)),
        "95th_percentile": float(np.percentile(vix_close, 95)),
    }


# ============================================================
# REGIME DETECTION
# ============================================================

def determine_regime(vix_level: float) -> str:
    """Determine VIX regime based on current level."""
    for regime, config in VIX_REGIMES.items():
        low, high = config["range"]
        if low <= vix_level < high:
            return regime
    return "CRISIS"  # Default if above all ranges


def get_regime_info(regime: str) -> Dict:
    """Get detailed info about a regime."""
    return VIX_REGIMES.get(regime, VIX_REGIMES["NORMAL"])


# ============================================================
# STRATEGY SELECTION
# ============================================================

def get_recommended_strategies(regime: str) -> Dict:
    """Get recommended strategies for the current regime."""
    return AGGRESSIVE_STRATEGIES.get(regime, AGGRESSIVE_STRATEGIES["NORMAL"])


def format_strategy_report(vix_stats: Dict, regime: str, strategies: Dict) -> str:
    """Format a readable strategy report."""
    regime_info = get_regime_info(regime)

    report = f"""
================================================================================
                     OPTIONS REGIME SELECTOR
                     High Risk Appetite Profile
================================================================================

CURRENT VIX: {vix_stats['current']:.2f}
PERCENTILE:  {vix_stats['percentile']:.0f}th (vs last 2 years)
REGIME:      {regime}

MARKET STATE: {regime_info['market_state']}
DESCRIPTION:  {regime_info['description']}

--------------------------------------------------------------------------------
VIX CONTEXT
--------------------------------------------------------------------------------
  Current:    {vix_stats['current']:.2f}
  2Y Mean:    {vix_stats['mean']:.2f}
  2Y Median:  {vix_stats['median']:.2f}
  2Y Range:   {vix_stats['min']:.2f} - {vix_stats['max']:.2f}

  5th %ile:   {vix_stats['5th_percentile']:.2f}  (complacency)
  25th %ile:  {vix_stats['25th_percentile']:.2f}  (low)
  75th %ile:  {vix_stats['75th_percentile']:.2f}  (elevated)
  95th %ile:  {vix_stats['95th_percentile']:.2f}  (fear)

--------------------------------------------------------------------------------
RECOMMENDED STRATEGIES (Aggressive Profile)
--------------------------------------------------------------------------------
"""

    for i, strategy in enumerate(strategies["primary"], 1):
        report += f"""
{i}. {strategy['name']}
   Focus:     {strategy['ticker_focus']}
   Rationale: {strategy['rationale']}
   Risk:      {strategy['risk']}
   Example:   {strategy['example']}
   Edge:      {strategy['edge']}
"""

    report += f"""
--------------------------------------------------------------------------------
AVOID IN THIS REGIME
--------------------------------------------------------------------------------
"""
    for item in strategies["avoid"]:
        report += f"  - {item}\n"

    report += f"""
--------------------------------------------------------------------------------
POSITION SIZING GUIDANCE
--------------------------------------------------------------------------------
  {strategies['position_sizing']}

================================================================================
"""
    return report


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_options_regime_selector():
    """Run the full options regime analysis."""

    print("Fetching VIX data...")
    vix_stats = get_vix_stats(lookback_years=2)

    regime = determine_regime(vix_stats["current"])
    strategies = get_recommended_strategies(regime)

    report = format_strategy_report(vix_stats, regime, strategies)
    print(report)

    return {
        "vix_stats": vix_stats,
        "regime": regime,
        "strategies": strategies,
    }


# ============================================================
# ADDITIONAL UTILITIES
# ============================================================

def get_regime_history(lookback_years: int = 2) -> pd.DataFrame:
    """Get historical regime classification."""
    vix_data = fetch_vix_data(lookback_years=lookback_years)
    vix_close = vix_data["Close"].dropna()

    regimes = []
    for date, vix in vix_close.items():
        regime = determine_regime(float(vix))
        regimes.append({"date": date, "vix": float(vix), "regime": regime})

    return pd.DataFrame(regimes)


def regime_distribution(lookback_years: int = 2) -> Dict:
    """Calculate how much time spent in each regime."""
    history = get_regime_history(lookback_years)
    total = len(history)

    distribution = {}
    for regime in VIX_REGIMES.keys():
        count = len(history[history["regime"] == regime])
        distribution[regime] = {
            "days": count,
            "percentage": count / total * 100,
        }

    return distribution


def print_regime_distribution(lookback_years: int = 2):
    """Print regime distribution summary."""
    dist = regime_distribution(lookback_years)

    print(f"\nVIX Regime Distribution (Last {lookback_years} Years)")
    print("-" * 40)
    for regime, stats in dist.items():
        bar = "=" * int(stats["percentage"] / 2)
        print(f"{regime:10s} {stats['percentage']:5.1f}% {bar}")


if __name__ == "__main__":
    results = run_options_regime_selector()
    print_regime_distribution()
