"""
Fixed Income Dashboard
======================
Yield curve + credit spread analyzer for aggressive traders.

Uses FRED API (free) or yfinance as fallback for treasury data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ============================================================
# YIELD CURVE REGIME DEFINITIONS
# ============================================================

CURVE_REGIMES = {
    "STEEP": {
        "spread_range": (1.5, 5.0),  # 10Y-2Y spread in %
        "description": "Economy expanding, Fed accommodative",
        "implication": "Risk-on environment",
    },
    "NORMAL": {
        "spread_range": (0.5, 1.5),
        "description": "Typical conditions",
        "implication": "Balanced risk/reward",
    },
    "FLAT": {
        "spread_range": (-0.25, 0.5),
        "description": "Late cycle, Fed tightening",
        "implication": "Caution warranted",
    },
    "INVERTED": {
        "spread_range": (-5.0, -0.25),
        "description": "Recession signal (historically reliable)",
        "implication": "Defensive positioning",
    },
}

# Credit spread regimes
CREDIT_REGIMES = {
    "TIGHT": {
        "spread_range": (0, 3.5),  # HY spread in %
        "description": "Risk appetite high, credit easy",
        "implication": "Late cycle euphoria or early recovery",
    },
    "NORMAL": {
        "spread_range": (3.5, 5.0),
        "description": "Typical credit conditions",
        "implication": "Fair compensation for risk",
    },
    "WIDE": {
        "spread_range": (5.0, 8.0),
        "description": "Stress emerging",
        "implication": "Opportunity building",
    },
    "CRISIS": {
        "spread_range": (8.0, 25.0),
        "description": "Credit freeze, distress",
        "implication": "Generational buying opportunity in quality",
    },
}

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_treasury_yields_yfinance() -> Dict[str, float]:
    """
    Fetch current treasury yields using yfinance ETF proxies.

    Uses treasury ETF yields as approximation when FRED unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    # Treasury ETFs as proxies
    tickers = {
        "SHY": "1-3Y",    # Short-term
        "IEI": "3-7Y",    # Intermediate
        "IEF": "7-10Y",   # 10Y proxy
        "TLT": "20+Y",    # Long-term
    }

    yields = {}
    for ticker, label in tickers.items():
        try:
            data = yf.Ticker(ticker)
            info = data.info
            # Get SEC yield if available
            if "yield" in info and info["yield"]:
                yields[label] = info["yield"] * 100
        except Exception:
            pass

    return yields


def fetch_treasury_yields_fred() -> Dict[str, float]:
    """
    Fetch treasury yields from FRED API.

    Requires: pip install fredapi
    And FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
    """
    try:
        from fredapi import Fred
        import os

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY environment variable not set")

        fred = Fred(api_key=api_key)

        series = {
            "DGS1MO": "1M",
            "DGS3MO": "3M",
            "DGS6MO": "6M",
            "DGS1": "1Y",
            "DGS2": "2Y",
            "DGS5": "5Y",
            "DGS7": "7Y",
            "DGS10": "10Y",
            "DGS20": "20Y",
            "DGS30": "30Y",
        }

        yields = {}
        for series_id, label in series.items():
            try:
                data = fred.get_series(series_id)
                if not data.empty:
                    yields[label] = float(data.iloc[-1])
            except Exception:
                pass

        return yields

    except ImportError:
        return {}
    except Exception:
        return {}


def fetch_treasury_data_yfinance(lookback_years: int = 2) -> pd.DataFrame:
    """Fetch treasury ETF price data for historical analysis."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    # Use ^TNX (10Y yield) and ^FVX (5Y yield) indices
    tickers = ["^TNX", "^FVX", "^IRX"]  # 10Y, 5Y, 3M

    data = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"),
                       end=end_date.strftime("%Y-%m-%d"), progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        # Extract Close prices
        closes = data["Close"]
    else:
        closes = data

    return closes


def get_current_yields() -> Dict[str, float]:
    """Get current treasury yields from best available source."""

    # Try FRED first (more accurate)
    yields = fetch_treasury_yields_fred()

    if yields:
        return {"source": "FRED", "yields": yields}

    # Fallback to yfinance indices
    try:
        import yfinance as yf

        # Fetch yield indices directly
        tnx = yf.Ticker("^TNX")  # 10Y yield
        fvx = yf.Ticker("^FVX")  # 5Y yield
        irx = yf.Ticker("^IRX")  # 3M yield
        tyx = yf.Ticker("^TYX")  # 30Y yield

        yields = {}

        for ticker, label in [("^TNX", "10Y"), ("^FVX", "5Y"), ("^IRX", "3M"), ("^TYX", "30Y")]:
            try:
                data = yf.download(ticker, period="5d", progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        close = data["Close"].iloc[:, 0].iloc[-1]
                    else:
                        close = data["Close"].iloc[-1]
                    yields[label] = float(close)
            except Exception:
                pass

        # Estimate 2Y from interpolation or use proxy
        if "5Y" in yields and "3M" in yields:
            yields["2Y"] = yields["3M"] + (yields["5Y"] - yields["3M"]) * 0.4

        return {"source": "Yahoo Finance", "yields": yields}

    except Exception as e:
        return {"source": "Error", "yields": {}, "error": str(e)}


# ============================================================
# CURVE ANALYSIS
# ============================================================

def calculate_spreads(yields: Dict[str, float]) -> Dict[str, float]:
    """Calculate key yield curve spreads."""
    spreads = {}

    # 10Y - 2Y (most watched)
    if "10Y" in yields and "2Y" in yields:
        spreads["10Y-2Y"] = yields["10Y"] - yields["2Y"]

    # 10Y - 3M (Fed's preferred)
    if "10Y" in yields and "3M" in yields:
        spreads["10Y-3M"] = yields["10Y"] - yields["3M"]

    # 30Y - 10Y (long end steepness)
    if "30Y" in yields and "10Y" in yields:
        spreads["30Y-10Y"] = yields["30Y"] - yields["10Y"]

    # 5Y - 2Y (belly)
    if "5Y" in yields and "2Y" in yields:
        spreads["5Y-2Y"] = yields["5Y"] - yields["2Y"]

    return spreads


def determine_curve_regime(spreads: Dict[str, float]) -> str:
    """Determine yield curve regime from spreads."""

    # Use 10Y-2Y as primary indicator
    spread_10y2y = spreads.get("10Y-2Y", spreads.get("10Y-3M", 0))

    for regime, config in CURVE_REGIMES.items():
        low, high = config["spread_range"]
        if low <= spread_10y2y < high:
            return regime

    if spread_10y2y < -0.25:
        return "INVERTED"
    return "STEEP"


def get_curve_shape_description(yields: Dict[str, float]) -> str:
    """Describe the yield curve shape."""

    if len(yields) < 3:
        return "Insufficient data for curve analysis"

    # Sort by maturity
    maturity_order = ["3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    sorted_yields = []
    for mat in maturity_order:
        if mat in yields:
            sorted_yields.append((mat, yields[mat]))

    if len(sorted_yields) < 3:
        return "Insufficient data"

    # Check for inversions
    inversions = []
    for i in range(len(sorted_yields) - 1):
        if sorted_yields[i][1] > sorted_yields[i + 1][1]:
            inversions.append(f"{sorted_yields[i][0]}/{sorted_yields[i+1][0]}")

    if inversions:
        return f"INVERTED at: {', '.join(inversions)}"

    # Calculate steepness
    short = sorted_yields[0][1]
    long = sorted_yields[-1][1]
    steepness = long - short

    if steepness > 2.0:
        return f"STEEP (short-to-long spread: {steepness:.2f}%)"
    elif steepness > 1.0:
        return f"NORMAL (short-to-long spread: {steepness:.2f}%)"
    elif steepness > 0:
        return f"FLAT (short-to-long spread: {steepness:.2f}%)"
    else:
        return f"INVERTED (short-to-long spread: {steepness:.2f}%)"


# ============================================================
# STRATEGY RECOMMENDATIONS (Aggressive Profile)
# ============================================================

FI_STRATEGIES = {
    "STEEP": {
        "primary": [
            {
                "name": "Curve Steepener",
                "action": "Long 2Y, Short 10Y (or 30Y)",
                "rationale": "Bet curve normalizes as economy cools",
                "instruments": "TBT (short 20Y), SHY (long short-end)",
                "risk": "Curve can stay steep if growth surprises",
            },
            {
                "name": "Long Duration on Dips",
                "action": "Buy TLT on selloffs",
                "rationale": "Steep curve = rates peak coming",
                "instruments": "TLT, ZROZ, EDV",
                "risk": "Inflation re-accelerates",
            },
            {
                "name": "Underweight Cash",
                "action": "Deploy cash into risk assets",
                "rationale": "Steep curve = Fed easy, risk assets favored",
                "instruments": "Reduce T-bills, add equities/credit",
                "risk": "Early cycle can still have volatility",
            },
        ],
        "avoid": ["Heavy long duration (yields can rise more)", "Short-term bonds (opportunity cost)"],
    },
    "NORMAL": {
        "primary": [
            {
                "name": "Barbell Strategy",
                "action": "Mix short + long duration, skip middle",
                "rationale": "Belly often underperforms in transitions",
                "instruments": "SHY + TLT, avoid IEF",
                "risk": "Middle of curve outperforms in rare scenarios",
            },
            {
                "name": "Laddered Positions",
                "action": "Equal weight across maturities",
                "rationale": "Balanced for uncertain direction",
                "instruments": "1Y, 3Y, 5Y, 10Y treasuries",
                "risk": "Underperforms if you knew direction",
            },
        ],
        "avoid": ["Big directional bets", "Concentrated duration exposure"],
    },
    "FLAT": {
        "primary": [
            {
                "name": "Extend Duration",
                "action": "Add long bonds, reduce short",
                "rationale": "Flat curve often precedes rate cuts",
                "instruments": "TLT, EDV, ZROZ",
                "risk": "Inflation forces Fed to stay tight longer",
            },
            {
                "name": "Curve Flattener (if not already flat)",
                "action": "Long 10Y, Short 2Y",
                "rationale": "Flattening tends to continue into inversion",
                "instruments": "IEF vs SHY spread",
                "risk": "Steepening surprise",
            },
            {
                "name": "Reduce Credit Risk",
                "action": "Move from HY to IG or Treasuries",
                "rationale": "Flat curve = late cycle, credit risk rises",
                "instruments": "Sell HYG/JNK, buy LQD/TLT",
                "risk": "Miss final credit rally",
            },
        ],
        "avoid": ["High yield bonds", "Aggressive risk-taking"],
    },
    "INVERTED": {
        "primary": [
            {
                "name": "MAX LONG DURATION",
                "action": "Load up on long bonds",
                "rationale": "Inversion = recession coming = Fed cuts = bonds rally",
                "instruments": "TLT, ZROZ, EDV, TMF (3x leveraged)",
                "risk": "Timing - can take 6-18 months to play out",
            },
            {
                "name": "Curve Steepener",
                "action": "Long 2Y, Short 10Y",
                "rationale": "Curve will steepen when Fed cuts (2Y falls faster)",
                "instruments": "TBT + SHY, or futures spread",
                "risk": "Inversion can deepen before reversing",
            },
            {
                "name": "T-Bills for Optionality",
                "action": "Park cash in 3-6M T-bills",
                "rationale": "High yields + dry powder for buying panic",
                "instruments": "SGOV, BIL, direct T-bills",
                "risk": "Miss the rally if you wait too long",
            },
            {
                "name": "Quality Credit Only",
                "action": "AAA/AA corporates if any credit",
                "rationale": "Recession will blow out HY spreads",
                "instruments": "LQD, VCIT (investment grade)",
                "risk": "All credit sells off in panic",
            },
        ],
        "avoid": ["High yield bonds (spreads will blow out)", "Short duration (missing rally)"],
        "sizing": "INCREASE DURATION ALLOCATION - This is the signal",
    },
}


# ============================================================
# REPORTING
# ============================================================

def format_fixed_income_report(yield_data: Dict, spreads: Dict, regime: str) -> str:
    """Format the fixed income dashboard report."""

    yields = yield_data.get("yields", {})
    source = yield_data.get("source", "Unknown")
    regime_info = CURVE_REGIMES.get(regime, CURVE_REGIMES["NORMAL"])
    strategies = FI_STRATEGIES.get(regime, FI_STRATEGIES["NORMAL"])

    report = f"""
================================================================================
                     FIXED INCOME DASHBOARD
                     High Risk Appetite Profile
================================================================================

DATA SOURCE: {source}

--------------------------------------------------------------------------------
CURRENT YIELDS
--------------------------------------------------------------------------------
"""

    # Sort and display yields
    maturity_order = ["3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    for mat in maturity_order:
        if mat in yields:
            report += f"  {mat:6s}: {yields[mat]:.2f}%\n"

    report += f"""
--------------------------------------------------------------------------------
KEY SPREADS
--------------------------------------------------------------------------------
"""
    for spread_name, value in spreads.items():
        status = "INVERTED" if value < 0 else ""
        report += f"  {spread_name:10s}: {value:+.2f}%  {status}\n"

    curve_shape = get_curve_shape_description(yields)

    report += f"""
--------------------------------------------------------------------------------
CURVE ANALYSIS
--------------------------------------------------------------------------------
  REGIME:      {regime}
  SHAPE:       {curve_shape}
  DESCRIPTION: {regime_info['description']}
  IMPLICATION: {regime_info['implication']}

--------------------------------------------------------------------------------
RECOMMENDED STRATEGIES (Aggressive Profile)
--------------------------------------------------------------------------------
"""

    for i, strategy in enumerate(strategies["primary"], 1):
        report += f"""
{i}. {strategy['name']}
   Action:     {strategy['action']}
   Rationale:  {strategy['rationale']}
   Instruments:{strategy['instruments']}
   Risk:       {strategy['risk']}
"""

    report += f"""
--------------------------------------------------------------------------------
AVOID IN THIS REGIME
--------------------------------------------------------------------------------
"""
    for item in strategies["avoid"]:
        report += f"  - {item}\n"

    if "sizing" in strategies:
        report += f"""
--------------------------------------------------------------------------------
POSITION SIZING
--------------------------------------------------------------------------------
  {strategies['sizing']}
"""

    report += """
================================================================================
"""
    return report


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_fixed_income_dashboard():
    """Run the full fixed income analysis."""

    print("Fetching yield data...")
    yield_data = get_current_yields()

    if not yield_data.get("yields"):
        print("Warning: Could not fetch yield data. Check internet connection.")
        return None

    yields = yield_data["yields"]
    spreads = calculate_spreads(yields)
    regime = determine_curve_regime(spreads)

    report = format_fixed_income_report(yield_data, spreads, regime)
    print(report)

    return {
        "yield_data": yield_data,
        "spreads": spreads,
        "regime": regime,
    }


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

def get_spread_history(lookback_years: int = 5) -> pd.DataFrame:
    """Get historical spread data."""

    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)

        # Fetch 10Y and 2Y yield proxies
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)

        if isinstance(tnx.columns, pd.MultiIndex):
            tnx = tnx["Close"].iloc[:, 0]
        else:
            tnx = tnx["Close"]

        return pd.DataFrame({"10Y": tnx})

    except Exception as e:
        print(f"Error fetching spread history: {e}")
        return pd.DataFrame()


def print_inversion_history():
    """Print historical inversion periods."""

    print("\nHistorical Yield Curve Inversions (10Y-2Y)")
    print("-" * 50)
    print("""
  2006-2007: Inverted -> 2008 Financial Crisis
  2019-2020: Inverted -> COVID Recession
  2022-2024: Inverted -> ??? (now normalized)

  Average lead time: 12-18 months before recession

  Key insight: Inversions don't cause recessions,
  but they predict Fed tightening -> economic slowdown.
""")


if __name__ == "__main__":
    results = run_fixed_income_dashboard()
    print_inversion_history()
