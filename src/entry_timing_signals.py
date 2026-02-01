"""
Entry Timing Signals
====================
Unified signal system integrating VIX regime and yield curve analysis.

Identifies actionable entry points for aggressive traders.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import our other modules
from options_regime_selector import (
    get_vix_stats,
    determine_regime as determine_vix_regime,
    VIX_REGIMES,
)
from fixed_income_dashboard import (
    get_current_yields,
    calculate_spreads,
    determine_curve_regime,
)


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

class SignalStrength(Enum):
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4


@dataclass
class Signal:
    name: str
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: SignalStrength
    asset_class: str  # "OPTIONS", "FIXED_INCOME", "CROSS_ASSET"
    action: str
    rationale: str
    instruments: List[str]
    risk_note: str


# ============================================================
# VIX-BASED SIGNALS
# ============================================================

def generate_vix_signals(vix_stats: Dict) -> List[Signal]:
    """Generate trading signals based on VIX analysis."""
    signals = []

    vix = vix_stats["current"]
    percentile = vix_stats["percentile"]
    mean = vix_stats["mean"]

    # Signal 1: VIX Spike (ELEVATED/CRISIS entry)
    if vix > 25:
        strength = SignalStrength.STRONG if vix > 30 else SignalStrength.MODERATE
        signals.append(Signal(
            name="VIX SPIKE - BUY RISK",
            direction="BULLISH",
            strength=strength,
            asset_class="OPTIONS",
            action="Buy calls on quality, sell puts on names you want",
            rationale=f"VIX at {vix:.1f} ({percentile:.0f}th percentile) - fear elevated, mean reversion likely",
            instruments=["SPY calls", "QQQ calls", "AAPL/MSFT puts to sell", "VIX put spreads"],
            risk_note="VIX can stay elevated; size for drawdown",
        ))

    # Signal 2: VIX Extreme Low (complacency warning)
    if vix < 13:
        signals.append(Signal(
            name="VIX COMPLACENT - SELL PREMIUM CAREFULLY",
            direction="NEUTRAL",
            strength=SignalStrength.MODERATE,
            asset_class="OPTIONS",
            action="Sell premium but hedge tails; complacency can snap",
            rationale=f"VIX at {vix:.1f} is below 13 - historically precedes spikes",
            instruments=["Sell put spreads (not naked)", "Iron condors with wide wings"],
            risk_note="Low VIX environments can explode quickly (Feb 2018, Aug 2024)",
        ))

    # Signal 3: VIX at/near mean (no strong signal)
    if 14 <= vix <= 20:
        signals.append(Signal(
            name="VIX NEUTRAL - NO STRONG EDGE",
            direction="NEUTRAL",
            strength=SignalStrength.WEAK,
            asset_class="OPTIONS",
            action="Standard strategies; wait for better setup",
            rationale=f"VIX at {vix:.1f} is near mean ({mean:.1f}) - no extreme",
            instruments=["Put credit spreads", "Iron condors", "Calendar spreads"],
            risk_note="Market can move either way from here",
        ))

    # Signal 4: VIX percentile extreme
    if percentile > 90:
        signals.append(Signal(
            name="VIX 90TH+ PERCENTILE - GENERATIONAL",
            direction="BULLISH",
            strength=SignalStrength.EXTREME,
            asset_class="OPTIONS",
            action="MAX AGGRESSION - This is the opportunity",
            rationale=f"VIX at {percentile:.0f}th percentile - historically rare, strong reversion",
            instruments=["LEAPS on indices", "6-month calls", "Sell puts aggressively", "TMF if rates high"],
            risk_note="Be early, size to survive drawdown, but this is when fortunes are made",
        ))
    elif percentile < 10:
        signals.append(Signal(
            name="VIX 10TH PERCENTILE - HEDGE",
            direction="BEARISH",
            strength=SignalStrength.MODERATE,
            asset_class="OPTIONS",
            action="Add tail hedges; cheap protection available",
            rationale=f"VIX at {percentile:.0f}th percentile - protection is on sale",
            instruments=["SPY put spreads 3-6 months out", "VIX calls", "UVXY calls (small size)"],
            risk_note="Timing unknown, but hedges rarely this cheap",
        ))

    return signals


# ============================================================
# YIELD CURVE SIGNALS
# ============================================================

def generate_curve_signals(yields: Dict, spreads: Dict, regime: str) -> List[Signal]:
    """Generate trading signals based on yield curve."""
    signals = []

    spread_10y2y = spreads.get("10Y-2Y", 0)
    spread_10y3m = spreads.get("10Y-3M", 0)

    # Signal 1: Inversion (recession warning)
    if spread_10y2y < 0 or spread_10y3m < 0:
        inverted_spread = "10Y-2Y" if spread_10y2y < 0 else "10Y-3M"
        signals.append(Signal(
            name="CURVE INVERTED - RECESSION SIGNAL",
            direction="BEARISH",
            strength=SignalStrength.STRONG,
            asset_class="FIXED_INCOME",
            action="Go long duration NOW; reduce credit risk",
            rationale=f"{inverted_spread} inverted at {min(spread_10y2y, spread_10y3m):.2f}% - recession in 12-18 months",
            instruments=["TLT", "ZROZ", "EDV", "TMF (levered)", "Reduce HYG/JNK"],
            risk_note="Timing uncertain; can take 6-18 months to play out",
        ))

    # Signal 2: Curve just un-inverted (danger zone)
    if 0 < spread_10y2y < 0.75 and regime == "NORMAL":
        signals.append(Signal(
            name="CURVE JUST UN-INVERTED - WATCH OUT",
            direction="BEARISH",
            strength=SignalStrength.MODERATE,
            asset_class="CROSS_ASSET",
            action="Maintain long duration; be cautious on risk assets",
            rationale=f"10Y-2Y at +{spread_10y2y:.2f}% after inversion - historically recession follows",
            instruments=["Keep TLT", "Reduce equity beta", "Quality over junk"],
            risk_note="Un-inversion often precedes the actual recession starting",
        ))

    # Signal 3: Steep curve (risk on)
    if spread_10y2y > 1.5:
        signals.append(Signal(
            name="STEEP CURVE - RISK ON",
            direction="BULLISH",
            strength=SignalStrength.MODERATE,
            asset_class="CROSS_ASSET",
            action="Add risk; reduce duration; favor equities",
            rationale=f"10Y-2Y at +{spread_10y2y:.2f}% - early cycle, Fed easy",
            instruments=["Reduce TLT", "Add SPY/QQQ", "Financials (XLF) benefit"],
            risk_note="Steep curves can take time to normalize",
        ))

    # Signal 4: High absolute yields (income opportunity)
    ten_year = yields.get("10Y", 0)
    if ten_year > 4.5:
        signals.append(Signal(
            name="HIGH YIELDS - LOCK IN INCOME",
            direction="BULLISH",
            strength=SignalStrength.MODERATE,
            asset_class="FIXED_INCOME",
            action="Consider locking in yields at 10Y+",
            rationale=f"10Y at {ten_year:.2f}% - attractive vs historical average (~3.5%)",
            instruments=["10Y+ Treasuries", "I-bonds", "TIPS", "BND"],
            risk_note="Yields can go higher; dollar-cost average",
        ))
    elif ten_year < 2.0:
        signals.append(Signal(
            name="LOW YIELDS - AVOID DURATION",
            direction="BEARISH",
            strength=SignalStrength.MODERATE,
            asset_class="FIXED_INCOME",
            action="Minimize duration exposure; yields have room to rise",
            rationale=f"10Y at {ten_year:.2f}% - poor risk/reward for duration",
            instruments=["Short duration (SHY, SGOV)", "Floating rate", "Avoid TLT"],
            risk_note="Historically low yields = duration risk",
        ))

    return signals


# ============================================================
# CROSS-ASSET SIGNALS
# ============================================================

def generate_cross_asset_signals(vix_regime: str, curve_regime: str, vix_stats: Dict) -> List[Signal]:
    """Generate signals from cross-asset analysis."""
    signals = []

    # Matrix of regimes
    # VIX LOW/NORMAL + CURVE STEEP = Risk on (early cycle)
    # VIX ELEVATED + CURVE INVERTED = Maximum opportunity (crisis)
    # VIX LOW + CURVE INVERTED = Dangerous complacency
    # VIX HIGH + CURVE STEEP = Recovery underway

    if vix_regime == "LOW" and curve_regime == "INVERTED":
        signals.append(Signal(
            name="DANGEROUS COMPLACENCY",
            direction="BEARISH",
            strength=SignalStrength.STRONG,
            asset_class="CROSS_ASSET",
            action="ADD HEDGES - Market not pricing recession risk",
            rationale="Low VIX + Inverted curve = market ignoring recession signal",
            instruments=["SPY puts (cheap)", "VIX calls", "Reduce gross exposure"],
            risk_note="This setup preceded 2007 and 2019 corrections",
        ))

    elif vix_regime in ["ELEVATED", "CRISIS"] and curve_regime == "INVERTED":
        signals.append(Signal(
            name="MAXIMUM OPPORTUNITY ZONE",
            direction="BULLISH",
            strength=SignalStrength.EXTREME,
            asset_class="CROSS_ASSET",
            action="DEPLOY CAPITAL - Bonds AND equities both attractive",
            rationale="High VIX + Inverted curve = fear + rate cuts coming",
            instruments=["TLT/TMF", "SPY LEAPS", "Quality stocks", "Investment grade credit"],
            risk_note="This is where generational wealth is built",
        ))

    elif vix_regime == "CRISIS" and curve_regime == "STEEP":
        signals.append(Signal(
            name="RECOVERY TRADE",
            direction="BULLISH",
            strength=SignalStrength.STRONG,
            asset_class="CROSS_ASSET",
            action="Favor equities over bonds; recovery underway",
            rationale="High VIX + Steep curve = Fed easing + fear = equity rally",
            instruments=["SPY/QQQ calls", "Reduce TLT", "Cyclicals", "Small caps"],
            risk_note="Early cycle favors risk assets over bonds",
        ))

    elif vix_regime == "LOW" and curve_regime == "STEEP":
        signals.append(Signal(
            name="GOLDILOCKS - STAY INVESTED",
            direction="BULLISH",
            strength=SignalStrength.MODERATE,
            asset_class="CROSS_ASSET",
            action="Stay invested but don't chase; maintain discipline",
            rationale="Low vol + easy Fed = good environment, but not cheap",
            instruments=["Maintain equity exposure", "Sell premium strategies", "Avoid heroic bets"],
            risk_note="Good times don't last forever; don't get complacent",
        ))

    return signals


# ============================================================
# MAIN SIGNAL AGGREGATOR
# ============================================================

def get_all_signals() -> Dict:
    """Fetch all data and generate comprehensive signals."""

    print("Fetching market data...")

    # Get VIX data
    try:
        vix_stats = get_vix_stats(lookback_years=2)
        vix_regime = determine_vix_regime(vix_stats["current"])
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        vix_stats = None
        vix_regime = "NORMAL"

    # Get yield curve data
    try:
        yield_data = get_current_yields()
        yields = yield_data.get("yields", {})
        spreads = calculate_spreads(yields)
        curve_regime = determine_curve_regime(spreads)
    except Exception as e:
        print(f"Error fetching yields: {e}")
        yields = {}
        spreads = {}
        curve_regime = "NORMAL"

    # Generate signals
    all_signals = []

    if vix_stats:
        all_signals.extend(generate_vix_signals(vix_stats))

    if yields:
        all_signals.extend(generate_curve_signals(yields, spreads, curve_regime))

    all_signals.extend(generate_cross_asset_signals(vix_regime, curve_regime, vix_stats or {}))

    # Sort by strength
    all_signals.sort(key=lambda x: x.strength.value, reverse=True)

    return {
        "vix_stats": vix_stats,
        "vix_regime": vix_regime,
        "yields": yields,
        "spreads": spreads,
        "curve_regime": curve_regime,
        "signals": all_signals,
    }


def format_signal_report(data: Dict) -> str:
    """Format signals into readable report."""

    vix = data.get("vix_stats", {}).get("current", "N/A")
    vix_pct = data.get("vix_stats", {}).get("percentile", "N/A")
    vix_regime = data.get("vix_regime", "UNKNOWN")
    curve_regime = data.get("curve_regime", "UNKNOWN")
    spread_10y2y = data.get("spreads", {}).get("10Y-2Y", "N/A")

    signals = data.get("signals", [])

    # Format values safely
    vix_str = f"{vix:.2f}" if isinstance(vix, (int, float)) else str(vix)
    vix_pct_str = f"{vix_pct:.0f}" if isinstance(vix_pct, (int, float)) else str(vix_pct)
    spread_str = f"{spread_10y2y:+.2f}%" if isinstance(spread_10y2y, (int, float)) else str(spread_10y2y)

    report = f"""
################################################################################
                         ENTRY TIMING SIGNALS
                         {datetime.now().strftime("%Y-%m-%d %H:%M")}
################################################################################

================================================================================
MARKET STATE SUMMARY
================================================================================

  VIX:          {vix_str} ({vix_pct_str}th percentile)
  VIX REGIME:   {vix_regime}

  10Y-2Y:       {spread_str}
  CURVE REGIME: {curve_regime}

================================================================================
ACTIVE SIGNALS (Sorted by Strength)
================================================================================
"""

    strength_labels = {
        SignalStrength.EXTREME: "!!!! EXTREME !!!!",
        SignalStrength.STRONG: "*** STRONG ***",
        SignalStrength.MODERATE: "** MODERATE **",
        SignalStrength.WEAK: "* WEAK *",
        SignalStrength.NONE: "",
    }

    for i, signal in enumerate(signals, 1):
        direction_emoji = {
            "BULLISH": "[BULLISH]",
            "BEARISH": "[BEARISH]",
            "NEUTRAL": "[NEUTRAL]",
        }

        report += f"""
--------------------------------------------------------------------------------
{i}. {signal.name}
   {strength_labels[signal.strength]} {direction_emoji[signal.direction]}
--------------------------------------------------------------------------------
   Asset Class: {signal.asset_class}
   Action:      {signal.action}
   Rationale:   {signal.rationale}

   Instruments:
"""
        for inst in signal.instruments:
            report += f"     - {inst}\n"

        report += f"""
   Risk Note:   {signal.risk_note}
"""

    # Summary action
    strong_signals = [s for s in signals if s.strength.value >= SignalStrength.STRONG.value]
    bullish = [s for s in strong_signals if s.direction == "BULLISH"]
    bearish = [s for s in strong_signals if s.direction == "BEARISH"]

    report += """
================================================================================
BOTTOM LINE
================================================================================
"""
    if len(bullish) > len(bearish):
        report += "  BIAS: BULLISH - Strong signals favor adding risk\n"
    elif len(bearish) > len(bullish):
        report += "  BIAS: BEARISH - Strong signals favor reducing risk\n"
    else:
        report += "  BIAS: NEUTRAL - No dominant signal direction\n"

    report += f"""
  Strong Bullish Signals: {len(bullish)}
  Strong Bearish Signals: {len(bearish)}

  For aggressive profile: Focus on EXTREME and STRONG signals.
  Current environment: {vix_regime} VIX + {curve_regime} curve

################################################################################
"""
    return report


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_entry_timing_signals():
    """Run the complete timing signal analysis."""

    data = get_all_signals()
    report = format_signal_report(data)
    print(report)

    return data


# ============================================================
# ALERT THRESHOLDS (for automated monitoring)
# ============================================================

ALERT_THRESHOLDS = {
    "vix_spike_alert": 25,      # Alert when VIX crosses above
    "vix_extreme_alert": 30,    # Extreme alert
    "vix_low_alert": 13,        # Alert when VIX drops below
    "curve_inversion": 0,       # Alert when 10Y-2Y goes negative
    "curve_steep": 1.5,         # Alert when curve steepens significantly
}


def check_alerts(data: Dict) -> List[str]:
    """Check for alert conditions."""
    alerts = []

    vix = data.get("vix_stats", {}).get("current", 0)
    spread = data.get("spreads", {}).get("10Y-2Y", 0)

    if vix >= ALERT_THRESHOLDS["vix_extreme_alert"]:
        alerts.append(f"EXTREME ALERT: VIX at {vix:.1f} - CRISIS LEVEL")
    elif vix >= ALERT_THRESHOLDS["vix_spike_alert"]:
        alerts.append(f"ALERT: VIX spiked to {vix:.1f} - Consider buying dip")

    if vix <= ALERT_THRESHOLDS["vix_low_alert"]:
        alerts.append(f"ALERT: VIX at {vix:.1f} - Complacency zone")

    if spread < ALERT_THRESHOLDS["curve_inversion"]:
        alerts.append(f"ALERT: Curve INVERTED at {spread:.2f}% - Recession signal")

    if spread > ALERT_THRESHOLDS["curve_steep"]:
        alerts.append(f"ALERT: Curve STEEP at {spread:.2f}% - Risk-on signal")

    return alerts


if __name__ == "__main__":
    data = run_entry_timing_signals()

    # Check for any alerts
    alerts = check_alerts(data)
    if alerts:
        print("\n" + "=" * 60)
        print("ACTIVE ALERTS")
        print("=" * 60)
        for alert in alerts:
            print(f"  {alert}")
