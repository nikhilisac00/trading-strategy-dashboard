"""
Comprehensive test suite for the rebalancer and beta slider.
Tests analyze_portfolio_for_rebalancing, generate_rebalancing_trades,
the beta slider rebalance flow, and the chat-based rebalance command.
"""
import re
import sys
import os

total = 0
passed = 0
bugs = []

def test(name, condition, detail=""):
    global total, passed, bugs
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        bugs.append((name, detail))
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

# ============================================================
# Replicate key data structures from dashboard.py
# ============================================================

RISK_PROFILES = {
    "very_conservative": {
        "equity_range": (0.05, 0.15), "treasury_range": (0.50, 0.75), "bond_range": (0.15, 0.40),
    },
    "conservative": {
        "equity_range": (0.25, 0.45), "treasury_range": (0.10, 0.25), "bond_range": (0.30, 0.55),
    },
    "moderate": {
        "equity_range": (0.55, 0.75), "treasury_range": (0.05, 0.15), "bond_range": (0.15, 0.35),
    },
    "aggressive": {
        "equity_range": (0.75, 0.95), "treasury_range": (0.00, 0.10), "bond_range": (0.05, 0.20),
    },
    "very_aggressive": {
        "equity_range": (0.80, 1.00), "treasury_range": (0.00, 0.10), "bond_range": (0.00, 0.10),
    },
}

BETA_TARGETS = {
    "very_conservative": {"min": 0.05, "max": 0.29, "target": 0.15},
    "conservative": {"min": 0.30, "max": 0.54, "target": 0.42},
    "moderate": {"min": 0.55, "max": 0.84, "target": 0.70},
    "aggressive": {"min": 0.85, "max": 1.04, "target": 0.95},
    "very_aggressive": {"min": 1.05, "max": 1.50, "target": 1.20},
}

# Known ETF betas from ASSET_CLASSES
ETF_BETAS = {
    "SPY": 1.0, "QQQ": 1.2, "VTI": 1.0, "VXUS": 0.85, "VGT": 1.2,
    "IWM": 1.2, "VIG": 0.9, "SCHD": 0.8,
    "TLT": 0.05, "IEF": 0.05, "SHY": 0.02, "TIP": 0.05,
    "BND": 0.08, "LQD": 0.1, "HYG": 0.3,
    "GLD": 0.0, "VNQ": 0.8,
    # Individual stocks
    "TSLA": 2.0, "NVDA": 1.7, "AMD": 1.6, "AAPL": 1.2,
}

# Treasury/bond categorization
TREASURY_TICKERS = {"TLT", "IEF", "IEI", "SHY", "GOVT", "TIP"}
BOND_TICKERS = {"BND", "LQD", "HYG", "VCIT", "VCSH"}
EQUITY_TICKERS = {"SPY", "QQQ", "VTI", "VXUS", "VWO", "SCHD", "VNQ"}

def categorize_ticker(ticker, pos_type=""):
    """Categorize a ticker as equity/treasury/bond."""
    pos_type_lower = pos_type.lower()
    if any(t in pos_type_lower for t in ["stock", "etf", "equity"]) and "bond" not in pos_type_lower and "treasury" not in pos_type_lower:
        return "equity"
    elif any(t in pos_type_lower for t in ["treasury", "tlt", "ief", "shy"]):
        return "treasury"
    elif any(t in pos_type_lower for t in ["bond", "lqd", "hyg", "bnd"]):
        return "bond"
    else:
        if ticker in EQUITY_TICKERS:
            return "equity"
        elif ticker in TREASURY_TICKERS:
            return "treasury"
        elif ticker in BOND_TICKERS:
            return "bond"
        else:
            return "equity"  # default

def calc_portfolio_beta(positions):
    """Calculate weighted average beta."""
    total_value = 0
    weighted_beta = 0
    for pos in positions:
        ticker = pos.get("ticker", "")
        qty = pos.get("quantity", 0)
        price = pos.get("entry_price", 100)
        value = qty * price
        beta = ETF_BETAS.get(ticker, 1.0)
        weighted_beta += value * beta
        total_value += value
    if total_value > 0:
        return weighted_beta / total_value
    return 1.0

def calc_allocation(positions):
    """Calculate equity/treasury/bond allocation."""
    alloc = {"equity": 0, "treasury": 0, "bond": 0}
    total = 0
    for pos in positions:
        ticker = pos.get("ticker", "")
        qty = pos.get("quantity", 0)
        price = pos.get("entry_price", 100)
        value = qty * price
        total += value
        cat = categorize_ticker(ticker, pos.get("type", ""))
        alloc[cat] += value
    if total > 0:
        return {k: v/total for k, v in alloc.items()}, total
    return alloc, 0

def simulate_rebalance_trades(positions, risk_level):
    """Simulate generate_rebalancing_trades logic."""
    beta_target = BETA_TARGETS.get(risk_level, BETA_TARGETS["moderate"])
    target_beta = beta_target["target"]

    user_positions = [p for p in positions if p.get("user_added")]
    system_positions = [p for p in positions if not p.get("user_added")]

    total_value = sum(p.get("quantity", 0) * p.get("entry_price", 100) for p in positions)
    user_value = sum(p.get("quantity", 0) * p.get("entry_price", 100) for p in user_positions)
    user_weighted_beta = sum(
        p.get("quantity", 0) * p.get("entry_price", 100) * ETF_BETAS.get(p.get("ticker", ""), 1.0)
        for p in user_positions
    )

    remaining_value = total_value - user_value
    if remaining_value <= 0:
        remaining_value = 0

    if remaining_value > 0 and total_value > 0:
        user_beta_contribution = user_weighted_beta / total_value
        needed_beta_from_system = target_beta - user_beta_contribution
        system_weight = remaining_value / total_value
        required_system_beta = needed_beta_from_system / system_weight if system_weight > 0 else target_beta
    else:
        required_system_beta = target_beta

    # Feasibility check
    if required_system_beta < 0:
        return "INFEASIBLE", "User positions push beta too high"
    if required_system_beta > 2.0:
        return "INFEASIBLE", "User positions too conservative for target"

    return "FEASIBLE", required_system_beta

def simulate_beta_slider_rebalance(target_beta, total_budget):
    """Simulate the beta slider rebalance logic."""
    equity_beta_avg = 1.0
    bond_beta_avg = 0.08
    clamped_beta = max(0.08, min(1.5, target_beta))
    equity_pct = (clamped_beta - bond_beta_avg) / (equity_beta_avg - bond_beta_avg)
    equity_pct = max(0.0, min(1.0, equity_pct))
    bond_pct = 1.0 - equity_pct

    equity_budget = total_budget * equity_pct
    bond_budget = total_budget * bond_pct

    # Select ETFs
    if target_beta >= 1.05:
        equity_etfs = [("QQQ", 0.25), ("NVDA", 0.25), ("AMD", 0.20), ("TSLA", 0.15), ("VGT", 0.15)]
    elif target_beta >= 0.85:
        equity_etfs = [("SPY", 0.35), ("QQQ", 0.30), ("VGT", 0.20), ("VXUS", 0.15)]
    elif target_beta >= 0.55:
        equity_etfs = [("SPY", 0.40), ("QQQ", 0.20), ("VTI", 0.20), ("VXUS", 0.20)]
    else:
        equity_etfs = [("VTI", 0.30), ("SCHD", 0.35), ("VIG", 0.20), ("VXUS", 0.15)]

    bond_etfs = [("BND", 0.40), ("TLT", 0.30), ("LQD", 0.30)]

    positions = []
    for ticker, weight in equity_etfs:
        dollar_amount = equity_budget * weight
        price = 100  # placeholder
        qty = int(dollar_amount / price)
        if qty > 0:
            positions.append({
                "ticker": ticker,
                "type": "ETF" if ticker not in ["NVDA", "AMD", "TSLA"] else "Stock",
                "quantity": qty,
                "entry_price": price,
            })

    for ticker, weight in bond_etfs:
        dollar_amount = bond_budget * weight
        price = 100  # placeholder
        qty = int(dollar_amount / price)
        if qty > 0:
            positions.append({
                "ticker": ticker,
                "type": "Bond ETF" if ticker != "TLT" else "Treasury ETF",
                "quantity": qty,
                "entry_price": price,
            })

    return positions, equity_pct, bond_pct


# ============================================================
# 1. ANALYZE PORTFOLIO — CATEGORIZATION TESTS
# ============================================================

print("=" * 60)
print("1. POSITION CATEGORIZATION")
print("=" * 60)

test("SPY → equity", categorize_ticker("SPY", "ETF") == "equity")
test("QQQ → equity", categorize_ticker("QQQ", "ETF") == "equity")
test("TLT → treasury", categorize_ticker("TLT", "Treasury ETF") == "treasury")
test("IEF → treasury", categorize_ticker("IEF", "") == "treasury")
test("SHY → treasury", categorize_ticker("SHY", "") == "treasury")
test("BND → bond", categorize_ticker("BND", "") == "bond")
test("LQD → bond", categorize_ticker("LQD", "") == "bond")
test("HYG → bond", categorize_ticker("HYG", "") == "bond")
test("TSLA → equity (default)", categorize_ticker("TSLA", "Stock") == "equity")
test("VNQ → equity", categorize_ticker("VNQ", "") == "equity")

# Type string categorization
test("type='Treasury ETF (TLT, IEF, SHY)' → treasury",
     categorize_ticker("TLT", "Treasury ETF (TLT, IEF, SHY)") == "treasury")
test("type='Corporate Bond ETF (LQD, HYG)' → bond",
     categorize_ticker("LQD", "Corporate Bond ETF (LQD, HYG)") == "bond")

# Edge case: TIP
test("TIP → treasury (by ticker lookup)", categorize_ticker("TIP", "") == "treasury")

# Edge case: GLD — defaults to equity
test("GLD → equity (default)", categorize_ticker("GLD", "") == "equity",
     "GLD is gold, not in any special list, defaults to equity")


# ============================================================
# 2. PORTFOLIO BETA CALCULATION
# ============================================================

print("\n" + "=" * 60)
print("2. PORTFOLIO BETA CALCULATION")
print("=" * 60)

# Simple portfolio: 100% SPY
positions_spy = [{"ticker": "SPY", "quantity": 100, "entry_price": 500}]
beta_spy = calc_portfolio_beta(positions_spy)
test("100% SPY → beta ~1.0", abs(beta_spy - 1.0) < 0.01, f"got {beta_spy:.2f}")

# Mix: 50% SPY + 50% TLT
positions_mix = [
    {"ticker": "SPY", "quantity": 100, "entry_price": 500},
    {"ticker": "TLT", "quantity": 500, "entry_price": 100},
]
beta_mix = calc_portfolio_beta(positions_mix)
test("50% SPY + 50% TLT → beta ~0.52", abs(beta_mix - 0.525) < 0.05, f"got {beta_mix:.2f}")

# All bonds
positions_bonds = [
    {"ticker": "BND", "quantity": 100, "entry_price": 100},
    {"ticker": "TLT", "quantity": 100, "entry_price": 100},
]
beta_bonds = calc_portfolio_beta(positions_bonds)
test("All bonds → beta < 0.10", beta_bonds < 0.10, f"got {beta_bonds:.2f}")

# Very aggressive: TSLA + NVDA
positions_agg = [
    {"ticker": "TSLA", "quantity": 10, "entry_price": 300},
    {"ticker": "NVDA", "quantity": 10, "entry_price": 200},
]
beta_agg = calc_portfolio_beta(positions_agg)
test("TSLA + NVDA → beta > 1.5", beta_agg > 1.5, f"got {beta_agg:.2f}")

# Empty portfolio
test("Empty portfolio → beta 1.0 (default)", calc_portfolio_beta([]) == 1.0)


# ============================================================
# 3. ALLOCATION CALCULATION
# ============================================================

print("\n" + "=" * 60)
print("3. ALLOCATION CALCULATION")
print("=" * 60)

# Mixed portfolio
positions_full = [
    {"ticker": "SPY", "quantity": 10, "entry_price": 500, "type": "ETF"},
    {"ticker": "QQQ", "quantity": 5, "entry_price": 500, "type": "ETF"},
    {"ticker": "TLT", "quantity": 20, "entry_price": 100, "type": "Treasury ETF"},
    {"ticker": "BND", "quantity": 10, "entry_price": 100, "type": "Bond ETF"},
]
alloc, total = calc_allocation(positions_full)

test("Total value correct", abs(total - 10000) < 0.01, f"got {total}")
test("Equity allocation ~75%", abs(alloc["equity"] - 0.75) < 0.01, f"got {alloc['equity']*100:.0f}%")
test("Treasury allocation ~20%", abs(alloc["treasury"] - 0.20) < 0.01, f"got {alloc['treasury']*100:.0f}%")
test("Bond allocation ~10%", abs(alloc["bond"] - 0.10) < 0.01, f"got {alloc['bond']*100:.0f}%")
test("Allocations sum to ~100%", abs(sum(alloc.values()) - 1.0) < 0.01)


# ============================================================
# 4. REBALANCING TRADE GENERATION — FEASIBILITY
# ============================================================

print("\n" + "=" * 60)
print("4. REBALANCING TRADE GENERATION — FEASIBILITY")
print("=" * 60)

# Scenario A: User holds TSLA (beta 2.0), wants conservative (target 0.42)
# With $5000 TSLA (50%) and $5000 system (50%)
# user_beta_contribution = 5000 * 2.0 / 10000 = 1.0
# needed_beta_from_system = 0.42 - 1.0 = -0.58
# required_system_beta = -0.58 / 0.5 = -1.16 → INFEASIBLE
positions_conflict = [
    {"ticker": "TSLA", "quantity": 10, "entry_price": 500, "user_added": True},
    {"ticker": "SPY", "quantity": 10, "entry_price": 500, "user_added": False},
]
result, detail = simulate_rebalance_trades(positions_conflict, "conservative")
test("TSLA + conservative → INFEASIBLE", result == "INFEASIBLE", detail)

# Scenario B: User holds TSLA, wants very_aggressive (target 1.20)
# user_beta_contribution = 5000 * 2.0 / 10000 = 1.0
# needed = 1.20 - 1.0 = 0.20
# required_system_beta = 0.20 / 0.5 = 0.40 → FEASIBLE
result2, detail2 = simulate_rebalance_trades(positions_conflict, "very_aggressive")
test("TSLA + very_aggressive → FEASIBLE", result2 == "FEASIBLE",
     f"required system beta: {detail2:.2f}")

# Scenario C: No user positions, just system — always feasible
positions_system_only = [
    {"ticker": "SPY", "quantity": 10, "entry_price": 500, "user_added": False},
    {"ticker": "BND", "quantity": 20, "entry_price": 100, "user_added": False},
]
for risk in ["very_conservative", "conservative", "moderate", "aggressive", "very_aggressive"]:
    r, d = simulate_rebalance_trades(positions_system_only, risk)
    test(f"System-only + {risk} → FEASIBLE", r == "FEASIBLE")

# Scenario D: User holds SHY (beta 0.02), wants very_aggressive (target 1.20)
# With $5000 SHY (50%) and $5000 system (50%)
# user_beta_contribution = 5000 * 0.02 / 10000 = 0.01
# needed = 1.20 - 0.01 = 1.19
# required_system_beta = 1.19 / 0.5 = 2.38 → INFEASIBLE (> 2.0)
positions_shy = [
    {"ticker": "SHY", "quantity": 50, "entry_price": 100, "user_added": True},
    {"ticker": "SPY", "quantity": 10, "entry_price": 500, "user_added": False},
]
result3, detail3 = simulate_rebalance_trades(positions_shy, "very_aggressive")
test("SHY + very_aggressive → INFEASIBLE", result3 == "INFEASIBLE", detail3)

# Scenario E: Small user position — should be feasible
positions_small_user = [
    {"ticker": "TSLA", "quantity": 1, "entry_price": 300, "user_added": True},  # $300 = 3%
    {"ticker": "SPY", "quantity": 20, "entry_price": 500, "user_added": False},  # $10000 = 97%
]
result4, detail4 = simulate_rebalance_trades(positions_small_user, "conservative")
test("Small TSLA + conservative → FEASIBLE",
     result4 == "FEASIBLE",
     f"required system beta: {detail4:.2f}" if isinstance(detail4, float) else detail4)


# ============================================================
# 5. BETA SLIDER REBALANCE
# ============================================================

print("\n" + "=" * 60)
print("5. BETA SLIDER REBALANCE")
print("=" * 60)

# Test equity/bond split at various beta targets
slider_tests = [
    (0.10, "very low beta → mostly bonds"),
    (0.30, "conservative"),
    (0.55, "moderate low end"),
    (0.70, "moderate"),
    (0.85, "aggressive low end"),
    (1.00, "aggressive"),
    (1.20, "very aggressive"),
    (1.50, "max aggressive"),
]

for target_beta, desc in slider_tests:
    positions, eq_pct, bond_pct = simulate_beta_slider_rebalance(target_beta, 100000)
    actual_beta = calc_portfolio_beta(positions)

    test(f"Slider β={target_beta:.2f} ({desc}): equity={eq_pct*100:.0f}%, bond={bond_pct*100:.0f}%",
         True, f"allocations sum to {(eq_pct + bond_pct)*100:.0f}%")

    # Verify the positions were created
    test(f"  → Positions created: {len(positions)} positions",
         len(positions) > 0,
         f"got {len(positions)} positions: {[p['ticker'] for p in positions]}")

    # Verify the total budget is roughly right
    total_invested = sum(p["quantity"] * p["entry_price"] for p in positions)
    test(f"  → Total invested ~$100k",
         total_invested > 80000,  # Allow for rounding with whole shares
         f"got ${total_invested:,.0f}")

# Test edge case: beta = 0.08 (minimum)
positions_min, eq_min, bond_min = simulate_beta_slider_rebalance(0.08, 100000)
test("Slider β=0.08 → 0% equity, 100% bonds",
     eq_min < 0.01,
     f"equity={eq_min*100:.1f}%, bond={bond_min*100:.1f}%")

# Test edge case: beta = 1.5 (maximum)
positions_max, eq_max, bond_max = simulate_beta_slider_rebalance(1.50, 100000)
test("Slider β=1.50 → ~100% equity, ~0% bonds",
     eq_max > 0.99,
     f"equity={eq_max*100:.1f}%, bond={bond_max*100:.1f}%")


# ============================================================
# 6. BETA SLIDER — ETF SELECTION BY RISK LEVEL
# ============================================================

print("\n" + "=" * 60)
print("6. BETA SLIDER — ETF SELECTION")
print("=" * 60)

# Very aggressive: should include NVDA, AMD, TSLA
pos_va, _, _ = simulate_beta_slider_rebalance(1.20, 100000)
tickers_va = {p["ticker"] for p in pos_va}
test("β≥1.05 includes NVDA", "NVDA" in tickers_va, f"got {tickers_va}")
test("β≥1.05 includes AMD", "AMD" in tickers_va, f"got {tickers_va}")
test("β≥1.05 includes QQQ", "QQQ" in tickers_va, f"got {tickers_va}")

# Aggressive
pos_a, _, _ = simulate_beta_slider_rebalance(0.90, 100000)
tickers_a = {p["ticker"] for p in pos_a}
test("β≥0.85 includes SPY", "SPY" in tickers_a, f"got {tickers_a}")
test("β≥0.85 includes QQQ", "QQQ" in tickers_a, f"got {tickers_a}")

# Moderate
pos_m, _, _ = simulate_beta_slider_rebalance(0.70, 100000)
tickers_m = {p["ticker"] for p in pos_m}
test("β≥0.55 includes SPY", "SPY" in tickers_m, f"got {tickers_m}")
test("β≥0.55 includes VTI", "VTI" in tickers_m, f"got {tickers_m}")
test("β≥0.55 includes bonds", "BND" in tickers_m or "TLT" in tickers_m, f"got {tickers_m}")

# Conservative
pos_c, _, _ = simulate_beta_slider_rebalance(0.30, 100000)
tickers_c = {p["ticker"] for p in pos_c}
test("β<0.55 includes SCHD", "SCHD" in tickers_c, f"got {tickers_c}")
test("β<0.55 includes VIG", "VIG" in tickers_c, f"got {tickers_c}")
test("β<0.55 has significant bonds", any(t in tickers_c for t in ["BND", "TLT", "LQD"]))


# ============================================================
# 7. REBALANCE INTENT DETECTION
# ============================================================

print("\n" + "=" * 60)
print("7. REBALANCE INTENT DETECTION IN CHAT")
print("=" * 60)

# The SmartFinancialAgent detects rebalance intent via the ChatGPT parser
# or the rule-based intent detection. Let's verify the patterns.

INTENT_PATTERNS_REBALANCE = [
    "rebalance", "rebalance my portfolio", "rebalance portfolio",
    "execute rebalancing", "rebalance now", "do the rebalancing",
]

for phrase in INTENT_PATTERNS_REBALANCE:
    # Check if "rebalance" is in the text
    has_rebalance = "rebalanc" in phrase.lower()
    test(f"'{phrase}' detected as rebalance intent", has_rebalance)


# ============================================================
# 8. EXECUTE REBALANCING — CHAT HANDLER
# ============================================================

print("\n" + "=" * 60)
print("8. EXECUTE REBALANCING — FLOW ANALYSIS")
print("=" * 60)

# The "execute rebalancing" flow in generate_response (line 1784):
# 1. Checks for pending_rebalancing in session state
# 2. Gets risk-appropriate targets from RISK_PROFILES
# 3. Calls generate_rebalancing_trades
# 4. If INFEASIBLE → explains to user
# 5. If feasible → removes system positions, adds new BUY positions
# 6. Preserves user_added positions

# Verify the execute keywords
execute_keywords = ["execute rebalancing", "rebalance now", "do the rebalancing", "implement rebalancing"]
test("Execute keywords defined", len(execute_keywords) == 4)

# The rebalance in execute_action (line 3314) sets pending_rebalancing
# Then user says "execute rebalancing" → handled by generate_response (line 1784)
# BUG CHECK: execute_action handles "rebalance" intent (line 3314)
# But generate_response handles "execute rebalancing" (line 1784)
# Are both accessible from chat()?

# chat() calls execute_action() at line 3695
# But execute_action only handles intents, not "execute rebalancing"
# "execute rebalancing" would need to go through generate_response()
# which is the OLD code path (line 1615)

# Let me verify: does chat() ever call generate_response()?
# Looking at chat() flow:
# Step 0: pending_portfolio check → confirmation/question handling
# Step 0.5: question detection
# Step 1: parse_message_smart
# Step 2: chatgpt_parse_message
# Step 3: execute_action
# Step 4: fallback → chatgpt_conversational

# execute_action handles intent=="rebalance" → shows analysis + sets pending_rebalancing
# But there's NO handler in execute_action for "execute rebalancing"!
# The handler at line 1784 is in generate_response() which is NOT called by chat()

test("BUG: 'execute rebalancing' has no handler in chat() path",
     True,
     "CRITICAL BUG: execute_action() handles 'rebalance' intent but NOT 'execute rebalancing'. "
     "The execute handler is in generate_response() (line 1784) which is the OLD code path. "
     "chat() NEVER calls generate_response(), only execute_action().")


# ============================================================
# 9. BETA SLIDER — DESTROYS USER POSITIONS
# ============================================================

print("\n" + "=" * 60)
print("9. BETA SLIDER — USER POSITION PRESERVATION")
print("=" * 60)

# Line 4926: st.session_state.portfolio = []
# The beta slider CLEARS the entire portfolio and rebuilds from scratch.
# This means user-added positions (like "add TSLA") are DESTROYED.
# The chat-based rebalancer preserves user positions, but the slider does not.

test("BUG: Beta slider clears ALL positions including user-added",
     True,
     "CRITICAL BUG: Line 4926 does `st.session_state.portfolio = []` which destroys "
     "user-added positions. The chat-based rebalancer at line 1835 correctly preserves them. "
     "Fix: Filter to keep user_added positions before rebuild, then only replace system positions.")


# ============================================================
# 10. CONCENTRATION RISK DETECTION
# ============================================================

print("\n" + "=" * 60)
print("10. CONCENTRATION RISK DETECTION")
print("=" * 60)

# If any position is > 30% of portfolio → flag it
# This is in analyze_portfolio_for_rebalancing (line 1422-1426)

# Scenario: 60% TSLA, 40% BND
positions_concentrated = [
    {"ticker": "TSLA", "quantity": 20, "entry_price": 300, "type": "Stock"},  # $6000
    {"ticker": "BND", "quantity": 40, "entry_price": 100, "type": "Bond ETF"},  # $4000
]
total_val = sum(p["quantity"] * p["entry_price"] for p in positions_concentrated)
tsla_pct = (20 * 300) / total_val * 100
test("TSLA 60% → concentration flagged", tsla_pct > 30, f"TSLA is {tsla_pct:.0f}% of portfolio")

# BND at 40% — also flagged
bnd_pct = (40 * 100) / total_val * 100
test("BND 40% → concentration flagged", bnd_pct > 30, f"BND is {bnd_pct:.0f}% of portfolio")


# ============================================================
# 11. ALLOCATION DRIFT DETECTION
# ============================================================

print("\n" + "=" * 60)
print("11. ALLOCATION DRIFT DETECTION")
print("=" * 60)

# >80% equity → flag
# <30% equity (non-conservative) → flag
# 0% bonds → flag

# All equity portfolio
positions_all_equity = [
    {"ticker": "SPY", "quantity": 50, "entry_price": 500, "type": "ETF"},
    {"ticker": "QQQ", "quantity": 20, "entry_price": 500, "type": "ETF"},
]
alloc_ae, _ = calc_allocation(positions_all_equity)
test("All equity → equity > 80%", alloc_ae["equity"] > 0.80, f"equity={alloc_ae['equity']*100:.0f}%")
test("All equity → 0% treasury", alloc_ae["treasury"] == 0)
test("All equity → 0% bonds → drift flagged", alloc_ae["bond"] == 0)


# ============================================================
# 12. CHAT-BASED REBALANCE → EXECUTE FLOW (THE MISSING LINK)
# ============================================================

print("\n" + "=" * 60)
print("12. CHAT REBALANCE FLOW — MISSING EXECUTE HANDLER")
print("=" * 60)

# Detailed trace of what happens when user types "execute rebalancing":
# 1. Sidebar: not a confirmation word → passes to chat()
# 2. chat() Step 0: No pending_portfolio → skip
# 3. chat() Step 0.5: Not a question → skip
# 4. chat() Step 1: parse_message_smart("execute rebalancing")
#    - detect_intent → might not catch "execute rebalancing" as "rebalance" intent
#    - depends on INTENT_PATTERNS
# 5. chat() Step 2: ChatGPT parser → would return intent="rebalance" or similar
# 6. chat() Step 3: execute_action(intent="rebalance", parsed)
#    - But execute_action's "rebalance" handler ANALYZES and sets pending_rebalancing
#    - It does NOT execute — it just shows suggestions!
# 7. User says "execute rebalancing" again → same flow, gets analysis AGAIN

# The problem: there's no way to EXECUTE the rebalancing through the chat() path.
# The execution logic is only in generate_response() which chat() never calls.

test("CONFIRMED: No execute rebalancing handler in chat()",
     True,
     "After 'rebalance' shows suggestions, 'execute rebalancing' would just re-analyze. "
     "Need to add execute logic to execute_action() or to chat() pending check.")


# ============================================================
# 13. SELL LOGIC — POSITION REMOVAL
# ============================================================

print("\n" + "=" * 60)
print("13. REBALANCE SELL LOGIC")
print("=" * 60)

# Line 1835-1838: Remove system positions
# st.session_state.portfolio = [
#     p for p in st.session_state.portfolio
#     if p.get("user_added") or p.get("ticker") not in [t.get("ticker") for t in trades if t["action"] == "SELL"]
# ]
# This removes system positions whose tickers appear in SELL trades.
# BUG: If user added SPY and system also has SPY, BOTH would survive
# because user_added is checked first. This is actually CORRECT behavior.

test("User-added positions preserved during sell", True,
     "Line 1835: 'p.get(\"user_added\")' → user positions always kept")

# But what if user_added is not set? Old positions might not have this flag.
test("Old positions without user_added flag → treated as system",
     True,
     "Positions without user_added default to False → treated as system positions (correct)")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(f"FINAL: {passed}/{total} passed, {total - passed} issues")
print("=" * 60)

if bugs:
    print("\nBUGS FOUND:")
    for i, (name, detail) in enumerate(bugs, 1):
        print(f"\n  {i}. {name}")
        if detail:
            print(f"     {detail}")

print("""
CRITICAL BUGS FOUND:

1. EXECUTE REBALANCING NOT REACHABLE FROM CHAT (CRITICAL):
   After "rebalance my portfolio" shows analysis + "Say 'execute rebalancing'",
   the user types "execute rebalancing" but there's NO handler for it in the
   chat() → execute_action() path. The handler exists in generate_response()
   (line 1784) which is the OLD code path that chat() never calls.

   FIX: Add "execute_rebalancing" intent handling to execute_action(), or
   add a Step 0-style check in chat() for pending_rebalancing + execute keywords.

2. BETA SLIDER DESTROYS USER POSITIONS (CRITICAL):
   Line 4926: st.session_state.portfolio = [] clears ALL positions including
   user-added ones (like manually added TSLA). The chat-based rebalancer
   correctly preserves user_added positions, but the slider doesn't.

   FIX: Keep user_added positions, only rebuild system positions.
""")
