# PortfolioSpec Specification

## Overview
A structured Python dataclass stored in `st.session_state` that captures everything the user told the AI advisor — goals, preferences, and constraints — and makes it available to every tab. It is the single source of truth between the conversation and the portfolio engine.

## User Flows

### Flow 1: Advisor populates PortfolioSpec from chat
1. User says: "I have $100k, I like TSLA, moderate risk, scared of drawdowns"
2. AI extracts and writes to PortfolioSpec:
   - budget: 100000
   - risk_level: "moderate"
   - preferred_tickers: ["TSLA"]
   - max_drawdown: (inferred from "scared") → 0.15
3. Allocator reads PortfolioSpec — TSLA gets meaningful weight, not a token 10%

### Flow 2: User updates a preference mid-conversation
1. User says: "Actually add QQQ too, and no bonds"
2. AI updates PortfolioSpec: preferred_tickers += ["QQQ"], bond_pct = 0
3. Portfolio is regenerated respecting new spec

### Flow 3: Any tab can read the spec
1. Performance tab shows: "Your preference: TSLA ≥ 15% — current: 17% ✅"
2. Signals tab filters strategies to match risk_level from PortfolioSpec
3. Rebalancing respects preferred_tickers (never sells them)

## Key Information

**Fields:**
- `budget` — total investable amount
- `risk_level` — very_conservative / conservative / moderate / aggressive / very_aggressive
- `preferred_tickers` — list of tickers user explicitly requested (get priority weight)
- `excluded_tickers` — tickers user said to avoid
- `max_drawdown` — max acceptable portfolio drawdown (optional, inferred or stated)
- `return_target` — target annual return (optional)
- `horizon` — investment horizon in years (optional)
- `bond_pct` — explicit bond allocation override (optional)
- `treasury_pct` — explicit treasury allocation override (optional)
- `regime_tilts_allowed` — whether regime overlays can adjust weights (default True)

## Out of Scope
- Options positions (handled separately in Options Builder)
- Tax optimization
- Account type / brokerage integration
- Full Markowitz optimization (that's Section 2)
