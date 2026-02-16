---
marp: true
title: "Case 2: ChatGPT Launch 2022"
subtitle: "When does a new game begin?"
author: "MGMT 69000: Mastering AI for Finance"
theme: default
paginate: true
---

# Case 2: ChatGPT Launch 2022

## "When does a new game begin?"

**Week 3 | Entropy Concept: Sample Space Expansion**
**Scaffolding Level: HEAVY**

MGMT 69000: Mastering AI for Finance
Purdue University | Spring 2026

---

# The Regime That Ended

What was "normal" before November 30, 2022?

- **AI as niche/research technology** — Enterprise software, not consumer phenomenon
- **Human knowledge work as non-automatable** — Homework help, tutoring, analysis presumed safe
- **Tech sector in bear market** — S&P 500 down 25%, Nvidia down 70% from peak
- **Nvidia as "gaming chip" company** — Data center secondary to gaming revenue

> The old game: AI existed, but not as an investable theme.

---

# Key Timeline

| Date | Event | Market Impact |
|------|-------|---------------|
| Oct 12, 2022 | S&P 500 hits post-COVID bottom | Market down 25% from January peak |
| **Nov 30, 2022** | **ChatGPT launches** | **100M users in 2 months** |
| May 2, 2023 | Chegg CEO admits ChatGPT impact | Stock crashes 49% in single day |
| May 24, 2023 | Nvidia blowout earnings | AI chip demand "blows away" forecasts |
| Nov 2024 | Two-year anniversary | Nvidia +700%, Chegg -99% |

---

# What Happened: Creative Destruction

| Metric | Nov 2022 | 2025 | Change |
|--------|----------|------|--------|
| Nvidia market cap | $345B | $4.4T | **+1,175%** |
| Nvidia revenue (annual) | $27B | $130B | **+381%** |
| OpenAI valuation | $14B | $500B | **+3,471%** |
| Chegg stock price | $25 | $0.25 | **-99%** |
| Mag 7 S&P 500 weight | ~20% | ~32% | **+60%** |

**Keith Lerner, CIO Truist Advisory Services:**
> "The dominant theme of this bull market is technology and AI, and it really kicked off with ChatGPT."

---

# The Creative Destruction Map

```
ChatGPT Launch (Nov 2022)
    │
    ├──► AI Infrastructure CREATION
    │    ├── Nvidia: +700% (GPU demand)
    │    ├── Microsoft: Azure AI platform
    │    └── Data centers, energy demand
    │
    ├──► Knowledge Work DESTRUCTION
    │    ├── Chegg: -99% (homework help obsolete)
    │    ├── Traditional tutoring services
    │    └── Entry-level analyst tasks
    │
    └──► Second-Order Effects
         ├── Energy stocks: AI data center power
         ├── Real estate: Data center REITs
         └── Education: Curriculum restructuring
```

---

# Entropy Concept: Sample Space Expansion

**Week 1 (Tariff Shock):** Transition probabilities changed (P₁ → P₂)
- Same assets, new dynamics

**Week 3 (ChatGPT):** The sample space itself expanded (X₁ → X₂)
- **New assets entered that didn't exist before**

| Dimension | Regime Shift (Tariff) | Sample Space Expansion (ChatGPT) |
|-----------|----------------------|----------------------------------|
| **What changes** | Transition probabilities | The investment universe itself |
| **Example** | Tariff probabilities | "AI infrastructure" becomes asset class |
| **Recovery** | Possible (policy reversal) | Usually impossible (new game) |

---

# Why This Is "Sample Space Expansion"

**Before ChatGPT:**
- Portfolio allocation: Stocks, Bonds, Cash, Real Estate
- Tech sector was a **sector**, not a dominant theme
- "AI exposure" was not a standard investment consideration

**After ChatGPT:**
- "AI exposure" became a **required** allocation question
- Magnificent 7 went from 20% to 32% of S&P 500
- New investment category: "AI infrastructure"

> The investment universe itself changed. New states entered that didn't previously exist.

---

# Measuring Sample Space Expansion

**Sector Entropy:** How concentrated is the market?

```python
def sector_entropy(weights):
    """Shannon entropy of sector weights."""
    weights = np.array(weights)
    weights = weights[weights > 0]
    return -np.sum(weights * np.log2(weights))

# Lower entropy = more concentrated
# Mag 7 dominance = entropy decreased (concentration increased)
```

**The Paradox:** Sample space expanded (new category), but market concentration increased (Mag 7 dominance).

---

# Comparing Week 1 vs Week 3

| Dimension | Week 1: Tariff Shock | Week 3: ChatGPT |
|-----------|---------------------|-----------------|
| **Type** | Policy shock | Technology shock |
| **Entropy concept** | Textual entropy, QEWS | Sample space expansion |
| **What changed** | Trade probabilities | Investment universe |
| **Reversibility** | Possible (new admin) | Unlikely (new game) |
| **Winners/Losers** | FX, Trade-exposed | AI infra vs. knowledge work |
| **Detection signal** | Novel policy language | New asset class emergence |

---

# DRIVER Application for This Case

| Stage | Your Task |
|-------|-----------|
| **D** Discover | What sectors were most affected by ChatGPT? |
| **R** Represent | Map the disruption cascade (winners → losers) |
| **I** Implement | Calculate sector entropy before/after ChatGPT |
| **V** Validate | Does entropy measure capture the shift? |
| **E** Evolve | Can this detect future "new game" events? |
| **R** Reflect | How is sample space expansion different from regime shift? |

---

# Student Deliverable

**Task:** Map the "disruption cascade" from ChatGPT launch. Create a network diagram showing which sectors gained, which lost, and causal pathways.

**Deliverable Format:**
- Network visualization of disruption cascade
- Sector-by-sector entropy analysis
- 3-minute video explaining the cascade
- One-page executive summary

**Evaluation Criteria:**
| Criterion | Weight |
|-----------|--------|
| Completeness of cascade mapping | 25% |
| Valid entropy quantification | 25% |
| Insight on second/third-order effects | 25% |
| Professional presentation | 25% |

---

# What's Provided (HEAVY Scaffolding)

**Data Sources:**
- Stock price data: Nvidia, Chegg, Mag 7, S&P 500 (yfinance)
- Sector ETF weights (XLK, XLY, etc.)
- Timeline of key events

**Code Templates:**
- `starter_template.py` — DRIVER-structured cascade analysis
- `validate_thesis.py` — Working validation code
- Sector entropy calculation functions

**What YOU Must Find:**
- Additional affected companies/sectors
- Third-order effects beyond immediate cascade

---

# Discussion Questions

1. **Why did VIX not predict the magnitude of this shift?**
   - Traditional risk captures volatility, not paradigm change

2. **When does a technology become an "asset class"?**
   - At what point did "AI exposure" become required?

3. **What signals would indicate the AI regime is ending?**
   - What entropy signature would you look for?

4. **How is this different from the dot-com bubble?**
   - Revenue growth vs. speculation

---

# Connection to Course Arc

| Week | Case | Builds On |
|------|------|-----------|
| 1 | Tariff Shock | Foundation: Regime shift detection |
| **3** | **ChatGPT Launch** | **Sample space expansion vs. regime shift** |
| 4 | GENIUS Act | Both expand sample space (tech vs. regulation) |
| 5 | Europe Energy | Both show irreversibility |
| 7 | China Property | Both show creative destruction |

**Key distinction:** Tariff = P changed. ChatGPT = X changed.

---

# Sources and Further Reading

1. [Yahoo Finance: Three Years of AI Mania](https://finance.yahoo.com/news/three-years-ai-mania-chatgpt-113000269.html)
2. [Visual Capitalist: Top Sectors Since ChatGPT](https://www.visualcapitalist.com/ranked-the-top-performing-sectors-since-chatgpt-launched/)
3. [CNBC: Chegg Says ChatGPT Killing Business](https://www.cnbc.com/2023/05/02/chegg-drops-more-than-40percent-after-saying-chatgpt-is-killing-its-business.html)
4. [Motley Fool: Magnificent Seven's Market Cap](https://www.fool.com/research/magnificent-seven-sp-500/)
5. [Nvidia FY2025 Results](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025)

---

# Next Steps

**Before Next Session:**
- Review starter code: `src/starter_template.py`
- Fetch stock data for Nvidia, Chegg using yfinance
- Read the Creative Destruction Map

**Weekend Project:**
- Build YOUR OWN sample space expansion analysis
- Different event, different market
- Not a replication — original work

---

*MGMT 69000: Mastering AI for Finance | Purdue MSF | DRIVER Framework*
*"Probability measures outcomes within a fixed game. Entropy measures when the game itself is changing."*
