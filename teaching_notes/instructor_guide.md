# Instructor Guide: Case 2 - ChatGPT Launch 2022

## Session Overview

- **Duration:** 90 minutes
- **Week:** 3 (Monday, January 26)
- **Entropy Concept:** Sample Space Expansion
- **Scaffolding Level:** HEAVY
- **Prerequisites:** Week 1 case completed, basic entropy concepts understood

---

## Learning Objectives

By the end of this session, students will be able to:

1. **Identify "sample space expansion" events** — When new asset classes emerge
2. **Map disruption cascades** — Trace second and third-order effects
3. **Distinguish "big news" from "new game" signals** — Paradigm shift vs. noise
4. **Apply entropy to sector analysis** — Measure creative destruction

---

## Session Flow

### Opening (0-15 min)

**Hook (3 min):**
> "On October 12, 2022, Nvidia was worth $345 billion and down 70% from its peak. Today it's worth $4.4 trillion. What happened in between?"

**Context (7 min):**
- Show the timeline: Bear market bottom → ChatGPT launch → AI mania
- Display the contrast: Nvidia +700% vs. Chegg -99%
- Ask: "Is this just a stock story, or did something fundamental change?"

**Today's Question (5 min):**
> "When does a new game begin? How do we detect when the investment universe itself expands?"

Connect to Week 1:
- Week 1: P changed (transition probabilities)
- Week 3: X changed (the sample space itself)

---

### Guided DRIVER Walkthrough (15-60 min)

**CRITICAL: Teach the Loops, Not Just the Stages**

Before diving in, remind students:
> "DRIVER has six stages, but it's not a linear checklist. Real work loops back. Today we'll experience the R-I loop when our plan reveals gaps."

#### DISCOVER & DEFINE (15-25 min) — 10 minutes

**DEFINE FIRST (Provisional):**

Write on board:
```
OBJECTIVE (One Sentence):
Map how ChatGPT caused sample space expansion.

WHAT DOES "DONE" LOOK LIKE?
□ Cascade mapped
□ Returns calculated
□ Entropy quantified
□ Can explain P vs X difference
```

**Ask students:** "How will we know if we're WRONG?"
- Expected: If divergence is small, if entropy increases, if no new category emerged

**DISCOVER (Resources & Constraints):**

Walk through the Creative Destruction Map:
- ChatGPT launches → 100M users in 2 months
- AI infrastructure demand explodes → Nvidia
- Knowledge work disrupted → Chegg

**Ask students:**
- "Before ChatGPT, was 'AI exposure' a standard portfolio consideration?"
- "What made ChatGPT different from previous AI announcements?"

**Resources available:**
- Stock price data from yfinance (NVDA, CHGG, SPY)
- Sector ETF data (XLK, XLY)
- Timeline of events with dates

**DEFINE (Refined):**
> "This wasn't a price change in existing assets. A new asset class entered the investment universe. Week 1 was P changing; Week 3 is X changing."

---

#### REPRESENT (25-35 min) — 10 minutes

**How do we approach this?**

Draw the cascade diagram on whiteboard:

```
Level 0: Trigger
    └── ChatGPT Launch (Nov 30, 2022)

Level 1: Direct Effects
    ├── AI Infrastructure (+)
    │   ├── Nvidia (GPUs)
    │   └── Microsoft (Azure)
    └── Knowledge Work (-)
        └── Chegg (homework help)

Level 2: Second-Order
    ├── Data Centers (+)
    ├── Energy Demand (+)
    └── Education Disruption (-)

Level 3: Third-Order
    ├── Real Estate (data center REITs)
    ├── Utilities (power demand)
    └── Curriculum Changes
```

**Methodology:**
1. Identify the trigger event (ChatGPT launch)
2. Map direct winners and losers
3. Trace second-order effects
4. Calculate sector entropy change

---

#### IMPLEMENT (35-50 min) — 15 minutes

**Live coding demonstration points:**

**Part 1: Fetch stock data (5 min)**

```python
import yfinance as yf

# Winners and losers
tickers = {
    "NVDA": "Nvidia (Winner)",
    "CHGG": "Chegg (Loser)",
    "SPY": "S&P 500 (Benchmark)",
    "MSFT": "Microsoft (Winner)",
}

# ChatGPT launch date
chatgpt_launch = "2022-11-30"
```

Show how to fetch and plot the divergence.

**Part 2: Calculate returns since ChatGPT (5 min)**

```python
def returns_since_chatgpt(ticker, launch_date="2022-11-30"):
    data = yf.download(ticker, start=launch_date)
    return (data["Close"][-1] / data["Close"][0] - 1) * 100
```

**Key teaching moment:**
> "Nvidia +700%, Chegg -99%. This isn't normal sector rotation. This is creative destruction."

**Part 3: Sector entropy calculation (5 min)**

```python
def sector_entropy(weights):
    """Lower entropy = more concentrated market."""
    weights = np.array([w for w in weights if w > 0])
    weights = weights / weights.sum()
    return -np.sum(weights * np.log2(weights))
```

Show how Mag 7 weight increase → lower entropy → more concentrated.

---

#### VALIDATE (50-60 min) — 10 minutes

**How do we know if we're right?**

Validation questions:

1. **Is this actually sample space expansion?**
   - Did "AI infrastructure" exist as investment category before?
   - Is this a new asset class or existing asset repricing?

2. **Is the cascade logic sound?**
   - ChatGPT → AI demand → Nvidia: Clear causal link
   - ChatGPT → Chegg destruction: Direct disruption

3. **Does entropy capture the shift?**
   - Market concentration increased (Mag 7 dominance)
   - But sample space also expanded (new category)

4. **Would we have detected this in real-time?**
   - What signals would have indicated "new game"?

---

### Discussion (60-80 min)

**Key Discussion Questions with Expected Responses:**

**Q1: "Why did VIX not predict this shift?"**
- Expected: VIX measures expected volatility, not paradigm change
- Push: "What would a 'paradigm shift detector' measure?"
- Key insight: Entropy of information flow, not price volatility

**Q2: "When does a technology become an 'asset class'?"**
- Expected: When it requires dedicated portfolio allocation
- Push: "At what point did you HAVE to have AI exposure?"
- Timeline: By May 2023 (Nvidia earnings), it was unavoidable

**Q3: "What signals would indicate the AI regime is ending?"**
- Expected: Revenue growth slowing, new competing paradigm
- Push: "What's the entropy signature of a paradigm ending?"
- Key insight: Entropy would spike as new uncertainty enters

**Q4: "How is this different from the dot-com bubble?"**
- Expected: Real revenue growth this time (Nvidia $130B)
- Push: "But weren't there 'real' companies in 2000 too?"
- Nuance: Revenue velocity and profitability are different

---

### Wrap-up (80-90 min)

**Key Takeaways (5 min):**

1. **Sample space expansion ≠ regime shift:** Week 1 was P changing; Week 3 is X changing (the investment universe itself)

2. **Creative destruction is measurable:** Winners (+700%) and losers (-99%) diverge in quantifiable ways

3. **New asset classes can emerge rapidly:** ChatGPT made "AI infrastructure" mandatory in months, not years

**Preview Wednesday (2 min):**
- Wednesday: Live case discovery
- Apply sample space expansion lens to current events
- Students bring case suggestions

**Deliverable Reminder (3 min):**
- Map the disruption cascade
- Calculate sector entropy before/after
- Network visualization required
- Submission 1 due end of Week 3

---

## Common Student Questions

**Q: "Isn't this just momentum trading?"**
**A:** Momentum captures price trends. Sample space expansion captures structural change. Momentum would have missed Chegg's destruction until too late. The cascade map would have flagged it immediately.

**Q: "How do we detect sample space expansion in real-time?"**
**A:** Look for: (1) New terminology entering analyst reports, (2) Portfolio allocation questions that didn't exist before, (3) Companies being valued on metrics that didn't exist. ChatGPT triggered all three.

**Q: "Why entropy and not just correlation?"**
**A:** Correlation shows relationships within existing assets. Entropy shows when the information structure itself changes. New asset classes entering means the entropy of the market structure changes.

**Q: "Could we have predicted Chegg's destruction?"**
**A:** Yes, with cascade analysis. Once ChatGPT demonstrated competence in homework help, Chegg's business model was threatened. The question was timing, not direction.

---

## Technical Notes

### Required Packages
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
yfinance>=0.2
networkx>=3.0 (for cascade visualization)
pytest>=7.0
```

### Data Access

**yfinance (Primary):**
- No API key required
- Tickers: NVDA, CHGG, MSFT, SPY, GOOGL, META, AMZN, AAPL, TSLA
- Sector ETFs: XLK, XLY, XLE, XLF

**Sector Weights:**
- S&P 500 sector weights from public sources
- Mag 7 weight tracking from financial news

### Potential Issues

**1. yfinance Rate Limits**
- Solution: Cache responses locally
- Alternative: Use provided CSV files

**2. Date Alignment**
- ChatGPT launch was Nov 30, 2022 (Wednesday)
- First full trading day after: Dec 1, 2022

**3. Chegg Stock Splits/Adjustments**
- Use adjusted close prices
- yfinance handles this automatically

---

## DRIVER Loops: Teaching Iteration, Not Just Stages

**The Most Important Lesson: Loops Are Not Failure**

Students often think looping back means they did something wrong. Reframe this:
> "The loop is the methodology working correctly. Building reveals what planning couldn't."

### The R-I Loop (Represent ↔ Implement)

**When it happens in this case:**
- Student starts fetching data, discovers yfinance returns MultiIndex columns
- Student calculates entropy, realizes weights don't sum to 1
- Student plots visualization, realizes log scale needed for 700% vs -99%

**How to facilitate:**
1. When student hits issue: "Great — you've discovered something your plan couldn't predict."
2. Ask: "What needs to change in your approach?"
3. Validate: "Now update your mental model before continuing."

**Teaching moment:**
> "You cannot fully understand a problem until you've tried to solve it. The R-I loop is how you turn that reality into a feature rather than a bug."

### The V-D Loop (Validate ↔ Discover)

**When it happens in this case:**
- Validation shows divergence is only 300% (less than expected) — date range issue?
- Entropy actually increased — wrong sector weights?
- Cascade logic doesn't hold — maybe this ISN'T sample space expansion?

**How to facilitate:**
1. When validation fails: "Don't force the conclusion. What does this tell us?"
2. Ask: "Did we define the right problem?"
3. If needed: Return to DISCOVER & DEFINE and refine

**Teaching moment:**
> "Sometimes validation reveals you solved the wrong problem. The V-D loop is uncomfortable but essential."

### The E-I Loop (Evolve ↔ Implement)

**When it happens in this case:**
- While cleaning up code, student realizes core entropy calculation has bug
- While generalizing, student finds edge case that breaks fundamentals

**How to facilitate:**
1. "EVOLVE isn't just polish — sometimes it's rebuild correctly."
2. Don't punish the discovery; reward the catch.

---

## Scaffolding Notes (HEAVY)

**Week 3 = Still High Support**

**What Instructor Provides:**
- Complete timeline with dates
- Stock data fetching code
- Cascade map template
- Entropy calculation functions
- Validation code that runs

**What Students Must Figure Out:**
- Additional affected companies
- Third-order effects
- Network visualization design
- Narrative connecting the cascade

**Key Instructor Behaviors:**
- Guide students through cascade logic
- Help with yfinance issues
- Focus on "why" not just "what"
- Connect back to Week 1 concepts
- **Explicitly call out loops when they happen**
- **Normalize iteration as part of the process**

---

## Timing Adjustments

**If running short on time:**
- Skip third-order effects discussion
- Focus on Nvidia/Chegg contrast only
- Assign detailed cascade mapping as homework

**If running ahead:**
- Extended discussion on "when is a bubble vs. paradigm?"
- Live demo of networkx cascade visualization
- Compare to dot-com bubble (2000)

**If technical issues:**
- Have pre-computed returns ready
- Show screenshots of cascade visualization
- Focus on conceptual understanding

---

## Post-Session Checklist

- [ ] Students understand sample space expansion vs. regime shift
- [ ] Students can articulate the cascade logic
- [ ] Students understand entropy as concentration measure
- [ ] Students know deliverable expectations
- [ ] Preview of Wednesday session given
- [ ] Submission 1 deadline reminder (end of Week 3)

---

## AI Prompting Guidance by DRIVER Stage

**Teach students to use AI strategically, not as autopilot.**

### DISCOVER & DEFINE Stage

**Good prompts:**
- "What aspects of ChatGPT's market impact am I likely overlooking?"
- "What data would an expert typically want for analyzing technology disruptions?"
- "What are common ways this type of analysis fails?"

**Dangerous prompts:**
- "Tell me everything about ChatGPT's impact on markets" (too broad)
- "What should my objective be?" (outsourcing judgment)

### REPRESENT Stage

**Good prompts:**
- "Here's my approach [describe]. What am I likely missing?"
- "What are alternatives to mapping disruption as a cascade?"
- "Help me visualize the flow from trigger to third-order effects"

**Dangerous prompts:**
- "Create a complete analysis plan for me" (no ownership)
- "What's the best way to do this?" (too vague)

### IMPLEMENT Stage

**Good prompts:**
- "Write a function that calculates Shannon entropy given sector weights"
- "This code produces [error]. I expected [behavior]. Here's the code..."
- "Explain why this return calculation uses (end/start - 1) * 100"

**Dangerous prompts:**
- "Build me a complete ChatGPT market analysis"
- "Write code to prove my thesis"

### VALIDATE Stage

**Good prompts:**
- "What are common errors in entropy calculations?"
- "Help me design test cases for this function"
- "What edge cases should I check for return calculations?"

**Critical rule:** NEVER use AI alone to validate AI output. Always verify with external sources.

### EVOLVE Stage

**Good prompts:**
- "How can I make this function more readable?"
- "What patterns from this analysis might be reusable?"
- "What would make this code easier to maintain?"

### REFLECT Stage

**Good prompts:**
- "Help me articulate what I learned from this analysis"
- "What broader principles does this case illustrate?"

**Key teaching point:**
> "AI is your cognitive co-pilot, not autopilot. You set the parameters, verify outputs, and take responsibility."

---

*MGMT 69000: Mastering AI for Finance | Purdue MSF | DRIVER Framework*
