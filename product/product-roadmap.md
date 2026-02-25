# Trading Strategy Dashboard — Improvement Roadmap

## Vision
Make the AI advisor actually enforce what users say. "I like TSLA, moderate risk" should produce a portfolio that visibly reflects both constraints — not a generic moderate allocation with a token TSLA position.

## Sections (build order)

### 1. PortfolioSpec ← START HERE
A canonical data object that captures user goals, preferences, and constraints from the AI conversation. Every tab reads from it. Nothing is lost in translation.

**Why first:** Everything else (allocator, UI feedback, regime overlays) depends on preferences being a first-class object — not extracted text.

### 2. Two-Stage Allocator
Replace flat risk-profile weights with a 2-stage engine:
- Stage A: risk profile → strategic baseline
- Stage B: apply PortfolioSpec tilts + regime overlay + constraint solver

### 3. Preference Adherence UI
Show users a checklist: "You asked for X → we did Y."
Closes the feedback loop so users trust the advisor.

### 4. Regime Explainability
Bound how much regime can move weights. Show what changed and why.
Minimum hold period to prevent whipsawing.

### 5. Shared State Flow
All 6 tabs read/write the same PortfolioSpec and Portfolio objects.
Options Builder actions reflect in Portfolio. Regime changes show in Performance.
