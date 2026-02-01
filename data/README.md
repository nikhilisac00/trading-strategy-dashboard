# Data Sources: Case 2 - ChatGPT Launch 2022

## Primary Data Source

### yfinance (Yahoo Finance)
- **No API key required**
- Provides historical stock prices for all analysis

## Key Tickers

### Winners (AI Infrastructure)
| Ticker | Company | Role |
|--------|---------|------|
| NVDA | Nvidia | GPU demand for AI training |
| MSFT | Microsoft | Azure AI, OpenAI partnership |
| GOOGL | Alphabet | AI search, Gemini |
| META | Meta | AI investment |

### Losers (Disrupted)
| Ticker | Company | Role |
|--------|---------|------|
| CHGG | Chegg | Homework help disrupted |
| PSO | Pearson | Education content disrupted |

### Benchmarks
| Ticker | Description |
|--------|-------------|
| SPY | S&P 500 ETF |
| QQQ | Nasdaq 100 ETF |

### Sector ETFs
| Ticker | Sector |
|--------|--------|
| XLK | Technology |
| XLY | Consumer Discretionary |
| XLF | Financials |
| XLE | Energy |
| XLV | Healthcare |
| XLI | Industrials |
| XLB | Materials |
| XLU | Utilities |
| XLRE | Real Estate |
| XLC | Communication Services |

## Key Dates

| Date | Event | Significance |
|------|-------|--------------|
| 2022-11-30 | ChatGPT Launch | Start of AI mania |
| 2023-05-02 | Chegg earnings | CEO admits ChatGPT impact (-49% single day) |
| 2023-05-24 | Nvidia earnings | AI demand "blows away" forecasts |

## Fetching Data

```python
import yfinance as yf

# Fetch Nvidia data since ChatGPT launch
nvda = yf.download("NVDA", start="2022-11-30", end="2024-12-01")

# Fetch multiple tickers
tickers = ["NVDA", "CHGG", "SPY"]
data = yf.download(tickers, start="2022-11-30", end="2024-12-01")
```

## Sector Weight Data

S&P 500 sector weights are approximated from public sources:
- [S&P Dow Jones Indices](https://www.spglobal.com/spdji/)
- [Visual Capitalist](https://www.visualcapitalist.com/)
- Financial news (Bloomberg, CNBC, etc.)

### Approximate Weights Used

**November 2022 (Before AI Boom):**
| Sector | Weight |
|--------|--------|
| Technology | 20% |
| Healthcare | 15% |
| Financials | 12% |
| Consumer Disc. | 10% |
| Other sectors | 43% |

**November 2024 (After AI Boom):**
| Sector | Weight |
|--------|--------|
| Technology | 32% |
| Healthcare | 12% |
| Financials | 10% |
| Consumer Disc. | 8% |
| Other sectors | 38% |

## Magnificent 7 Data

The "Magnificent 7" (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA) weight in S&P 500:
- November 2022: ~20%
- November 2024: ~32%

## Notes

1. **Data Quality**: yfinance provides adjusted close prices (split and dividend adjusted)
2. **Timezone**: Data is in market timezone; code handles timezone conversions
3. **Missing Data**: Some tickers may have gaps; code handles gracefully
4. **Real-time vs. Historical**: This case uses historical data only

## References

1. [Yahoo Finance: Three Years of AI Mania](https://finance.yahoo.com/news/three-years-ai-mania-chatgpt-113000269.html)
2. [Visual Capitalist: Top Sectors Since ChatGPT](https://www.visualcapitalist.com/ranked-the-top-performing-sectors-since-chatgpt-launched/)
3. [CNBC: Chegg Says ChatGPT Killing Business](https://www.cnbc.com/2023/05/02/chegg-drops-more-than-40percent-after-saying-chatgpt-is-killing-its-business.html)
4. [Motley Fool: Magnificent Seven's Market Cap](https://www.fool.com/research/magnificent-seven-sp-500/)
