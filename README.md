# FRAUD // CONSOLE

A Streamlit dashboard for hunting the features that predict credit-card fraud,
built on `credit_card_fraud_10k.csv` (10,000 transactions, 151 confirmed frauds,
1.51% base rate). Terminal/hacker visual style, dark green on black.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What each tab does

| Tab | Question it answers |
|---|---|
| **OVERVIEW** | When and where does fraud land? Hourly curve, merchant category, amount distribution, category × time-of-day heatmap. |
| **SIGNAL SCAN** | Which feature actually predicts fraud? Information Value ranking, point-biserial correlation, and a per-level breakdown (fraud rate, lift, share of frauds, WoE) for any feature. |
| **SEGMENT HUNTER** | Which *rule* catches fraud? Auto-mines every single condition and every profitable pair, ranked by lift; plus a manual rule builder that reports precision, recall and dollars caught. |
| **MODEL LAB** | How much signal is there in total? Out-of-fold random forest and logistic regression, ROC / PR curves, feature importance, odds ratios, and a review-budget slider. |
| **RAW DATA** | The filtered rows, with CSV export. |

Every panel reacts to the sidebar filters (class, category, amount, hour, device
trust, velocity, age, foreign flag, location mismatch).

## Findings on this dataset

Ranked by Information Value, on the unfiltered book:

| feature | IV | worst level | fraud rate there | lift |
|---|---|---|---|---|
| device trust score | 1.76 | 20-40 | 5.9% | 3.9× |
| hour of day | 1.55 | 00:00-05:00 | 5.1% | 3.3× |
| foreign transaction | 1.12 | yes | 8.4% | 5.6× |
| location mismatch | 0.93 | yes | 8.4% | 5.6× |
| velocity last 24h | 0.51 | 6+ | 8.2% | 5.5× |
| amount | 0.15 | $1k+ | 14.3% | 9.5× |
| merchant category | 0.04 | Grocery | 2.0% | 1.3× |
| cardholder age | 0.01 | 36-45 | 1.8% | 1.2× |

*(`stacked risk flags` scores higher still, but it is a derived counter of
foreign + mismatch + night — a summary of the three, not a ninth signal.)*

The picture is consistent across methods: **device trust, night hours, the two
geography flags and velocity carry the signal; merchant category and cardholder
age carry almost none.** Amount matters only in the thin $1k+ tail.

The strongest two-condition segments:

| rule | txns | fraud rate | lift | % of all fraud |
|---|---|---|---|---|
| location mismatch AND foreign transaction | 85 | 34.1% | 22.6× | 19% |
| location mismatch AND device trust 20-40 | 167 | 32.3% | 21.4× | 36% |
| foreign transaction AND device trust 20-40 | 198 | 30.8% | 20.4× | 40% |
| device trust 20-40 AND night 00-05 | 553 | 17.9% | 11.9× | 66% |

A forest scored out of fold reaches ROC AUC 0.998 / PR AUC 0.931. Sending the
top **2%** of transactions to review catches **90% of fraud at 68% precision** —
about 199 alerts per 10k transactions.

## Layout

```
app.py               # streamlit UI: filters, tabs, charts
fraud_analytics.py   # binning, lift/WoE/IV, segment mining, models  (no streamlit)
fraud_theme.py       # palette, plotly template, terminal CSS
.streamlit/config.toml
```

## About the colours

The categorical palette is validated for the dark surface `#0a0e0a`: every slot
sits inside the OKLCH lightness band, clears the chroma floor, and every adjacent
pair clears both the colour-vision-deficiency gate (worst pair, green ↔ crimson,
ΔE 11.7 under deuteranopia) and the normal-vision floor (ΔE 30.9), at ≥3:1
contrast against the background. Neon `#00ff41` is chrome only — headings,
borders, KPI numbers — never a data mark.
