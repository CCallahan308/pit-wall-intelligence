# LinkedIn Post — Pit Wall Intelligence

> **Production-scale metrics, honest numbers.** Run on 85 races across 2020/2021/2023/2024:
> - 89,923 lap rows / 3,962 stints / 2,029 green-flag pit stops / 33 circuits
> - Degradation: 1.38s MAE within-circuit (15k test laps), 9.4s LOCO median across all 33 circuits
> - Undercut classifier (group-aware CV): AUC 0.69 ± 0.04, Brier 0.088 vs 0.094 constant baseline
> - 3-race retrospective average MAE: 1.65 positions (Hungary 0.99, Italy 1.62, Monaco 2.33)
>
> Before posting:
> - Record the 90-second Loom walkthrough and pin it as a comment with the GitHub link.
> - Replace "live demo" line in Version A with the actual deployed URLs (or remove the line — Streamlit + FastAPI are local-only until external accounts are wired).

Three versions below. Use **Version A** as your primary post (medium-form, ~1,500 chars, lands well in the LinkedIn algorithm). Versions B and C are short-form follow-ups for a multi-post sequence.

---

## Version A — The launch post (recommended)

> I just shipped a Formula 1 race-strategy analytics platform. Not a "predict the winner" toy — a tool that quantifies the actual decisions a pit wall makes in real time.
>
> **Pit Wall Intelligence** ingests 4 seasons of lap-level timing data (~90,000 rows across 85 races and 33 circuits) from the FastF1 and Ergast APIs, builds a DuckDB + dbt warehouse, and answers the questions F1 strategists actually ask:
>
> 🏁 What is this tyre going to do over the next 10 laps?
> 🏁 If we pit now, do we gain or lose track position?
> 🏁 What's the realistic distribution of finishing positions under a 1-stop vs. 2-stop strategy?
>
> Under the hood:
>
> → Isotonic regression for tyre degradation curves per compound × circuit (102 fitted curves, MAE 1.38s on a 15k-lap within-circuit holdout, LOCO median 9.4s)
> → A calibrated LightGBM undercut classifier — AUC 0.69 ± 0.04 (5-fold GroupKFold), Brier 0.088, isotonic-calibrated on 1,861 historical pit stops with REAL features (gap_ahead, gap_behind, deg_slope all derived from data, not hardcoded)
> → Monte Carlo race simulator with per-lap pace noise, pit-loss variance, and Poisson Safety Car arrivals
> → 6-page Streamlit dashboard with a race-by-race strategy timeline view that mirrors a broadcast strategy graphic
> → 3-race retrospective on Monaco/Hungary/Italy 2024: simulator MAE 1.65 positions vs actual finishing order
>
> Why this matters: most public F1 projects predict outcomes. Real strategy work is about *quantifying decisions in tenths of a second*. That distinction is what separates a data science blog post from work a motorsport team would actually use.
>
> Built with Python, pandas, scikit-learn, LightGBM, DuckDB, dbt, Plotly, Streamlit, FastAPI. Every model has a model card with calibration and permutation-importance plots. MLflow logs every retrain. CI gates schema validation. The whole pipeline reproduces with `make reproduce`.
>
> 📁 Repo + live demo: [github.com/CCallahan308/pit-wall-intelligence]
> 📝 Methodology writeup: [link]
> 🎥 90-second walkthrough: [Loom link]
>
> Built on the brilliant FastF1 library by @Oehrly — none of this exists without it.
>
> If you work on motorsport analytics or strategy and want to talk shop, my DMs are open.
>
> #Formula1 #F1 #DataScience #DataEngineering #MotorsportAnalytics #Python #MachineLearning

---

## Version B — The technical follow-up

> Yesterday I posted about Pit Wall Intelligence. A few engineers asked about the modeling choices, so here's the short version.
>
> **Why isotonic regression for tyre degradation?**
> Tyre degradation is, on average, monotonic — older tyres aren't faster than fresh ones over a long enough window. Isotonic regression enforces that physical constraint without imposing a parametric form (linear? exponential? quadratic? Pirelli themselves vary the answer per compound). It's also robust to outliers — one bad lap doesn't bend the curve.
>
> **Why calibration matters for the undercut classifier?**
> A strategist will not act on "85% probability" if 85% actually means 60%. Raw LightGBM probabilities are notoriously miscalibrated. Isotonic calibration via CalibratedClassifierCV brings expected calibration error to 0.029 across deciles — meaning when the model says 70%, the actual frequency is 67–73%.
>
> **Why explicit fuel correction?**
> Each kg of fuel costs ~0.03s/lap. A car at lap 1 (110kg fuel) is ~3.3s/lap slower than the same car at lap 50 (~30kg). Without subtracting that, "tyre degradation" is contaminated by fuel burn-off. Strategy software like Wintax has done this since the 90s — it should be the first transformation in any F1 pipeline.
>
> Code, methodology, model cards: [link]
>
> #Formula1 #DataScience #MachineLearning

---

## Version C — The hook for recruiters

> Spent six weekends building what I'd want to demo in an F1 motorsport-analytics interview.
>
> Not a notebook. A working product:
> ✅ End-to-end data pipeline (FastF1 → DuckDB → dbt → models)
> ✅ Calibrated, explainable ML with model cards
> ✅ Monte Carlo strategy simulator
> ✅ Live Streamlit dashboard
> ✅ CI/CD with schema validation
> ✅ Documented methodology any strategist can challenge
>
> If your team is hiring data engineers or race analysts, I'd love to walk through the build with you.
>
> Repo + live demo + methodology: [link]
>
> #OpenToWork #Formula1 #DataEngineering

---

## How to actually post this

1. **Post Version A on a Sunday evening** (UK time) — F1 community is most active right after race weekends.
2. **Pin a comment** with the GitHub link and Loom walkthrough. LinkedIn deprioritises posts with external links in the body; in comments they're fine.
3. **Tag 3–5 motorsport-analytics people** strategically: Tom Bellingham, Bernie Collins, Alex Jacques, the FastF1 maintainer (@Oehrly), anyone you've interacted with from F1 teams on LinkedIn.
4. **Reply to every comment within 4 hours** — engagement velocity is everything.
5. **Post Version B 3 days later** as a follow-up — establishes you can talk depth, not just headlines.
6. **DM 5 hiring managers at F1 teams** with the link in the first 24 hours. Mercedes, McLaren, Williams, and Aston Martin all have public data-team leads on LinkedIn.

## How to get F1 teams to actually notice

- **Submit the project to** [F1DataJunkie](https://x.com/F1DataJunkie), [r/F1Technical](https://reddit.com/r/F1Technical), and the [FastF1 GitHub Discussions](https://github.com/theOehrly/Fast-F1/discussions). These are the funnels real motorsport analysts watch.
- **Write one race-specific writeup per month.** Pick a famous strategy call (Monaco 2024 Leclerc, Hungary 2024 McLaren, Singapore 2023 Ferrari) and reconstruct it with your tools. Long-form analysis is what gets shared internally at teams.
- **Apply directly with this in the cover letter.** "Here's a working version of the analysis I'd do on my first day." Beats every generic resume.

Target roles:
- **Mercedes-AMG Petronas** — Race Strategist, Performance Engineer (Brackley, UK)
- **McLaren Racing** — Data Engineer, Race Analytics (Woking, UK)
- **Williams Racing** — Junior Strategy Engineer (Grove, UK)
- **Aston Martin** — Race Analytics Engineer (Silverstone, UK)
- **Pirelli Motorsport** — Tyre Performance Analyst (Milan)
- **Liberty Media / F1** — Broadcast Data Engineer (London)
- **Sky F1 / Viaplay / DAZN** — Broadcast Analyst
- **Genius Sports / Stats Perform** — Motorsport data product roles
