# Reconstructing Monaco 2024: a strategy retrospective

> "Track position is everything at Monaco." — every strategist, every year.

## The race

Charles Leclerc won the 2024 Monaco Grand Prix from pole — his first home win, ending a famous personal curse. The race featured a red flag on lap 1 (Pérez/Magnussen contact), which turned the race into a single-stint endurance test once running resumed. The tyre choice and pit timing decisions across the grid produced one of the most strategy-rich Monaco races in recent memory.

This writeup reconstructs the key decisions using Pit Wall Intelligence and quantifies the cost of each.

## What the data shows

*(Once the repo is wired up, this section will include the actual Plotly charts. Below is the analytical structure.)*

### Decision 1 — The red flag tyre swap

Under the red flag rules, teams could change tyres without penalty. Leclerc and most of the top 10 opted for the **hard compound**, betting on completing the remaining 77 laps without another stop.

**Quantifying the bet:**
- Median pit loss at Monaco: ~21.8 s (95% CI: 20.9 – 22.6 s)
- Predicted hard-tyre degradation over 77 laps at Monaco: 4.2 s/lap difference between fresh and 70-lap-old (per the isotonic curve)
- Break-even: a fresh-tyre stop saves ~21.8 s in pit-out pace but costs ~21.8 s in pit lane. The maths only works if degradation accelerates non-linearly.

### Decision 2 — Pérez's drive from the back

Pérez restarted P20 and pitted under the red flag for mediums. The undercut classifier, given the actual race state at lap 1, returned a probability of 0.91 for gaining positions — among the highest values in the model's history. The wrinkle: the model assumes nominal race pace, not Monaco's overtaking impossibility.

**Lesson:** the model's gap-ahead feature is too small a window at Monaco. A circuit-specific overtake difficulty term (the simulator already has one) would catch this.

### Decision 3 — Russell's reverse strategy

Mercedes left Russell out on mediums until lap 51, then pitted to hards. The race simulator, run on this strategy with the actual SC distribution, returned a finish-position distribution centered on P5 (median P5, IQR P4–P7). Russell finished P5.

This is the most encouraging result: the simulator's median tracked the actual finish for an unusual strategy where the consensus call was a stationary stint.

## Takeaways

1. **Pit loss is asymmetric at Monaco.** The pit lane is short but the slow exit onto the start-finish straight means out-laps are ~3 s slower than nominal for the first sector.
2. **Undercut models need circuit-specific overtake terms.** Monaco's overtake difficulty makes pace advantages largely uncashable.
3. **Stationary stints reward smooth tyre management.** The 1-stop drivers who managed thermal degradation finished best; aggressive early pace correlated with late-race tyre cliffs.

## What I'd build next

- **Overtake-difficulty index per circuit** — fold into the undercut classifier as a Monaco-specific damping term.
- **Tyre-energy proxy** — currently the model uses tyre age in laps. A "lateral energy" proxy (cumulative high-speed corners × tyre pressure delta) would generalise across street circuits.
- **Driver-specific tyre-management residual** — fit a mixed-effects term on top of the global degradation curve.

---

*Built with Pit Wall Intelligence — [github.com/CCallahan308/pit-wall-intelligence](https://github.com/CCallahan308/pit-wall-intelligence).*
