# Changelog

## v1.0 - 21 Nov 2026 - project close: Hanoi final analysis (Steps 3-7) + integration of collaborator work
**Added**
- `src/run_steps_3_7.py`  event x candidate detection matrices (Level A at 1x/2x/5x accuracy; Level B at the frozen setting plus 3 sensitivity settings), fixed-count placement on development events (exact MIP, greedy, sensitivity-ranked, 1,000 random layouts), scoring on held-out events, layout-level false alarms, per-sensor explanations, equal-count alternatives, setting sensitivity, coverage-vs-detectable-flow comparison, figures.
- `src/run_step5_mdf.py`  minimum detectable flow on the base network (idealised and realistic thresholds).
- `src/coverage_optimiser.py`, `src/min_detectable_flow.py` + their tests, from the collaborator track (HiGHS MIP, no licence needed). Retested on the pinned WNTR 1.2.0.
- `docs/`  collaborator's verified sources: `paper_facts.csv`, `claims_check.csv` (Du et al. rows corrected from the PDF), SA Water context tables, device-candidate notes.
- 44 leak events in 36 leak scenarios (Hanoi, seed 1): 24 development / 20 held-out. Split by scenario ID, declared before results on scenarios 8-37 were seen.
**Verified**
- Matched baseline reconstructs the dataset to 0.00001 m in 36 of 36 Hanoi leak scenarios.
- MIP objective equals exhaustive search in 28 of 28 checks (k <= 4); all 28 solves reported `optimal`.
- 81 automated tests pass on the pinned environment.
**Headline Hanoi results** are in `results/leakg3pd/Hanoi/final/` and the final report.
**Known limits**
- Regenerated data, not the published archive. Net3 is reported as a limitation only (no placement claims).
- 20 held-out events: one event is 5 percentage points.

## v0.3 - 20 Nov 2026 - LeakG3PD track: Steps 1 and 2 on Hanoi
**Added**
- `src/leakg3pd_io.py`  loader + audit for LeakG3PD-format scenario folders (stable original candidate IDs; leak nodes, reservoirs and tanks never candidates).
- `src/baseline.py`  Level A matched baseline (re-run saved .inp without the leak) and Level B estimated baselines (frozen, rolling), both causal.
- `src/detector.py`  residuals (drop = baseline - observed), one-sided alarms, persistence, thresholds from development normal data, event scoring that never uses labels, false-alarm rate, rise check.
- `src/run_steps_1_2.py`  one command: audit, reconstruct, Level A, Level B grid, declared selection rule, FROZEN settings, held-out evaluation, plots.
- `tools/`  seedable Linux copy of the official generator (EUPL-1.2 notice inside) and run script.
- `tests/`  29 new tests (loader, baselines, detector). All 51 tests pass.
- `requirements-leakg3pd.txt`  pinned environment (wntr 1.2.0, numpy 1.26.4, pandas 2.1.4).
**Verified (Hanoi, 7 regenerated full-year scenarios, seed 1)**
- Hanoi: matched baseline reconstructs the dataset pressures to 0.00001 m in 6 of 6 leak scenarios.
- All labels match the leak files; no missing values; no negative pressures; maximum 686.5 kPa (inside 689 kPa).
- Detector settings chosen on development scenarios only, then frozen, then evaluated on held-out scenarios.
**Changed**
- Old tiny-network code (`main.py`, `scenario_runner.py`, `detection.py`) is unchanged and still runs; it is NOT used by the LeakG3PD track.
**Net3 results (same 7 regenerated scenarios) - important caveats**
- Device range screen: 84 of 91 junctions are compatible with a 0-689 kPa gauge logger. Node 61 reaches 984 kPa; six nodes go negative (node 10 is negative 79% of the time). Even the leak-free scenario has a negative node.
- Matched baseline: exact (0.00001 m) in 5 of 6 leak scenarios; scenario 5 diverged on 1.2% of pre-onset steps (first on day 199, up to 45 m), most likely tank/pump control timing differences from rounded pattern values in the saved .inp.
- Net3 pressures are dominated by tank-pump cycling. A leak changes the cycle timing, so residuals swing about +-12 m in BOTH directions; pressure rises exceed accuracy in 28-50% of leak samples (none in Hanoi). The one-sided drop rule is NOT justified for Net3.
- Estimated baseline (time-of-day) cannot model pump cycles. Only one development leak event had enough history; the only setting meeting the false-alarm target detected almost nothing on development data.
- Net3 is therefore reported as a limitation of the simple detector, not as a clean placement example.
**Known limits**
- Data are regenerated with the official generator, not the published Google Drive archive (not reachable from the build environment).
- Only 6 leak scenarios per network. Net3: only 3 of 6 leak events start after the baseline warm-up.
- Step 3 onward (coverage matrix, optimiser, minimum detectable flow) not started in this track.

## v0.2 - purchase-only cost + tiny-network leak simulation
- Hardware-only cost from a sourced Australian logger price, WNTRSimulator leak runs, exhaustive placement search, explanations, 22 tests.

## v0.1 - original prototype
- Lifecycle cost model and one-hop adjacency coverage on the tiny network.
