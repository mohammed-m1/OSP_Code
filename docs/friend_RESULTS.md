# RESULTS: side-task pack (final status, 2026-11-20)

Rule used throughout: nothing invented; every value is sourced (document + page/URL + date), derived (calculation shown) or assumed (reason given). Where not found: "NOT FOUND".

## 1. Completed
| Step | Deliverable | Status |
|---|---|---|
| A1/A2 | `paper_facts.csv`, `claims_check.csv` | Done for Quintiliani 2020 and Zhao 2020 (full text, journal page numbers). Du 2024: NOT FOUND (no source; owner says it does not exist). A third paper, an abstract pasted in chat (Bentley-type two-phase pressure-logger method), is recorded separately and is not Zhao. |
| A3 | (none) | Skipped: Sarrate/Blesa paper not retrievable. Only a secondary-source paraphrase of detectability vs isolability exists. |
| B1 | `data/pressure_logger_catalog_step_b_candidate.csv`, `data/provenance_step_b_rows.csv` | Done. EFENTO: AUD 726.36 ex GST, 0-689 kPa gauge, +/-0.25% FSO. Over-pressure rating NOT FOUND. |
| B2 | `data/pressure_logger_catalog_step_b2_candidates.csv`, `data/provenance_step_b2_rows.csv` | Done (partial). MadgeTech PR1000 above 1,000 kPa has published AU prices but +/-2% accuracy. Intervals not retrieved. |
| C1 | `sa_context_2018_19.csv` | Done: 35 page-referenced rows. |
| C2 | `sa_water_monthly_incidents.csv`, `sa_water_annual_context.csv` | Done. 4,798 / 27,742 km claim VERIFIED. |
| D | `src/coverage_optimiser.py`, `tests/test_coverage_optimiser.py` | Done: 18 tests pass. |
| E | `src/min_detectable_flow.py`, `tests/test_min_detectable_flow.py` | Done: 10 tests pass. |
| extra | `scripts/zhao_net3_sanity_check.py` | Approximate Net3 check against Zhao's published numbers (not a reproduction). |
| F | (none) | Waits for the main track's loader and detector and the real archive. |

## 2. What the literature does and does not support
Sourced (Zhao 2020, Quintiliani 2020): fixed sensor count, maximise events covered; binary detection matrix; an event counts if any sensor detects it; coverage = detected / ALL events; residual = leak minus baseline pressure; threshold from sensor accuracy (Zhao also adds a normal-difference term set to zero); added-demand leaks at nodes (Zhao p611) or emitters (Quintiliani p4); events at every node; fixed single operating condition (Quintiliani 3 PM p4).
NOT supported by any paper we have: the MIP (both papers use genetic algorithms), the persistence rule, the tie-break order, the matched baseline, leak duration. These are our adaptations. Describe the work as an adaptation, not a reproduction of Zhao's genetic algorithm or of any Du metric.

## 3. Design decisions and assumptions (all ours unless stated)
Step D: coverage = weighted detected / all weights; k clamped to number of candidates; k=0 returns an empty layout with status "trivial"; layouts use exactly min(k, n) sensors; mean delay = earliest alarm per detected scenario averaged over detected scenarios; MIP tie-break applied by enumeration only when layouts <= 200,000 (flagged `tie_break_applied`), otherwise the solver's layout is returned; status "optimal" only if HiGHS says so; time-limit tests use a mocked solver.
Step E: leak = added constant demand at a node (not a leak area); WNTRSimulator, demand-driven; one time step (`duration=0`, `pattern_start=time_s`; tanks at initial level); monotonicity checked before bisection; search-limit failures are `above_range`/`below_range` (unresolved); non-monotone pairs flagged and excluded from the mean; name is "mean minimum detectable flow", not ADT. Tested on WNTR 1.5.0 (project pins 1.2.0: untested).
Step B: accuracy in kPa assumes FSO = full-scale span (supplier does not define it); interval_max 864,000 s derived (10 days).

## 4. Findings that matter for the study
1. EFENTO accuracy 1.7225 kPa = 0.25 psi = 0.176 m head (derived). Zhao's base case is 0.05 psi; his worst case 0.09 psi gave only 64.7% (single leak) vs 93.1% at 0.03 psi (p615). Expect far lower coverage than Zhao's 92%.
2. Higher-range loggers with published AU prices have +/-2% FSO: at 300 PSIG that is 41 kPa = 4.2 m head. Not a like-for-like substitute for Net3.
3. Zhao says Net3 mean demand is about 10,950 GPM; Net3 from WNTR/epyt has base demands summing to about 3,052 GPM. Unresolved. Express leak size as a percentage of demand.
4. Zhao's sanity check (`scripts/zhao_net3_sanity_check.py`): N=5 gives 90.1% (midnight) or 72.6% (noon) vs Zhao 86.86%; N=25 gives 94.3% / 83.0% vs 92.23%. Coverage depends strongly on time of day, and the paper does not say which time it used. Fix and state the operating condition.
5. Equal-coverage alternative layouts exist (Zhao Table 1), so a declared tie-break is needed.
6. SA Water: 4,115 main breaks/leaks in 2018-19 (p21); 4,798 in 2025-26 on 27,742 km. Context only; never used to set leak sizes, failure probabilities or thresholds.

## 5. Discrepancies left unresolved
SA Water connections 709,407 (p8) vs 699,738 (p12); SA Water monthly sum Jul 2025-Jun 2026 = 4,891 (derived) vs 4,798 in the annual report; BoM 2023-24 rate 14.3 in text vs 14.2/16.8 in table; MadgeTech 300 PSIG price $1,872.40 vs $1,656; EFENTO max interval 10 vs 30 days; Quintiliani eligible sites 5,814 vs 5,818; Zhao "21 cases" vs N=1..25.

## 6. Open items
Du 2024 (verify the DOI in the progress note or drop every Du-based statement); the abstract pasted in chat (identify its authors or leave it uncited); EFENTO over-pressure rating; MadgeTech intervals; SA Water SCADA/logger practice/pressure ranges (only SA Water can answer); A3; WNTR 1.2.0 retest; Step F.

## 7. Files outside my tasks changed
None. Existing project files were only read.
