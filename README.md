# Explainable, purchase-cost-aware placement of pressure loggers (simulated leaks, benchmark networks)

South Australia-informed BENCHMARK study. It is NOT a model of SA Water's network.

## Run
    python -m pip install -r requirements.txt
    python src/main.py                    # tiny debugging network
    python src/main.py --network Net1     # shows the mandatory device-range check (all junctions exceed 689 kPa)
    python src/main.py --existing J6      # try a network that already has a logger at J6
    python -m unittest discover -s tests -v

## Layout
    network/tiny_network.inp              debugging network (changes documented inside the file)
    data/devices/pressure_logger_catalog.csv   sourced device price, range, accuracy, interval
    data/provenance.csv                   where every sourced number came from
    src/cost_func.py                      purchase cost only: C(S) = unit price x number of new loggers
    src/scenario_runner.py                WNTRSimulator normal and leak runs (fresh model each time)
    src/detection.py                      residual + persistence rule, device range check
    src/placement_search.py               cheapest layout meeting a coverage target (exhaustive, small candidate lists)
    src/explanations.py                   removal, uniqueness and replacement explanations
    src/reporting.py                      plots, assumptions register, run metadata
    results/<network>/                    everything produced by a run

## What each result file is
    scenarios.csv, detection_matrix.csv (scenario_id,candidate_id,detected,first_detection_time,max_residual,...),
    range_check.csv, placement_results.csv, explanations.csv, assumptions_register.csv, run_metadata.csv,
    pressure_residual_example.png, cost_vs_coverage.png, selected_layout.png

## Not claimed
Leak localisation, failure prevention, detection of all small leaks, measured false-alarm performance,
SA Water pressure thresholds or procurement prices, utility approval of the logger.


## LeakG3PD track (Steps 1 and 2) - added v0.3
Needs the pinned environment in `requirements-leakg3pd.txt` (wntr 1.2.0). Details in `CHANGELOG.md`.

    python src/run_steps_1_2.py --net-dir <folder containing Hanoi.inp and Scenario-1..7> --out results/leakg3pd/Hanoi
    python src/run_steps_1_2.py --net-dir <Net3 folder> --out results/leakg3pd/Net3 [--time-budget 120]   # resumable

Scenario folders can be regenerated with `tools/` or, once available, taken from the published archive
(file names are listed in `FILE_NAMES` at the top of `src/leakg3pd_io.py`).
Run all tests with `python -m unittest discover -s tests` (use the pinned environment).
Results: `results/leakg3pd/<network>/` (audit.csv, node_range_screen.csv, matched_reconstruction.csv, level_a_summary.csv,
level_b_grid_development.csv, detector_settings_FROZEN.json, level_b_heldout.csv, sign_check_leaks.csv, example plot).


## Reproducing the final Hanoi results (v1.0)
Environment: `requirements-leakg3pd.txt` (wntr 1.2.0, numpy 1.26.4, pandas 2.1.4). One CPU is enough.
1. Regenerate scenarios (or use the published archive once audited): see `tools/README.md`
   (`python run_gen3.py Hanoi 365 1 1 37 ./out 215` style; seed 1; scenarios 1-37).
2. `python src/run_steps_1_2.py --net-dir <Hanoi folder> --out results/leakg3pd/Hanoi [--time-budget 190]`  (resumable)
3. `python src/run_step5_mdf.py --net-dir <Hanoi folder> --step12-out results/leakg3pd/Hanoi --out results/leakg3pd/Hanoi/final`
4. `python src/run_steps_3_7.py --net-dir <Hanoi folder> --step12-out results/leakg3pd/Hanoi --out results/leakg3pd/Hanoi/final`
5. `python -m unittest discover -s tests`
