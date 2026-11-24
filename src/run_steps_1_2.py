"""
STEPS 1 AND 2 for one LeakG3PD-format network folder.

    python src/run_steps_1_2.py --net-dir /path/to/Hanoi --out results/leakg3pd/Hanoi

STEP 1  load + audit scenarios, reconstruct matched baselines, build estimated baselines, plot pressure and residuals.
STEP 2  detector: sign/direction, thresholds from development NORMAL data, persistence grid, declared selection rule,
        FROZEN settings, then evaluation on held-out scenarios.
Development / held-out split is by WHOLE scenario and is fixed here, before any result is seen.
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import frozen_baseline, matched_baseline, reconstruction_error, rolling_baseline, warmup_steps
from cost_func import KPA_PER_M_HEAD, load_device_catalogue
from detector import (alarm_state, calibrate_thresholds, drop_residual, false_alarm_rate, rise_check, score_events)
from leakg3pd_io import audit, load_all

ROOT = Path(__file__).resolve().parents[1]
# Split by WHOLE scenario, declared before any result on scenarios 8-37 was seen.
# Scenarios 1-7 keep their original assignment; 8-23 are development and 24-37 are held-out.
DEV_IDS = tuple(range(1, 5)) + tuple(range(8, 24))
HELD_IDS = (5, 6, 7) + tuple(range(24, 38))
HORIZON_DAYS = 14                                     # an event must be detected within this many days of onset
TRAIN_DAYS, WINDOW_DAYS, GUARD_DAYS = 56, 28, 7
GRID_Q, GRID_K = (0.99, 0.999), (1, 2, 3, 4, 6)
FA_MEDIAN_MAX, FA_P90_MAX = 0.02, 0.10                # declared false-alarm targets: episodes per node per day
ACC_MULTIPLES = (1, 2, 5)                             # Level A thresholds as multiples of sensor accuracy


def estimated(p, method):
    return frozen_baseline(p, TRAIN_DAYS) if method == "frozen" else rolling_baseline(p, WINDOW_DAYS, GUARD_DAYS)


def normal_mask(scn, warm):
    m = np.zeros(len(scn.pressures), dtype=bool)
    m[warm: scn.events[0].start if scn.events else len(m)] = True      # leak-free history only
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=1e9, help="seconds; stop starting new matched-baseline runs after this (resume by re-running)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev_spec = load_device_catalogue(ROOT / "data" / "devices" / "pressure_logger_catalog.csv")["EFENTO_BT_689KPA"]
    acc = dev_spec.accuracy_m_head
    scns = {s.sid: s for s in load_all(args.net_dir)}
    net = next(iter(scns.values())).network
    print(f"== {net}: scenarios {sorted(scns)} | logger accuracy {acc:.3f} m | range {dev_spec.range_max_kpa:.0f} kPa")

    # ------------------------------------------------ STEP 1: audit
    aud = pd.DataFrame([audit(s, dev_spec.range_max_kpa) for s in scns.values()])
    aud.to_csv(out / "audit.csv", index=False)
    print(aud.drop(columns=["network", "over_range_nodes"]).to_string(index=False))

    # ------------------------------------------------ STEP 1: device range screen (hard eligibility condition)
    allP = pd.concat([s.pressures for s in scns.values()])
    screen = pd.DataFrame(dict(min_m=allP.min(), max_m=allP.max(), max_kpa=allP.max() * KPA_PER_M_HEAD,
                               share_negative=(allP < 0).mean()))
    screen["eligible"] = (screen.min_m * KPA_PER_M_HEAD >= dev_spec.range_min_kpa) & (screen.max_kpa <= dev_spec.range_max_kpa)
    screen.round(4).to_csv(out / "node_range_screen.csv")
    elig = list(screen.index[screen.eligible])
    print(f"\nDEVICE RANGE SCREEN ({dev_spec.range_min_kpa:.0f}-{dev_spec.range_max_kpa:.0f} kPa gauge): {len(elig)} of {len(screen)} candidates eligible")
    if len(elig) < len(screen):
        print(screen[~screen.eligible].round(3).to_string())
    for s in scns.values():
        s.all_pressures, s.pressures = s.pressures, s.pressures[elig]

    # ------------------------------------------------ STEP 1: matched baselines (Level A) over each leak window (cached, resumable)
    matched, rec = {}, []
    cache = out / "cache"; cache.mkdir(exist_ok=True)
    for sid, s in scns.items():
        if s.is_leak_free:
            continue
        upto = min(len(s.pressures), s.events[-1].start + HORIZON_DAYS * 48)
        cf = cache / f"matched_S{sid}_{upto}.pkl"
        if cf.exists():
            b = pickle.loads(cf.read_bytes())
        else:
            if time.time() - t0 > args.time_budget:
                print(f"TIME BUDGET reached before scenario {sid}: re-run the same command to resume (cached baselines are kept).")
                sys.exit(3)
            b = matched_baseline(s, upto)
            cf.write_bytes(pickle.dumps(b))
            print(f"  matched baseline scenario {sid} computed ({upto} steps), t={time.time()-t0:.0f}s", flush=True)
        rec.append(dict(scenario=sid, **reconstruction_error(s, b)))
        matched[sid] = b[elig]
    pd.DataFrame(rec).to_csv(out / "matched_reconstruction.csv", index=False)
    print("\nMatched-baseline reconstruction (re-run .inp without leak vs dataset, non-leak steps):")
    print(pd.DataFrame(rec).to_string(index=False))

    # ------------------------------------------------ LEVEL A detectability (idealised)
    rows = []
    for sid, b in matched.items():
        s = scns[sid]
        dropA = drop_residual(b, s.pressures.iloc[:len(b)]).reindex(s.pressures.index)
        for m in ACC_MULTIPLES:
            sc = score_events(alarm_state(dropA, m * acc, 1), dropA, s.events, HORIZON_DAYS * 48)
            sc["scenario"], sc["multiple_of_accuracy"], sc["threshold_m"] = sid, m, round(m * acc, 3)
            rows.append(sc)
    A = pd.concat(rows)
    A.to_csv(out / "level_a_event_node.csv", index=False)
    summA = (A.groupby(["scenario", "event", "threshold_m"])
               .agg(nodes_detecting=("detected", "sum"), nodes=("node", "count"),
                    best_max_drop_m=("max_drop_m", "max"), median_delay_h=("delay_h", "median")).reset_index())
    summA.to_csv(out / "level_a_summary.csv", index=False)
    print("\nLEVEL A (matched baseline): nodes (of all candidates) that detect each event within", HORIZON_DAYS, "days")
    print(summA.round(3).to_string(index=False))

    # ------------------------------------------------ LEVEL B: estimated baselines, development data only
    methods = ("frozen", "rolling")
    dev = [scns[i] for i in DEV_IDS if i in scns]
    held = [scns[i] for i in HELD_IDS if i in scns]
    spread = {}
    for m in methods:
        warm = warmup_steps(m, TRAIN_DAYS, WINDOW_DAYS, GUARD_DAYS)
        parts = []
        for s in dev:
            d = drop_residual(estimated(s.pressures, m), s.pressures)
            parts.append(d[normal_mask(s, warm)])
        nd = pd.concat(parts)
        spread[m] = dict(n_normal_steps=len(nd), median_node_q999=float(nd.quantile(0.999).median()),
                         median_node_q99=float(nd.quantile(0.99).median()), median_node_std=float(nd.std().median()))
    pd.DataFrame(spread).T.to_csv(out / "baseline_choice.csv")
    chosen = min(methods, key=lambda m: spread[m]["median_node_q999"])        # declared rule: smaller normal spread wins
    print("\nEstimated-baseline comparison on DEVELOPMENT normal data (drop, metres):")
    print(pd.DataFrame(spread).T.round(3).to_string()); print("chosen baseline:", chosen)

    warm = warmup_steps(chosen, TRAIN_DAYS, WINDOW_DAYS, GUARD_DAYS)
    base_est = {s.sid: estimated(s.pressures, chosen) for s in scns.values()}
    drop_est = {sid: drop_residual(base_est[sid], scns[sid].pressures) for sid in scns}
    dev_normal = pd.concat([drop_est[s.sid][normal_mask(s, warm)] for s in dev])

    grid = []
    for q in GRID_Q:
        tau = calibrate_thresholds(dev_normal, q, acc)
        for k in GRID_K:
            fa, hits, n_ev, ev_any = [], [], 0, 0
            for s in dev:
                st = alarm_state(drop_est[s.sid], tau, k)
                fa.append(false_alarm_rate(st, normal_mask(s, warm)))
                evs = [e for e in s.events if e.start >= warm]
                if evs:
                    sc = score_events(st, drop_est[s.sid], evs, HORIZON_DAYS * 48)
                    hits.append(sc); n_ev += len(evs); ev_any += int(sc.groupby("event")["detected"].any().sum())
            far = pd.concat(fa, axis=1).mean(axis=1)
            allh = pd.concat(hits) if hits else pd.DataFrame(columns=["detected"])
            grid.append(dict(q=q, k=k, median_tau_m=round(float(tau.median()), 3),
                             fa_median_per_day=round(float(far.median()), 4), fa_p90_per_day=round(float(far.quantile(.9)), 4),
                             dev_events_evaluable=n_ev, dev_events_any_node=ev_any,
                             dev_node_event_rate=round(float(allh["detected"].mean()), 4) if len(allh) else np.nan))
    G = pd.DataFrame(grid); G.to_csv(out / "level_b_grid_development.csv", index=False)
    print("\nLEVEL B grid on DEVELOPMENT data (false alarms: episodes per node per day):")
    print(G.to_string(index=False))

    ok = G[(G.fa_median_per_day <= FA_MEDIAN_MAX) & (G.fa_p90_per_day <= FA_P90_MAX)]
    if len(ok):
        pick = ok.sort_values(["dev_events_any_node", "dev_node_event_rate", "k", "q"], ascending=[False, False, True, True]).iloc[0]
        settings = dict(baseline=chosen, q=float(pick.q), persistence=int(pick.k), accuracy_floor_m=acc,
                        train_days=TRAIN_DAYS, window_days=WINDOW_DAYS, guard_days=GUARD_DAYS,
                        selection_rule=f"fa_median<={FA_MEDIAN_MAX} and fa_p90<={FA_P90_MAX} per node per day on development "
                                       "normal data; then most development events detected by any node; then node-event rate; "
                                       "then smaller k; then smaller q")
    else:
        settings = None
        print("\nNO setting meets the declared false-alarm targets on development data.")
    tau_final = calibrate_thresholds(dev_normal, settings["q"], acc) if settings else None
    if settings:
        settings["thresholds_m"] = {k: round(float(v), 4) for k, v in tau_final.items()}
        (out / "detector_settings_FROZEN.json").write_text(json.dumps(settings, indent=2))
        print("\nFROZEN settings:", {k: v for k, v in settings.items() if k not in ("thresholds_m", "selection_rule")})

    # ------------------------------------------------ held-out evaluation with FROZEN settings
    if settings and held:
        rows, fa = [], []
        for s in held:
            st = alarm_state(drop_est[s.sid], tau_final, settings["persistence"])
            fa.append(false_alarm_rate(st, normal_mask(s, warm)))
            for i, e in enumerate(s.events):
                if e.start < warm:
                    rows.append(dict(scenario=s.sid, event=i, status="NOT EVALUABLE: onset before baseline warm-up (day %.0f)" % (e.start / 48)))
                    continue
                sc = score_events(st, drop_est[s.sid], [e], HORIZON_DAYS * 48)
                rows.append(dict(scenario=s.sid, event=i, status="evaluated", onset_day=round(e.start / 48, 1),
                                 nodes_detecting=int(sc.detected.sum()), nodes=len(sc),
                                 first_delay_h=float(sc.delay_h.min()), best_max_drop_m=round(float(sc.max_drop_m.max()), 3)))
        H = pd.DataFrame(rows); H.to_csv(out / "level_b_heldout.csv", index=False)
        print("\nHELD-OUT (frozen settings). False alarms per node per day (median over nodes): %.4f" %
              float(pd.concat(fa, axis=1).mean(axis=1).median()))
        print(H.to_string(index=False))

    # ------------------------------------------------ sign check: do rises matter?
    rc = []
    for sid, b in matched.items():
        s = scns[sid]
        mask = np.zeros(len(b), dtype=bool)
        for e in s.events:
            mask[e.start: min(e.end, e.start + HORIZON_DAYS * 48, len(b))] = True
        rc.append(dict(scenario=sid, **rise_check(b, s.pressures.iloc[:len(b)], mask, acc)))
    pd.DataFrame(rc).to_csv(out / "sign_check_leaks.csv", index=False)
    print("\nSIGN CHECK (matched baseline, leak windows, threshold = accuracy): do pressure RISES exceed the threshold?")
    print(pd.DataFrame(rc).round(4).to_string(index=False))
    rise_q, drop_q = (-dev_normal).quantile(0.999).median(), dev_normal.quantile(0.999).median()
    print("Normal-period 99.9th percentile (median node): drop side %.3f m, rise side %.3f m" % (drop_q, rise_q))

    # ------------------------------------------------ plots for one example event
    ex = next(((s, i, e) for s in dev for i, e in enumerate(s.events) if e.start >= warm and s.sid in matched), None)
    if ex and settings:
        s, i, e = ex
        lo, hi = max(0, e.start - 3 * 48), min(len(s.pressures), e.start + 10 * 48)
        b = matched[s.sid]
        dA = drop_residual(b, s.pressures.iloc[:len(b)])
        mx = dA.iloc[e.start:min(len(dA), e.start + HORIZON_DAYS * 48)].max()
        strong, weak = mx.idxmax(), mx.idxmin()
        t = s.pressures.index[lo:hi]
        fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        ax[0].plot(t, s.pressures[strong].iloc[lo:hi], lw=.8, label="observed"); ax[0].plot(b.index[lo:min(hi, len(b))], b[strong].iloc[lo:hi], "--", lw=.8, label="matched baseline (no leak)")
        ax[0].set_ylabel("pressure (m)"); ax[0].set_title(f"{net} scenario {s.sid}: leak onset day {e.start/48:.0f}; node {strong} (strongest response)"); ax[0].legend()
        for n_, st_ in ((strong, "-"), (weak, ":")):
            ax[1].plot(b.index[lo:min(hi, len(b))], dA[n_].iloc[lo:hi], st_, lw=.9, label=f"matched drop, node {n_}")
        ax[1].axhline(acc, color="grey", lw=.8, label=f"sensor accuracy {acc:.3f} m"); ax[1].set_ylabel("drop (m)"); ax[1].legend()
        ax[2].plot(t, s.pressures[strong].iloc[lo:hi], lw=.8, label="observed"); ax[2].plot(t, base_est[s.sid][strong].iloc[lo:hi], "--", lw=.8, label=f"estimated baseline ({chosen})")
        ax[2].set_ylabel("pressure (m)"); ax[2].legend()
        for n_, st_ in ((strong, "-"), (weak, ":")):
            ax[3].plot(t, drop_est[s.sid][n_].iloc[lo:hi], st_, lw=.8, label=f"estimated-baseline drop, node {n_}")
            ax[3].axhline(tau_final[n_], color="C0" if n_ == strong else "C1", lw=.8, ls="--")
        ax[3].set_ylabel("drop (m)"); ax[3].legend(); ax[3].set_title(f"dashed lines = frozen thresholds (q={settings['q']}, persistence {settings['persistence']})")
        for a in ax:
            a.axvspan(s.pressures.index[e.start], s.pressures.index[min(e.end, hi) - 1], color="red", alpha=.1); a.grid(alpha=.3); a.set_xlim(t[0], t[-1])
        fig.tight_layout(); fig.savefig(out / f"example_{net}_scenario{s.sid}.png", dpi=130); plt.close(fig)
        print("saved example plot for scenario", s.sid, "strong node", strong, "weak node", weak)
    print(f"\nfinished in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
