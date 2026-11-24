"""
STEPS 3, 4, 6, 7 on one network (run after run_steps_1_2.py and run_step5_mdf.py).

    python src/run_steps_3_7.py --net-dir <Hanoi folder> --step12-out results/leakg3pd/Hanoi --out results/leakg3pd/Hanoi/final

STEP 3  event x candidate detection matrices (Level A idealised at several thresholds; Level B estimated baseline at the
        FROZEN setting plus sensitivity settings). Missed events stay in the denominator.
STEP 4  fixed-count placement on DEVELOPMENT events: exact MIP, greedy, sensitivity-ranked, random layouts; exhaustive
        search verifies the MIP for k <= 4. Every layout is then scored on HELD-OUT events.
STEP 6  coverage-oriented layouts versus minimum-detectable-flow-oriented layouts at the same sensor count (no merged score).
STEP 7  per-sensor and per-layout explanations, sensitivity of layouts to detector settings, layout-level false alarms.
"""
import argparse, glob, itertools, json, pickle, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_steps_1_2 as s12
from baseline import warmup_steps
from cost_func import load_device_catalogue
from coverage_optimiser import (exhaustive_coverage, greedy_coverage, layout_metrics, maximise_coverage_mip,
                                natural_key, random_layouts, sensitivity_ranked)
from detector import alarm_onsets, alarm_state, calibrate_thresholds, drop_residual, score_events
from leakg3pd_io import load_all

ROOT = Path(__file__).resolve().parents[1]
K_MAX, N_RANDOM, WEAK_DROP_M = 8, 1000, 1.0           # WEAK_DROP_M: ASSUMED label for 'weak' events (largest matched drop < 1 m)
A_MULTIPLES = (1, 2, 5)
B_SETTINGS = ((0.99, 3), (0.99, 4), (0.999, 3), (0.999, 6))     # (quantile, persistence); the frozen one must be among them
PRICE = None                                                     # AUD ex GST per logger, read from the catalogue


def to_matrix(score, keys, nodes, col):
    pv = score.pivot(index="key", columns="node", values=col).reindex(index=keys, columns=nodes)
    return pv.to_numpy()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    global PRICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-dir", required=True); ap.add_argument("--step12-out", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    s1 = Path(a.step12_out); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev_spec = load_device_catalogue(ROOT / "data" / "devices" / "pressure_logger_catalog.csv")["EFENTO_BT_689KPA"]
    PRICE, acc = dev_spec.price_aud_ex_gst, dev_spec.accuracy_m_head
    settings = json.loads((s1 / "detector_settings_FROZEN.json").read_text())
    H = s12.HORIZON_DAYS * 48

    scns = {s.sid: s for s in load_all(a.net_dir)}
    screen = pd.read_csv(s1 / "node_range_screen.csv", index_col=0, dtype={0: str})
    nodes = [str(n) for n in screen.index[screen.eligible]]
    for s in scns.values():
        s.pressures = s.pressures[nodes]
    net = next(iter(scns.values())).network
    warm = warmup_steps(settings["baseline"], s12.TRAIN_DAYS, s12.WINDOW_DAYS, s12.GUARD_DAYS)
    matched = {}
    for sid, s in scns.items():
        if s.is_leak_free:
            continue
        f = glob.glob(str(s1 / "cache" / f"matched_S{sid}_*.pkl"))
        matched[sid] = pickle.loads(Path(f[0]).read_bytes())[nodes]
    base_est = {sid: s12.estimated(s.pressures, settings["baseline"]) for sid, s in scns.items()}
    drop_est = {sid: drop_residual(base_est[sid], scns[sid].pressures) for sid in scns}
    dropA = {sid: drop_residual(matched[sid], scns[sid].pressures.iloc[:len(matched[sid])]).reindex(scns[sid].pressures.index) for sid in matched}
    print(f"== {net}: {len(scns)} scenarios, {len(nodes)} eligible candidates, price AUD {PRICE} ex GST, accuracy {acc:.3f} m")

    # ------------------------------------------------ STEP 3: event table
    rows = []
    for sid, s in sorted(scns.items()):
        for i, e in enumerate(s.events):
            md = dropA[sid].iloc[e.start:min(e.end, e.start + H)].max() if sid in dropA else None
            rows.append(dict(key=f"S{sid}_E{i}", scenario=sid, event=i, onset_day=round(e.start / 48, 1),
                             split="dev" if sid in s12.DEV_IDS else "held", kinds=",".join(sorted({l.kind for l in s.leaks})),
                             start=e.start, evaluable_B=e.start >= warm,
                             matched_max_drop_m=float(md.max()) if md is not None else np.nan))
    EV = pd.DataFrame(rows); EV["weak"] = EV.matched_max_drop_m < WEAK_DROP_M
    EV.to_csv(out / "events.csv", index=False)
    keys = list(EV.key)
    print("events:", len(EV), "| dev", int((EV.split == "dev").sum()), "| held-out", int((EV.split == "held").sum()),
          "| weak (matched drop <1 m)", int(EV.weak.sum()), "| evaluable for Level B", int(EV.evaluable_B.sum()))

    def level_a(m):
        parts = []
        for sid, s in scns.items():
            if sid not in dropA: continue
            st = alarm_state(dropA[sid], m * acc, 1)
            sc = score_events(st, dropA[sid], s.events, H); sc["key"] = [f"S{sid}_E{x}" for x in sc.event]; parts.append(sc)
        sc = pd.concat(parts)
        return to_matrix(sc, keys, nodes, "detected").astype(float) > 0.5, to_matrix(sc, keys, nodes, "delay_h") * 60, to_matrix(sc, keys, nodes, "max_drop_m")

    dev_normal = pd.concat([drop_est[s.sid][s12.normal_mask(s, warm)] for s in scns.values() if s.sid in s12.DEV_IDS])
    states_B = {}

    def level_b(q, k):
        tau = calibrate_thresholds(dev_normal, q, acc)
        parts, st_all = [], {}
        for sid, s in scns.items():
            st = alarm_state(drop_est[sid], tau, k); st_all[sid] = st
            evs = [(i, e) for i, e in enumerate(s.events) if e.start >= warm]
            if evs:
                sc = score_events(st, drop_est[sid], [e for _, e in evs], H)
                sc["key"] = [f"S{sid}_E{evs[x][0]}" for x in sc.event]; parts.append(sc)
        sc = pd.concat(parts)
        states_B[(q, k)] = st_all
        return to_matrix(sc, keys, nodes, "detected").astype(float) > 0.5, to_matrix(sc, keys, nodes, "delay_h") * 60

    settings_runs = {}                                    # name -> dict(level, D, T, mask)
    MD_A = None
    for m in A_MULTIPLES:
        D, T, MD = level_a(m)
        settings_runs[f"A_{m}x"] = dict(level="A", D=D, T=T, mask=np.ones(len(EV), bool))
        if m == 1: MD_A = MD
    pq = (settings["q"], settings["persistence"])
    runsB = list(dict.fromkeys([pq] + list(B_SETTINGS)))
    for q, k in runsB:
        D, T = level_b(q, k)
        settings_runs[f"B_q{q}_k{k}"] = dict(level="B", D=D, T=T, mask=EV.evaluable_B.to_numpy())
    primary_B = f"B_q{pq[0]}_k{pq[1]}"
    det_rows = []
    for name, r in settings_runs.items():
        for lvl in ("dev", "held"):
            msk = r["mask"] & (EV.split.to_numpy() == lvl)
            det_rows.append(dict(setting=name, split=lvl, events=int(msk.sum()),
                                 events_detected_by_any_candidate=int(r["D"][msk].any(axis=1).sum()),
                                 mean_share_of_candidates_detecting=round(float(r["D"][msk].mean()), 4)))
    pd.DataFrame(det_rows).to_csv(out / "step3_detection_summary.csv", index=False)
    np.savez_compressed(out / "step3_detection_matrices_primary.npz", D_A=settings_runs["A_1x"]["D"], T_A=settings_runs["A_1x"]["T"],
                        D_B=settings_runs[primary_B]["D"], T_B=settings_runs[primary_B]["T"], events=np.array(keys), nodes=np.array(nodes))
    print(pd.DataFrame(det_rows).to_string(index=False))

    # ------------------------------------------------ STEP 4: placement
    def split_idx(r, lvl):
        return np.flatnonzero(r["mask"] & (EV.split.to_numpy() == lvl))

    def score(r, idx_nodes, lvl, subset=None):
        ii = split_idx(r, lvl)
        if subset is not None: ii = ii[subset[ii]]
        m = layout_metrics(r["D"][ii], r["T"][ii], np.ones(len(ii)), idx_nodes)
        return m, len(ii)

    rows, layouts, verify = [], {}, []
    for name, r in settings_runs.items():
        idev = split_idx(r, "dev")
        Dd, Td = r["D"][idev], r["T"][idev]
        for k in range(1, K_MAX + 1):
            res = {"MIP": maximise_coverage_mip(Dd, None, k, 60.0, Td, nodes), "greedy": greedy_coverage(Dd, Td, None, k, nodes),
                   "sensitivity_ranked": sensitivity_ranked(Dd, Td, None, k, nodes)}
            if k <= 4:
                ex = exhaustive_coverage(Dd, Td, None, k, nodes)
                verify.append(dict(setting=name, k=k, mip_objective=res["MIP"].objective, exhaustive_objective=ex.objective,
                                   match=abs(res["MIP"].objective - ex.objective) < 1e-9, mip_status=res["MIP"].status))
            for meth, lr in res.items():
                layouts[(name, k, meth)] = lr.indexes
                md, nd = score(r, lr.indexes, "dev"); mh, nh = score(r, lr.indexes, "held")
                wk, nwk = score(r, lr.indexes, "held", EV.weak.to_numpy()); st, nst = score(r, lr.indexes, "held", ~EV.weak.to_numpy())
                rows.append(dict(setting=name, level=r["level"], k=k, method=meth, nodes=" ".join(lr.nodes), hardware_cost_aud_ex_gst=round(k * PRICE, 2),
                                 dev_events=nd, dev_coverage=round(md["coverage"], 4), held_events=nh, held_coverage=round(mh["coverage"], 4),
                                 held_detected=mh["n_detected"], held_mean_delay_h=round(mh["mean_delay"] / 60, 2) if np.isfinite(mh["mean_delay"]) else np.nan,
                                 held_weak_events=nwk, held_weak_coverage=round(wk["coverage"], 4) if nwk else np.nan,
                                 held_strong_events=nst, held_strong_coverage=round(st["coverage"], 4) if nst else np.nan,
                                 solver_status=lr.status, gap=lr.gap, runtime_s=round(lr.runtime_s, 3)))
            rnd = random_layouts(Dd, None, k, N_RANDOM, seed=k, T=Td, candidates=nodes)
            held_cov = np.array([score(r, row, "held")[0]["coverage"] for row in rnd["layouts"]])
            rows.append(dict(setting=name, level=r["level"], k=k, method="random_mean", nodes="", hardware_cost_aud_ex_gst=round(k * PRICE, 2),
                             dev_coverage=round(rnd["mean_coverage"], 4), held_coverage=round(float(held_cov.mean()), 4),
                             held_cov_p05=round(float(np.percentile(held_cov, 5)), 4), held_cov_p95=round(float(np.percentile(held_cov, 95)), 4),
                             held_events=int(len(split_idx(r, "held")))))
    P = pd.DataFrame(rows); P.to_csv(out / "step4_placement_results.csv", index=False)
    V = pd.DataFrame(verify); V.to_csv(out / "step4_solver_verification.csv", index=False)
    print("\nSolver check (MIP objective == exhaustive search, k<=4):", int(V.match.sum()), "of", len(V), "| MIP statuses:", V.mip_status.value_counts().to_dict())

    # ------------------------------------------------ STEP 7: layout-level false alarms (Level B primary)
    fa = []
    for k in range(1, K_MAX + 1):
        idx = layouts[(primary_B, k, "MIP")]
        for lvl, ids in (("dev", s12.DEV_IDS), ("held", s12.HELD_IDS)):
            ep, days = 0, 0.0
            for sid in ids:
                if sid not in scns: continue
                un = states_B[pq][sid].iloc[:, idx].any(axis=1)
                on = un & ~un.shift(1, fill_value=False)
                nm = s12.normal_mask(scns[sid], warm)
                ep += int(on[nm].sum()); days += nm.sum() / 48
            fa.append(dict(k=k, split=lvl, nodes=" ".join(nodes[i] for i in idx), episodes=ep, normal_days=round(days, 1),
                           layout_false_alarm_episodes_per_day=round(ep / days, 4), one_per_n_days=round(days / ep, 1) if ep else np.inf))
    FA = pd.DataFrame(fa); FA.to_csv(out / "step7_layout_false_alarms.csv", index=False)

    # ------------------------------------------------ STEP 7: explanations
    expl, alts = [], []
    for name in ("A_1x", primary_B):
        r = settings_runs[name]; idev, ihel = split_idx(r, "dev"), split_idx(r, "held")
        for k in (3, 5):
            idx = layouts[(name, k, "MIP")]
            full_d = layout_metrics(r["D"][idev], r["T"][idev], np.ones(len(idev)), idx)
            full_h = layout_metrics(r["D"][ihel], r["T"][ihel], np.ones(len(ihel)), idx)
            for i in idx:
                rest = [j for j in idx if j != i]
                own = r["D"][idev][:, i]; uniq = own & ~(r["D"][idev][:, rest].any(axis=1) if rest else False)
                oh = r["D"][ihel][:, i]; uh = oh & ~(r["D"][ihel][:, rest].any(axis=1) if rest else False)
                dd = layout_metrics(r["D"][idev], r["T"][idev], np.ones(len(idev)), rest); dh = layout_metrics(r["D"][ihel], r["T"][ihel], np.ones(len(ihel)), rest)
                weak_dev = idev[EV.weak.to_numpy()[idev]]
                expl.append(dict(setting=name, k=k, node=nodes[i], dev_events_detected_here=int(own.sum()), dev_unique_in_layout=int(uniq.sum()),
                                 dev_coverage_with=round(full_d["coverage"], 3), dev_coverage_if_removed=round(dd["coverage"], 3),
                                 held_events_detected_here=int(oh.sum()), held_unique_in_layout=int(uh.sum()),
                                 held_coverage_with=round(full_h["coverage"], 3), held_coverage_if_removed=round(dh["coverage"], 3),
                                 mean_max_drop_weak_dev_events_m=round(float(np.nanmean(MD_A[weak_dev, i])), 3) if len(weak_dev) else np.nan,
                                 overlap_with_other_sensors_jaccard=round(float(np.mean([jaccard(np.flatnonzero(r["D"][idev][:, i]), np.flatnonzero(r["D"][idev][:, j])) for j in rest])), 3) if rest else np.nan))
            if k <= 5:                                    # equal-count alternatives: all layouts with the same development objective
                target = full_d["objective"]; cnt = 0; best = []
                for combo in itertools.combinations(range(len(nodes)), k):
                    o = float(r["D"][idev][:, list(combo)].any(axis=1).sum())
                    if abs(o - target) < 1e-9:
                        cnt += 1
                        if cnt <= 2000: best.append((combo, layout_metrics(r["D"][ihel], r["T"][ihel], np.ones(len(ihel)), combo)["coverage"]))
                cv = np.array([c for _, c in best])
                alts.append(dict(setting=name, k=k, dev_objective_events=target, n_equal_dev_objective_layouts=cnt,
                                 held_coverage_min=round(float(cv.min()), 3), held_coverage_median=round(float(np.median(cv)), 3), held_coverage_max=round(float(cv.max()), 3),
                                 mip_layout_held_coverage=round(full_h["coverage"], 3)))
    E = pd.DataFrame(expl); E.to_csv(out / "step7_sensor_explanations.csv", index=False)
    AL = pd.DataFrame(alts); AL.to_csv(out / "step7_equal_count_alternatives.csv", index=False)

    # sensitivity of layouts to detector settings (Jaccard against the primary Level-B / Level-A layout)
    sens = []
    for name in settings_runs:
        ref = "A_1x" if name.startswith("A") else primary_B
        for k in (3, 5):
            a_, b_ = layouts[(name, k, "MIP")], layouts[(ref, k, "MIP")]
            sens.append(dict(setting=name, reference=ref, k=k, layout=" ".join(nodes[i] for i in a_), jaccard_vs_reference=round(jaccard(a_, b_), 3),
                             held_coverage=float(P[(P.setting == name) & (P.k == k) & (P.method == "MIP")].held_coverage.iloc[0])))
    S = pd.DataFrame(sens); S.to_csv(out / "step7_setting_sensitivity.csv", index=False)

    # ------------------------------------------------ STEP 6: coverage-oriented vs MDF-oriented layouts
    MM = {}
    for nm in ("idealised", "realistic"):
        d = pd.read_csv(out / f"mdf_matrix_{nm}.csv", dtype={"leak_node": str, "sensor": str})
        fmax, fmin = d.min_flow_m3s.max(), 0.0005
        d["value"] = d.min_flow_m3s.fillna(1e9)            # unresolved never becomes finite: huge sentinel, counted separately
        MM[nm] = (d.pivot(index="leak_node", columns="sensor", values="value").reindex(index=nodes, columns=nodes).to_numpy(),
                  d.pivot(index="leak_node", columns="sensor", values="status").reindex(index=nodes, columns=nodes).to_numpy())

    def mdf_stats(idx, nm):
        v, st = MM[nm]; best = v[:, idx].min(axis=1); resolved = best < 1e8
        return dict(resolved_leak_nodes=int(resolved.sum()), mean_mdf_resolved_m3s=float(best[resolved].mean()) if resolved.any() else np.nan,
                    unresolved_share=float(1 - resolved.mean()))

    def mdf_best(nm, k):
        v = MM[nm][0]; n = v.shape[1]; best, bi = (-1, np.inf), None
        for combo in itertools.combinations(range(n), k) if k <= 3 else []:
            b = v[:, list(combo)].min(axis=1); key = ((b < 1e8).sum() * -1, b[b < 1e8].mean() if (b < 1e8).any() else np.inf)
            if bi is None or key < best: best, bi = key, list(combo)
        if bi is None:                                       # greedy for larger k
            bi = []
            for _ in range(k):
                cands = [c for c in range(n) if c not in bi]
                def key(c):
                    b = v[:, bi + [c]].min(axis=1); return (-(b < 1e8).sum(), b[b < 1e8].mean() if (b < 1e8).any() else np.inf)
                bi.append(min(cands, key=key))
        return bi

    cmp_rows = []
    rA, rB = settings_runs["A_1x"], settings_runs[primary_B]
    for k in range(1, K_MAX + 1):
        sets = {"coverage_A_MIP": layouts[("A_1x", k, "MIP")], "coverage_B_MIP": layouts[(primary_B, k, "MIP")],
                "mdf_idealised": mdf_best("idealised", k), "mdf_realistic": mdf_best("realistic", k)}
        for nm, idx in sets.items():
            hA, _ = score(rA, idx, "held"); hB, _ = score(rB, idx, "held")
            si, sr = mdf_stats(idx, "idealised"), mdf_stats(idx, "realistic")
            cmp_rows.append(dict(k=k, objective=nm, nodes=" ".join(nodes[i] for i in idx), held_coverage_A=round(hA["coverage"], 4), held_coverage_B=round(hB["coverage"], 4),
                                 mdf_idealised_resolved=si["resolved_leak_nodes"], mean_mdf_idealised_m3s=round(si["mean_mdf_resolved_m3s"], 4) if np.isfinite(si["mean_mdf_resolved_m3s"]) else np.nan,
                                 mdf_realistic_resolved=sr["resolved_leak_nodes"], mean_mdf_realistic_m3s=round(sr["mean_mdf_resolved_m3s"], 4) if np.isfinite(sr["mean_mdf_resolved_m3s"]) else np.nan))
    C = pd.DataFrame(cmp_rows); C.to_csv(out / "step6_objective_comparison.csv", index=False)

    # ------------------------------------------------ plots
    import wntr
    base_inp = sorted(Path(a.net_dir).glob("*.inp"))[0]; wn = wntr.network.WaterNetworkModel(str(base_inp))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for a_, nm, title in ((ax[0], "A_1x", f"Level A: matched baseline, threshold {acc:.3f} m"), (ax[1], primary_B, f"Level B: estimated baseline, frozen setting")):
        d = P[P.setting == nm]
        for meth, st in (("MIP", "o-"), ("greedy", "s--"), ("sensitivity_ranked", "^:")):
            g = d[d.method == meth]; a_.plot(g.k, g.held_coverage * 100, st, label=meth)
        g = d[d.method == "random_mean"]; a_.plot(g.k, g.held_coverage * 100, "k-", lw=1, label="random (mean)")
        a_.fill_between(g.k, g.held_cov_p05 * 100, g.held_cov_p95 * 100, color="grey", alpha=.25, label="random 5th-95th pct")
        a_.set_title(title, fontsize=10); a_.set_xlabel("number of loggers k (hardware cost = k x AUD %.2f ex GST)" % PRICE); a_.grid(alpha=.3)
    ax[0].set_ylabel("HELD-OUT events detected (%)"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_performance_vs_sensors.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    k_show = 3
    for a_, (nm, idx, title) in zip(ax, (("A", layouts[("A_1x", k_show, "MIP")], f"coverage, Level A (k={k_show})"), ("B", layouts[(primary_B, k_show, "MIP")], f"coverage, Level B (k={k_show})"),
                                         ("mi", mdf_best("idealised", k_show), f"min detectable flow, idealised (k={k_show})"), ("mr", mdf_best("realistic", k_show), f"min detectable flow, realistic (k={k_show})"))):
        chosen = [nodes[i] for i in idx]
        attr = pd.Series({n: (1.0 if n in chosen else 0.0) for n in wn.junction_name_list})
        wntr.graphics.plot_network(wn, node_attribute=attr, node_size=[110 if v else 18 for v in attr], node_cmap=plt.cm.coolwarm, ax=a_, title=title, add_colorbar=False, node_labels=False)
        for n in chosen:
            x, y = wn.get_node(n).coordinates; a_.text(x, y, " " + n, fontsize=8, weight="bold")
    fig.tight_layout(); fig.savefig(out / "fig2_layout_maps_k3.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for obj, st in (("coverage_A_MIP", "o-"), ("coverage_B_MIP", "s-"), ("mdf_idealised", "^--"), ("mdf_realistic", "v--")):
        g = C[C.objective == obj]; ax[0].plot(g.k, g.held_coverage_A * 100, st, label=obj); ax[1].plot(g.k, g.mean_mdf_idealised_m3s * 1000, st, label=obj)
    ax[0].set_title("held-out events detected, Level A (%)", fontsize=10); ax[1].set_title("mean minimum detectable flow, idealised (L/s)  - lower is better", fontsize=10)
    for a_ in ax: a_.set_xlabel("number of loggers k"); a_.grid(alpha=.3)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig3_coverage_vs_detectable_flow.png", dpi=140); plt.close(fig)

    wkdf = P[(P.method == "MIP") & (P.setting.isin(["A_1x", primary_B]))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a_, nm in zip(ax, ("A_1x", primary_B)):
        g = wkdf[wkdf.setting == nm]; a_.plot(g.k, g.held_strong_coverage * 100, "o-", label="strong events (matched drop >= 1 m)"); a_.plot(g.k, g.held_weak_coverage * 100, "s-", label="weak events (< 1 m)")
        a_.set_title(nm, fontsize=10); a_.set_xlabel("number of loggers k"); a_.grid(alpha=.3)
    ax[0].set_ylabel("held-out events detected (%)"); ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig4_by_event_magnitude.png", dpi=140); plt.close(fig)

    # ------------------------------------------------ headline numbers for the write-up
    pr = lambda nm, k, m: P[(P.setting == nm) & (P.k == k) & (P.method == m)].iloc[0]
    head = {"candidates": len(nodes), "scenarios": len(scns), "events": len(EV), "dev_events_A": int(len(split_idx(rA, "dev"))), "held_events_A": int(len(split_idx(rA, "held"))),
            "dev_events_B": int(len(split_idx(rB, "dev"))), "held_events_B": int(len(split_idx(rB, "held"))), "primary_B": primary_B}
    for k in (1, 2, 3, 5, 8):
        for nm in ("A_1x", primary_B):
            head[f"{nm}_k{k}_MIP_held"] = float(pr(nm, k, "MIP").held_coverage); head[f"{nm}_k{k}_random_held"] = float(pr(nm, k, "random_mean").held_coverage)
            head[f"{nm}_k{k}_greedy_held"] = float(pr(nm, k, "greedy").held_coverage); head[f"{nm}_k{k}_sens_held"] = float(pr(nm, k, "sensitivity_ranked").held_coverage)
    head["solver_verification_match"] = f"{int(V.match.sum())}/{len(V)}"
    (out / "headline_numbers.json").write_text(json.dumps(head, indent=2, default=float))
    print("\nPERFORMANCE ON HELD-OUT EVENTS (MIP layouts)"); print(P[(P.method == "MIP") & (P.setting.isin(["A_1x", primary_B]))][["setting", "k", "nodes", "dev_coverage", "held_coverage", "held_weak_coverage", "held_strong_coverage"]].to_string(index=False))
    print(f"\nfinished in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
