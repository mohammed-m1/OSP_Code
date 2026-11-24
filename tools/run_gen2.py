import os, sys, time
net, days, seed, start, end, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.environ.update(LG3_NET=net, LG3_DAYS=days, LG3_SEED=seed, LG3_OUT=out)
import leakg3pd_gen_linux as g
for k in range(start, end + 1):
    t = time.time(); g.runScenarios(k)          # deterministic: seeded per scenario
    print(f"[{net}] scenario {k} done {time.time()-t:.0f}s", flush=True)
