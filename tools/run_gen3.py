import os, sys, time
net, days, seed, start, end, out, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], float(sys.argv[7])
os.environ.update(LG3_NET=net, LG3_DAYS=days, LG3_SEED=seed, LG3_OUT=out)
import leakg3pd_gen_linux as g
t0 = time.time(); done = []
for k in range(start, end + 1):
    folder = os.path.join(out, net, f"Scenario-{k}")
    if os.path.isfile(os.path.join(folder, "Labels.csv")): continue            # already complete
    if time.time() - t0 > budget: print("BUDGET reached, next is", k); break
    g.runScenarios(k); done.append(k)
print("generated:", done, f"{time.time()-t0:.0f}s")
