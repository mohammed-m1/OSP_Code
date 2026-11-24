import sys,os,time,numpy as np,wntr,epyt
sys.path.insert(0,"src")
import coverage_optimiser as co
path=os.path.join(os.path.dirname(epyt.__file__),"networks","asce-tf-wdst","Net3.inp")
GPM=6.30902e-5; PSI_M=0.703070
wn0=wntr.network.WaterNetworkModel(path)
tot=sum(sum(d.base_value for d in j.demand_timeseries_list) for _,j in wn0.junctions())
print("Net3 total base demand GPM:",round(tot/GPM,0),"| 50-100 GPM as % of 10950:",round(50/10950*100,3),round(100/10950*100,3))
J=wn0.junction_name_list; n=len(J)
def run(t,leaks):
    wn=wntr.network.WaterNetworkModel(path)
    wn.options.hydraulic.demand_model="DDA"; wn.options.time.duration=0; wn.options.time.pattern_start=int(t)
    for node,q in leaks: wn.get_node(node).add_demand(q,None)
    return wntr.sim.WNTRSimulator(wn).run_sim().node["pressure"][J].iloc[0]
rng=np.random.default_rng(1)
events=[]
for _ in range(1000):
    k=int(rng.integers(1,3)); nodes=list(rng.choice(J,k,replace=False)); tot_q=rng.uniform(50,100)*GPM
    events.append([(nd,tot_q/k) for nd in nodes])
thr=0.05*PSI_M
print("threshold m head:",round(thr,4),"| EFENTO 1.7225 kPa in psi:",round(1.7225/6.894757,3))
for t_h in [0,12]:
    t=t_h*3600; base=run(t,[]); t0=time.time()
    D=np.array([(base-run(t,ev)).to_numpy()>thr for ev in events])
    print(f"t={t_h}h sims {time.time()-t0:.0f}s; events detectable by any junction: {D.any(1).mean():.4f}")
    for N in (1,5,10,25):
        r=co.maximise_coverage_mip(D,None,N,60,candidates=J)
        print(f"  N={N:2d} DCR={r.coverage*100:.2f}% status={r.status} nodes={r.nodes if N<=5 else '...'}")
    g=co.greedy_coverage(D,None,None,5,J); print("  greedy N=5:",round(g.coverage*100,2),g.nodes)
