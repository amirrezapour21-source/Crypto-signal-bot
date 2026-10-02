# CANDIDATE 11 — DEPENDENCY-AWARE ROBUSTNESS
from collections import defaultdict

def cluster_boot(ev,B=20000,seed=20261002):
    rng=np.random.default_rng(seed)
    g=defaultdict(list)
    for e in ev:
        g[pd.Timestamp(e["ts"])].append(float(e["r"]))
    a=np.array([sum(v) for v in g.values()],float)
    x=rng.choice(a,(B,len(a)),replace=True).sum(1)
    return len(a),x.mean(),np.quantile(x,[.025,.5,.975]),(x>0).mean()

for name,ev in [("TP1.5",ev15),("TP2",ev20)]:
    n,m,ci,p=cluster_boot(ev)
    print(f"\n{name} — TIMESTAMP CLUSTER BOOTSTRAP")
    print("CLUSTERS",n)
    print("MEAN_TOTAL",round(m,2))
    print("CI95",np.round(ci,2))
    print("P_TOTAL_GT_0",round(p,4))

    d=pd.DataFrame(ev)
    d["month"]=pd.to_datetime(d["ts"],utc=True).dt.to_period("M")
    z=d.groupby("month")["r"].agg(["count","sum","mean"])
    print("\nMONTHLY")
    print(z.to_string())

print("\nDONE — Candidate 11 dependency-aware robustness")
