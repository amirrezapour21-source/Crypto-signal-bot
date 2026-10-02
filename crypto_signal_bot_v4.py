# CANDIDATE 11 — DEPENDENCY-AWARE ROBUSTNESS

def cluster_boot(ev,B=20000,seed=20261002):
    rng=np.random.default_rng(seed)
    d=pd.DataFrame(ev)
    d["net"]=d["r"]-0.003
    g=d.groupby("ts")["net"].sum().to_numpy(float)
    x=rng.choice(g,(B,len(g)),replace=True).sum(1)
    return len(g),x.mean(),np.quantile(x,[.025,.5,.975]),(x>0).mean()

for name,tp in [("TP1.5",1.5),("TP2",2.0)]:
    ev=ALL[tp]
    n,m,ci,p=cluster_boot(ev)

    print(f"\n{name} — TIMESTAMP CLUSTER BOOTSTRAP")
    print("CLUSTERS",n)
    print("MEAN_TOTAL",round(m,2))
    print("CI95",np.round(ci,2))
    print("P_TOTAL_GT_0",round(p,4))

    d=pd.DataFrame(ev)
    d["month"]=pd.to_datetime(d["ts"],utc=True).dt.strftime("%Y-%m")
    print("\nMONTHLY")
    print(d.groupby("month")["net"].agg(["count","sum","mean"]).to_string())

print("\nDONE — Candidate 11 dependency-aware robustness")
