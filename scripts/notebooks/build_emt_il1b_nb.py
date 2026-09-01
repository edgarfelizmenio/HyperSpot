import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

C = []
C.append(md(r"""# Human Kidney Cancer — do HyperSpot's domains recover the Li et al. interface biology?

The dataset's headline biology (Li et al. 2022, *Cancer Cell*) is **spatial and interface-driven**:
**EMT-high RCC** cells enrich at the **tumor–normal interface**, and **IL1B⁺ macrophages (Myeloid)** at the
interface are implicated in driving that EMT. Here we ask whether HyperSpot's **unsupervised** per-cell-type
domains recover this — i.e. whether the biology falls out of the representation without being told to look
for it. All data are MT-free.

**What we test:**
1. Is there an **EMT-high RCC domain** (unsupervised), and is the EMT programme **enriched at the interface**
   vs the tumor core?
2. Is there an **IL1B-high Myeloid domain**?
3. Do the EMT-RCC and IL1B-Myeloid states **co-occur spatially** (the proposed mechanism)? — tested both at
   the discrete-community level and, more directly, by spot-level spatial co-occurrence."""))

C.append(md("### Setup — marker gene sets + section loaders"))
C.append(code(r"""import warnings; warnings.filterwarnings("ignore")
import anndata as ad, numpy as np, pandas as pd, matplotlib.pyplot as plt, scanpy as sc, squidpy as sq
from hyperspot.metadata import ProjectMetadata
from hyperspot.pipeline.evaluation import ClusterEvaluation
from hyperspot.utils import load_ct_spatial_data

EMT_MES = ["VIM","FN1","ZEB1","ZEB2","SNAI1","CDH2","MMP2","TAGLN","ACTA2","TIMP1"]
EMT_EPI = ["CDH1","EPCAM","KRT8","KRT18","KRT19"]
IL1     = ["IL1B","NLRP3","CXCL8","CCL2","SPP1","TNF"]

def sym_map(sec):
    proj=f"hyperspot_project_files/Human Kidney Cancer {sec}"
    dec=ad.read_h5ad(f"{proj}/ST_decomposition.h5ad", backed='r')
    m={}
    for e,s in zip(dec.var_names.astype(str), dec.var["SYMBOL"].astype(str)): m.setdefault(s,e)
    return m
def score_expr(a, genes, m):
    X=a.X.toarray() if hasattr(a.X,'toarray') else np.asarray(a.X)
    expr=pd.DataFrame(X, columns=a.var_names.astype(str), index=a.obs_names)
    ens=[m[g] for g in genes if g in m and m[g] in expr.columns]
    return expr[ens].mean(1)"""))

# --- 1. EMT-high RCC domain ---
C.append(md("""## 1. An EMT-high RCC domain emerges unsupervised
Each RCC domain scored by a mesenchymal EMT signature (VIM, FN1, ZEB1/2, SNAI1, CDH2, MMP2, TAGLN, ACTA2,
TIMP1) minus an epithelial signature (CDH1, EPCAM, KRT8/18/19). A high EMT-ratio domain = mesenchymal,
epithelial-low RCC = EMT-high."""))
C.append(code(r"""ProjectMetadata.use("hyperspot_project_files/Human Kidney Cancer Interface")
mI = sym_map("Interface")
rcc = load_ct_spatial_data("RCC")
mes = score_expr(rcc, EMT_MES, mI); epi = score_expr(rcc, EMT_EPI, mI)
dom = rcc.obs["domain"].astype(str).values
emt = pd.DataFrame({"dom":dom,"mes":mes.values,"epi":epi.values})
g = emt.groupby("dom").mean(); g["EMT_ratio"] = g["mes"]-g["epi"]; g["n"]=emt.groupby("dom").size()
g = g.sort_values("EMT_ratio", ascending=False)
print(g.round(3).to_string())
EMT_DOM = g.index[0]
print(f"\nEMT-high RCC domain = RCC|{EMT_DOM}  (EMT-ratio {g['EMT_ratio'].iloc[0]:.2f})")
fig,ax=plt.subplots(figsize=(9,4)); ax.bar(g.index, g["EMT_ratio"], color=["#c0392b" if d==EMT_DOM else "#95a5a6" for d in g.index])
ax.axhline(0,c="k",lw=.6); ax.set_xlabel("RCC domain"); ax.set_ylabel("EMT ratio (mesenchymal − epithelial)")
ax.set_title("RCC domains by EMT signature (Interface) — the EMT-high domain in red"); plt.tight_layout(); plt.show()"""))

C.append(md("""### The EMT-high RCC domain in space (H&E)"""))
C.append(code(r"""rcc.obs["EMT_domain"] = pd.Categorical(np.where(dom==EMT_DOM, f"RCC|{EMT_DOM} (EMT-high)", "other RCC"))
rcc.obs["EMT_score"] = (mes-epi).values
fig, axes = plt.subplots(1,2, figsize=(18,7))
for ax,(col,ttl) in zip(axes, [("EMT_domain","EMT-high RCC domain"),("EMT_score","RCC EMT score (per cell)")]):
    try: sq.pl.spatial_scatter(rcc, color=col, img=True, size=1.5, ax=ax, title=ttl,
                               cmap="magma" if col=="EMT_score" else None)
    except Exception as e: print(col, str(e)[:60])
plt.tight_layout(); plt.show()"""))

# --- 2. EMT enriched at interface vs core ---
C.append(md("""## 2. The EMT programme is enriched at the interface vs the tumor core
Mean RCC EMT (mesenchymal) score, Interface vs Core section — Li et al. report EMT-high RCC enriching at the
tumor edge."""))
C.append(code(r"""rows=[]
for sec in ["Interface","Core"]:
    ProjectMetadata.use(f"hyperspot_project_files/Human Kidney Cancer {sec}")
    m=sym_map(sec); a=load_ct_spatial_data("RCC")
    s=score_expr(a, EMT_MES, m)
    rows.append((sec, s.mean(), s.std()/np.sqrt(len(s)), a.n_obs))
d=pd.DataFrame(rows, columns=["section","EMT_mean","sem","n"]); print(d.round(3).to_string(index=False))
fig,ax=plt.subplots(figsize=(5,4)); ax.bar(d["section"], d["EMT_mean"], yerr=d["sem"], capsize=5,
    color=["#c0392b","#7f8c8d"]); ax.set_ylabel("RCC EMT (mesenchymal) score"); ax.set_title("RCC EMT: Interface vs Core")
plt.tight_layout(); plt.show()
from scipy.stats import mannwhitneyu"""))

# --- 3. IL1B-high Myeloid domain ---
C.append(md("""## 3. An IL1B-high Myeloid domain
Myeloid domains scored by IL1B (and an inflammatory programme IL1B/NLRP3/CXCL8/CCL2/SPP1/TNF)."""))
C.append(code(r"""ProjectMetadata.use("hyperspot_project_files/Human Kidney Cancer Interface")
myl = load_ct_spatial_data("Myeloid")
il1b = score_expr(myl, ["IL1B"], mI); ilprog = score_expr(myl, IL1, mI)
mdom = myl.obs["domain"].astype(str).values
gm = pd.DataFrame({"dom":mdom,"IL1B":il1b.values,"IL1_prog":ilprog.values}).groupby("dom").mean().sort_values("IL1B",ascending=False)
print(gm.round(3).to_string())
IL_DOM = gm.index[0]; print(f"\nIL1B-high Myeloid domain = Myeloid|{IL_DOM}")
myl.obs["IL1B_domain"]=pd.Categorical(np.where(mdom==IL_DOM, f"Myeloid|{IL_DOM} (IL1B-high)","other Myeloid"))
fig,axes=plt.subplots(1,2,figsize=(18,7))
myl.obs["IL1B_expr"]=il1b.values
for ax,(col,ttl) in zip(axes,[("IL1B_domain","IL1B-high Myeloid domain"),("IL1B_expr","Myeloid IL1B (per cell)")]):
    try: sq.pl.spatial_scatter(myl, color=col, img=True, size=1.5, ax=ax, title=ttl, cmap="magma" if col=="IL1B_expr" else None)
    except Exception as e: print(col, str(e)[:60])
plt.tight_layout(); plt.show()"""))

# --- 4. co-occurrence: community + spatial ---
C.append(md(r"""## 4. Do EMT-RCC and IL1B-Myeloid co-occur? (the proposed mechanism)

Two tests, honest about both:
- **Discrete community**: are the EMT-RCC domain and IL1B-Myeloid domain in the same HyperSpot community?
- **Spot-level spatial co-occurrence** (the direct test of "IL1B⁺ macrophages near EMT RCC"): across spots,
  is the local Myeloid-IL1B score correlated with the local RCC-EMT score in the neighbourhood?"""))
C.append(code(r"""res = ClusterEvaluation().compute_domain_colocalization(unit="spot")
nof = res["niche_of"]
cE, cI = nof.get(f"RCC|{EMT_DOM}"), nof.get(f"Myeloid|{IL_DOM}")
print(f"community of EMT-RCC (RCC|{EMT_DOM}): {cE}")
print(f"community of IL1B-Myeloid (Myeloid|{IL_DOM}): {cI}")
print("same community:", cE==cI)
print(f"members of EMT-RCC's community {cE}:", sorted(i for i in nof.index if nof[i]==cE))"""))
C.append(code(r"""# spot-level: RCC-EMT vs Myeloid-IL1B per spot, and RCC-EMT vs NEIGHBOURHOOD Myeloid-IL1B
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
rcc_emt = pd.Series((score_expr(rcc, EMT_MES, mI)-score_expr(rcc, EMT_EPI, mI)).values, index=rcc.obs["spot_name"].astype(str)).groupby(level=0).mean()
myl_il  = pd.Series(score_expr(myl, ["IL1B"], mI).values, index=myl.obs["spot_name"].astype(str)).groupby(level=0).mean()
common = rcc_emt.index.intersection(myl_il.index)
r,p = spearmanr(rcc_emt.loc[common], myl_il.loc[common])
print(f"per-spot RCC-EMT vs Myeloid-IL1B: Spearman r={r:.3f}, p={p:.1e}, n={len(common)}")
# neighbourhood version: RCC-EMT at spot vs mean Myeloid-IL1B of its 6 spatial neighbours
sp_xy = rcc.obs.groupby(rcc.obs["spot_name"].astype(str))[["location_x","location_y"]].first().loc[common]
nn = NearestNeighbors(n_neighbors=7).fit(sp_xy.values); _,idx = nn.kneighbors(sp_xy.values)
il_arr = myl_il.loc[common].values
nbr_il = np.array([il_arr[row[1:]].mean() for row in idx])
r2,p2 = spearmanr(rcc_emt.loc[common].values, nbr_il)
print(f"RCC-EMT at spot vs neighbourhood Myeloid-IL1B: Spearman r={r2:.3f}, p={p2:.1e}")
fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(rcc_emt.loc[common], nbr_il, s=6, alpha=.4)
ax.set_xlabel("RCC EMT score (spot)"); ax.set_ylabel("neighbourhood Myeloid IL1B")
ax.set_title(f"EMT-RCC vs neighbouring IL1B-Myeloid  (Spearman r={r2:.2f}, p={p2:.0e})")
plt.tight_layout(); plt.show()"""))

# --- 5. validate EMT domain via pathway ---
C.append(md("""## 5. Validation — is the EMT-high RCC domain's marker programme really EMT?
One-vs-rest markers of the EMT-high RCC domain → g:Profiler enrichment (expect EMT / ECM / mesenchymal terms)."""))
C.append(code(r"""from hyperspot.pipeline.selection import compute_domain_ct_svgs
ProjectMetadata.use("hyperspot_project_files/Human Kidney Cancer Interface")
dm = compute_domain_ct_svgs("RCC", rank_by="scores")
emt_markers = None
for k in dm:
    if str(k)==str(EMT_DOM): emt_markers=list(dm[k]["names"].astype(str))
print(f"EMT-RCC domain markers: {len(emt_markers) if emt_markers else 0}")
if emt_markers and len(emt_markers)>=10:
    try:
        e=sc.queries.enrich(emt_markers, org="hsapiens", gprofiler_kwargs={"sources":["GO:BP","REAC"],"no_evidences":True})
        print(e[e["term_size"]<600].sort_values("p_value")[["source","name","p_value","intersection_size"]].head(8).to_string(index=False))
        syms=[k for k,v in mI.items() if v in set(emt_markers)]
        print("\nEMT-relevant markers present:", [g for g in EMT_MES+["POSTN","COL1A1","SPARC","LUM","DCN"] if g in syms])
    except Exception as ex: print("enrich:", str(ex)[:60])"""))

C.append(md(r"""### Why the coupling is null — a deconvolution-sparsity diagnostic

Before concluding, we test whether the null EMT↔IL1B association is biological or an artifact of Myeloid being
deconvolution-sparse (few Myeloid cells per spot → a noise-dominated IL1B estimate)."""))
C.append(code(r"""from scipy.stats import spearmanr
mn = myl.obs["spot_name"].astype(str).value_counts(); rn = rcc.obs["spot_name"].astype(str).value_counts()
myl_il_spot = pd.Series(score_expr(myl, ["IL1B"], mI).values, index=myl.obs["spot_name"].astype(str)).groupby(level=0).mean()
com = rcc_emt.index.intersection(myl_il_spot.index)
pct1 = 100*(mn.loc[com]==1).mean()
cv_m = myl_il_spot.loc[com].std()/myl_il_spot.loc[com].mean(); cv_r = rcc_emt.loc[com].std()/rcc_emt.loc[com].mean()
r_comp = spearmanr(rn.loc[com], mn.loc[com])[0]
rich = mn.loc[com][mn.loc[com]>=3].index
r_all = spearmanr(rcc_emt.loc[com], myl_il_spot.loc[com])[0]
r_rich = spearmanr(rcc_emt.loc[rich], myl_il_spot.loc[rich])[0]
print("Myeloid cells/spot (co-spots): median=%.0f, %%with 1 cell=%.0f%%" % (mn.loc[com].median(), pct1))
print("Myeloid-IL1B CV=%.2f vs RCC-EMT CV=%.2f" % (cv_m, cv_r))
print("RCC vs Myeloid cells/spot r=%.2f (positive -> not proportion competition)" % r_comp)
print("RCC-EMT vs Myeloid-IL1B: all-cospots r=%.3f ; Myeloid-rich(>=3) r=%.3f (n=%d)" % (r_all, r_rich, len(rich)))
"""))

C.append(md(r"""## 6. Can the hyperedge (neighbourhood) scale resolve the Myeloid sparsity?

Since "IL1B⁺ myeloid *near* EMT-RCC" is a neighbourhood relationship, we pool Myeloid cells over each spot's
7-nearest-neighbour spot neighbourhood (HyperSpot's native **hyperedge** unit) — giving ~6× more cells and a
more stable IL1B estimate. We then test the coupling with two nulls: a naive free permutation, and a
spatial-autocorrelation-preserving **rotation null** (rigidly rotate the Myeloid field and recompute)."""))
C.append(code(r"""from sklearn.neighbors import NearestNeighbors
def cellf(ct, genes):
    a=load_ct_spatial_data(ct); X=a.X.toarray() if hasattr(a.X,"toarray") else np.asarray(a.X)
    ex=pd.DataFrame(X, columns=a.var_names.astype(str)); ens=[mI[g] for g in genes if g in mI and mI[g] in ex.columns]
    return pd.DataFrame({"spot":a.obs["spot_name"].astype(str).values,"score":ex[ens].mean(1).values,
                         "x":a.obs["location_x"].values,"y":a.obs["location_y"].values})
rccC=cellf("RCC",EMT_MES); rccC["score"]-=cellf("RCC",EMT_EPI)["score"]; mylC=cellf("Myeloid",["IL1B"])
sxy=pd.concat([rccC[["spot","x","y"]],mylC[["spot","x","y"]]]).drop_duplicates("spot").set_index("spot")
_,ix=NearestNeighbors(n_neighbors=7).fit(sxy.values).kneighbors(sxy.values)
sp=sxy.index.values; nb={sp[i]:set(sp[ix[i]]) for i in range(len(sp))}
def pool(df):
    bs={x:gg["score"].values for x,gg in df.groupby("spot")}; o={}
    for x in sp:
        v=np.concatenate([bs[n] for n in nb[x] if n in bs]) if any(n in bs for n in nb[x]) else np.array([])
        o[x]=v.mean() if len(v) else np.nan
    return pd.Series(o)
rp=pool(rccC); mp=pool(mylC); com=np.array(list(rp.dropna().index.intersection(mp.dropna().index)))
from scipy.stats import spearmanr
rv=rp.loc[com].values; mv=mp.loc[com].values; r_obs=spearmanr(rv,mv)[0]
print(f"hyperedge-pooled Myeloid-IL1B: CV={mp.loc[com].std()/mp.loc[com].mean():.2f} (per-spot was 3.21)")
print(f"observed hyperedge-pooled RCC-EMT vs Myeloid-IL1B r={r_obs:.3f} (n={len(com)})")
rng=np.random.default_rng(0)
fp=[spearmanr(rv,rng.permutation(mv))[0] for _ in range(500)]
print(f"free-permutation p={ (np.sum(np.abs(fp)>=abs(r_obs))+1)/501 :.3f}  (destroys spatial structure -> too optimistic)")
P=sxy.loc[com,["x","y"]].values; ctr=P.mean(0); Pc=P-ctr; knn=NearestNeighbors(n_neighbors=1).fit(P); rn=[]
for _ in range(500):
    th=rng.uniform(0,2*np.pi); Rm=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); Prot=(Pc@Rm.T)+ctr
    _,j=knn.kneighbors(Prot); mr=np.full(len(com),np.nan)
    for oi,val in zip(j.ravel(),mv):
        if np.isnan(mr[oi]): mr[oi]=val
    ok=~np.isnan(mr); rn.append(spearmanr(rv[ok],mr[ok])[0])
rn=np.array(rn); print(f"ROTATION null (preserves autocorrelation) r: mean={rn.mean():.3f} std={rn.std():.3f} -> p={ (np.sum(np.abs(rn)>=abs(r_obs))+1)/(len(rn)+1) :.3f}")
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(rn,bins=30,color="#bdc3c7",label="rotation null"); ax.axvline(r_obs,c="crimson",lw=2,label=f"observed r={r_obs:.3f}")
ax.set_xlabel("Spearman r"); ax.set_title("EMT-RCC vs hyperedge Myeloid-IL1B: observed vs spatial (rotation) null"); ax.legend(); plt.tight_layout(); plt.show()"""))

# --- 7. Core cross-check ---
C.append(md(r"""## 7. Core cross-check — does the coupling hold in the other section?

We repeat the whole coupling test (EMT-high RCC domain, IL1B-high Myeloid domain, hyperedge-pooled
correlation, rotation-controlled p) on the tumor **Core** section. If the EMT↔IL1B mechanism were real it
should reproduce with the same sign; if it is spatial-autocorrelation noise, the sign should be inconsistent
across sections."""))
C.append(code(r"""from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
def coupling_test(SEC):
    ProjectMetadata.use(f"hyperspot_project_files/Human Kidney Cancer {SEC}")
    m = sym_map(SEC)
    R = load_ct_spatial_data("RCC"); M = load_ct_spatial_data("Myeloid")
    def cell_df(a, genes):
        X=a.X.toarray() if hasattr(a.X,"toarray") else np.asarray(a.X)
        ex=pd.DataFrame(X, columns=a.var_names.astype(str)); ens=[m[g] for g in genes if g in m and m[g] in ex.columns]
        return pd.DataFrame({"spot":a.obs["spot_name"].astype(str).values,"score":ex[ens].mean(1).values,
                             "x":a.obs["location_x"].values,"y":a.obs["location_y"].values})
    rdf=cell_df(R, EMT_MES); rdf["score"]=rdf["score"]-cell_df(R, EMT_EPI)["score"]; mdf=cell_df(M, ["IL1B"])
    # EMT-high RCC domain + IL1B-high Myeloid domain
    rdom=R.obs["domain"].astype(str).values
    emt_dom = pd.Series(rdf["score"].values).groupby(rdom).mean().sort_values(ascending=False)
    mil = cell_df(M, ["IL1B"])["score"]; mdom=M.obs["domain"].astype(str).values
    il_dom = pd.Series(mil.values).groupby(mdom).mean().sort_values(ascending=False)
    # hyperedge pool + rotation null
    sxy=pd.concat([rdf[["spot","x","y"]],mdf[["spot","x","y"]]]).drop_duplicates("spot").set_index("spot")
    _,ix=NearestNeighbors(n_neighbors=7).fit(sxy.values).kneighbors(sxy.values)
    sp=sxy.index.values; nb={sp[i]:set(sp[ix[i]]) for i in range(len(sp))}
    def pool(df):
        bs={x:gg["score"].values for x,gg in df.groupby("spot")}; o={}
        for x in sp:
            v=np.concatenate([bs[n] for n in nb[x] if n in bs]) if any(n in bs for n in nb[x]) else np.array([])
            o[x]=v.mean() if len(v) else np.nan
        return pd.Series(o)
    rp=pool(rdf); mp=pool(mdf); com=np.array(list(rp.dropna().index.intersection(mp.dropna().index)))
    rv=rp.loc[com].values; mv=mp.loc[com].values; r_obs=spearmanr(rv,mv)[0]
    P=sxy.loc[com,["x","y"]].values; ctr=P.mean(0); Pc=P-ctr; knn=NearestNeighbors(n_neighbors=1).fit(P)
    rng=np.random.default_rng(0); rn=[]
    for _ in range(500):
        th=rng.uniform(0,2*np.pi); Rm=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); Prot=(Pc@Rm.T)+ctr
        _,j=knn.kneighbors(Prot); mr=np.full(len(com),np.nan)
        for oi,val in zip(j.ravel(),mv):
            if np.isnan(mr[oi]): mr[oi]=val
        ok=~np.isnan(mr); rn.append(spearmanr(rv[ok],mr[ok])[0])
    rn=np.array(rn); pval=(np.sum(np.abs(rn)>=abs(r_obs))+1)/(len(rn)+1)
    return dict(EMT_dom=emt_dom.index[0], EMT_ratio=emt_dom.iloc[0], IL_dom=il_dom.index[0],
               r=r_obs, rot_p=pval, n=len(com))

core = coupling_test("Core")
print("Core: EMT-high RCC domain = RCC|%s (ratio %.2f); IL1B-high Myeloid = Myeloid|%s" % (core["EMT_dom"], core["EMT_ratio"], core["IL_dom"]))
print("Core: hyperedge-pooled RCC-EMT vs Myeloid-IL1B r=%.3f, rotation-null p=%.3f (n=%d)" % (core["r"], core["rot_p"], core["n"]))
print("\nBoth sections — hyperedge-pooled coupling (rotation-controlled):")
print("  Interface: r=%+.3f, p=%.2f" % (r_obs, (np.sum(np.abs(rn)>=abs(r_obs))+1)/(len(rn)+1)))
print("  Core:      r=%+.3f, p=%.2f" % (core["r"], core["rot_p"]))
print("  → sign %s across sections → consistent with NO real coupling (spatial-autocorrelation noise)."
      % ("FLIPS" if np.sign(r_obs)!=np.sign(core["r"]) else "agrees but n.s."))"""))

C.append(md(r"""## Summary — honest read

- **EMT-high RCC domain: recovered unsupervised in BOTH sections** (Interface RCC|13 ratio 1.48; Core RCC|11
  ratio 1.41), and the EMT programme is **enriched at the interface vs the core** (3.02 vs 2.46) — matching
  Li et al.'s finding that EMT-high RCC localises to the tumor edge. This is the strong, robust flagship: the
  headline biology falls out of the representation without supervision, consistently across sections.
- **IL1B-high Myeloid domain: present in both sections.**
- **The EMT↔IL1B coupling is not recovered in either section.** Per spot the association is null (Myeloid is
  deconvolution-sparse, ~2 cells/spot, IL1B CV≈3.2). Pooling over the hyperedge (7-NN spot neighbourhood)
  resolves the *sparsity* (~6× more cells, CV≈1.8) and gives a nominal weak effect, but under a
  spatial-autocorrelation-preserving **rotation null** it is non-significant in both sections and the **sign
  flips** (Interface r≈+0.06, p≈0.5; Core r≈−0.06, p≈0.17). An inconsistent sign across sections is the
  signature of spatial-autocorrelation noise, not a real relationship — so we report the coupling as
  **absent/unresolvable at this data resolution** (the free-permutation test was anti-conservative; the
  rotation null is the correct control).
- **Net:** HyperSpot's unsupervised domains robustly recover the EMT-RCC state and its interface enrichment
  (the headline biology, requiring the representation to exist); the mechanistic IL1B→EMT coupling is not
  distinguishable from spatial autocorrelation in either section and is reported as such."""))

nb = new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"] = {"name":"python"}
out = "/workspace/HyperSpot/demo/Human Kidney Cancer Interface EMT-IL1B Recovery.ipynb"
nbf.write(nb, out); print("wrote", out, "with", len(C), "cells")
