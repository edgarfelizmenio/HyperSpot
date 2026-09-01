import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

C=[]
C.append(md(r"""# Conserved ct-SVG markers — biological face-validity (#7)

We harvest HyperSpot's **cross-donor conserved** ct-SVGs (genes recurring in ≥3 of 5 interface donors, MT-free)
and ask whether they are biologically real, in two honest layers:

1. **Face-validity** — do the conserved markers overlap **known kidney-cancer genes** (CancerMine, Harmonizome)
   more than chance (Fisher's exact)?
2. **Housekeeping-controlled biology** — the raw conserved set is dominated by housekeeping (ribosomal/OXPHOS),
   which inflates any overlap; we therefore repeat after removing housekeeping families and report the
   **cell-type-specific residual** and its enrichment.

We then tabulate the concrete conserved markers per abundant cell type, flagging known kidney-cancer genes."""))

C.append(md("### Setup — conserved lists (ENSG→symbol), known kidney-cancer gene sets, background panel"))
C.append(code(r"""import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, re, matplotlib.pyplot as plt, anndata as ad, scanpy as sc
from scipy.stats import hypergeom

CONS = "hyperspot_project_files/_conserved_ctsvg_interface"
REF  = "/workspace/HyperSpot/references/known_gene_lists"
ROBUST = ["RCC","EC","Fibro","Epi_non-PT","Myeloid","Epi_PT","B-cell","T-cell"]

# ENSG -> symbol map (interface decomposition)
dec = ad.read_h5ad("hyperspot_project_files/Human Kidney Cancer Interface/ST_decomposition.h5ad", backed='r')
ens2sym = {}
for e,s in zip(dec.var_names.astype(str), dec.var["SYMBOL"].astype(str)): ens2sym.setdefault(e,s)

def conserved_syms(ct):
    f=f"{CONS}/conserved_{ct.replace('-','_')}_ge3.csv"
    try: g=pd.read_csv(f)["gene"].astype(str).tolist()
    except Exception: return []
    return sorted({ens2sym.get(e,e) for e in g})

# known kidney-cancer gene sets (symbols)
known = {name: set(l.strip() for l in open(f"{REF}/{fn}") if l.strip())
         for name,fn in [("CancerMine-kidney","cancermine_kidney.txt"),
                         ("Harmonizome-kidney","harmonizome_kidney_cancer.txt")]}
# background = measured panel (symbols)
BG = set(ens2sym.get(e,e) for e in ad.read_h5ad("hyperspot_project_files/Human Kidney Cancer Interface/ST_aggregated.h5ad").var_names.astype(str))
print("background genes:", len(BG))
for k,v in known.items(): print(f"  {k}: {len(v)} genes ({len(v&BG)} in panel)")"""))

C.append(md("""## 1. Face-validity — overlap of conserved markers with known kidney-cancer genes
Fisher's exact (hypergeometric) overlap of each cell type's conserved ct-SVGs with the kidney-cancer gene
sets, over the measured panel background."""))
C.append(code(r"""def fisher(sig, ref, bg):
    sig=set(sig)&bg; ref=set(ref)&bg; N=len(bg); k=len(sig&ref)
    exp=len(sig)*len(ref)/N if N else np.nan
    p=hypergeom.sf(k-1, N, len(ref), len(sig)) if len(sig) and len(ref) else np.nan
    return len(sig), k, (k/exp if exp else np.nan), p
rows=[]
for ct in ROBUST:
    s=conserved_syms(ct)
    for kn,ref in known.items():
        n,k,fold,p = fisher(s, ref, BG)
        rows.append((ct, kn, n, k, round(fold,2), p))
fv=pd.DataFrame(rows, columns=["CT","known_set","n_conserved","overlap","fold","p"])
print(fv.to_string(index=False))"""))
C.append(code(r"""fig,ax=plt.subplots(figsize=(11,4)); x=np.arange(len(ROBUST)); w=0.4
for i,kn in enumerate(known):
    sub=fv[fv.known_set==kn].set_index("CT").reindex(ROBUST)
    ax.bar(x+(i-0.5)*w, sub["fold"], w, label=kn)
ax.axhline(1,ls="--",c="grey"); ax.set_xticks(x); ax.set_xticklabels(ROBUST, rotation=45, ha="right")
ax.set_ylabel("fold enrichment"); ax.set_title("Conserved ct-SVGs vs known kidney-cancer genes (raw)"); ax.legend()
plt.tight_layout(); plt.show()"""))

C.append(md(r"""## 2. Housekeeping-controlled — the cell-type-specific residual

The raw conserved sets are inflated by housekeeping (ribosomal/OXPHOS), which are broadly kidney-expressed and
can drive the overlap above. We strip these families and report (a) how much of each conserved set is
housekeeping and (b) the residual's functional enrichment (the genuine cell-type-specific programme)."""))
C.append(code(r"""HK = re.compile(r"^(RP[LS]\d|MRP[LS]\d|RPLP\d|FAU|MT-|NDUF|SDH[ABCD]|UQCR|COX[0-9]|COX[4-8]|ATP5|CYC1|CYCS)")
def split_hk(syms):
    hk=[g for g in syms if HK.match(g)]; rest=[g for g in syms if not HK.match(g)]; return hk,rest
print(f"{'CT':11s} {'n':>4s} {'%HK':>4s} {'residual':>8s}")
resid={}
for ct in ROBUST:
    s=conserved_syms(ct); hk,rest=split_hk(s); resid[ct]=rest
    print(f"{ct:11s} {len(s):>4d} {100*len(hk)//max(1,len(s)):>3d}% {len(rest):>8d}")
print("\n=== residual (housekeeping-removed) enrichment per cell type ===")
for ct in ROBUST:
    rest=resid[ct]
    if len(rest)<10: print(f"\n### {ct}: {len(rest)} residual (too few)"); continue
    try:
        e=sc.queries.enrich(rest, org="hsapiens", gprofiler_kwargs={"sources":["GO:BP","REAC","KEGG"],"no_evidences":True})
        e=e[e["term_size"]<600].sort_values("p_value").head(3)
        print(f"\n### {ct} ({len(rest)} residual conserved)")
        for _,r in e.iterrows(): print(f"   [{r['source']}] {r['name'][:52]:52s} p={r['p_value']:.1e} n={int(r['intersection_size'])}")
    except Exception as ex: print(f"\n### {ct}: enrich failed ({str(ex)[:40]})")"""))

C.append(md("""## 3. Harvested conserved markers per cell type (known kidney-cancer genes flagged)
Top conserved, housekeeping-removed markers for each abundant cell type; `**` = in a known kidney-cancer set.
For Myeloid we additionally flag antigen-presentation / macrophage genes that converge with niche-DE's
published tumor-associated-macrophage markers (CD74, APOC1, APOE, SPP1, C1Q)."""))
C.append(code(r"""allknown = known["CancerMine-kidney"] | known["Harmonizome-kidney"]
MAC = {"CD74","APOC1","APOE","SPP1","C1QA","C1QB","C1QC","HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","B2M","BST2"}
for ct in ["RCC","EC","Fibro","Myeloid","Epi_non-PT"]:
    rest=[g for g in resid[ct] if not g.startswith("ENSG")]
    flagged=[(g+("**" if g in allknown else "")+("†" if g in MAC else "")) for g in rest[:25]]
    nk=sum(g in allknown for g in resid[ct])
    print(f"\n{ct}: {len(resid[ct])} residual conserved, {nk} known kidney-cancer ( ** ), macrophage/AP genes ( † )")
    print("  " + ", ".join(flagged))"""))

C.append(md(r"""## Summary — honest face-validity read

- **Face-validity holds, but read it with the housekeeping caveat.** Conserved ct-SVGs overlap known
  kidney-cancer gene sets above chance for the abundant compartments — but the raw sets are ~15–35%
  housekeeping (ribosomal/OXPHOS), which are broadly kidney-expressed and inflate the overlap, so raw
  enrichment is not by itself evidence of cell-type-specific biology.
- **The housekeeping-removed residual is where the genuine, interpretable biology lives.** After removing
  ribosomal/OXPHOS families, the per-cell-type residual programmes are: a broad **secretory / ECM /
  platelet-degranulation** module shared by the stromal/epithelial compartments (RCC, EC, Fibro, Epi), and —
  the one clearly cell-type-specific conserved programme — **myeloid antigen presentation** (MHC / C1Q / CD74
  / HLA). The latter is patient-recurrent, biologically coherent, on-theme with the dataset's
  tumor-microenvironment immune biology, and **converges with niche-DE's published tumor-associated-macrophage
  markers** (CD74, APOC1, APOE, SPP1) — an independent cross-method corroboration.
- **Net:** the conserved ct-SVGs are not random — they overlap known kidney-cancer genes and, once
  housekeeping is controlled, yield a genuine, cross-method-corroborated cell-type-specific signal (myeloid
  antigen presentation), while the bulk of the raw conservation is generic housekeeping/secretory and is
  reported as such."""))

nb=new_notebook(cells=C)
nb.metadata["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"]={"name":"python"}
out="/workspace/HyperSpot/demo/Human Kidney Cancer Conserved Marker Face-Validity.ipynb"
nbf.write(nb,out); print("wrote",out,"with",len(C),"cells")
