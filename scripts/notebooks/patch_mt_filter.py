import nbformat, glob, os

MT_DEF = (
    "# mitochondrial genes (13 protein-coding MT genes) to drop before the embedding\n"
    "MT_IDS = ['ENSG00000198886','ENSG00000198888','ENSG00000228253','ENSG00000198899',\n"
    "          'ENSG00000198695','ENSG00000198840','ENSG00000198938','ENSG00000198804',\n"
    "          'ENSG00000198712','ENSG00000212907','ENSG00000198763','ENSG00000198727',\n"
    "          'ENSG00000198786']\n"
    "is_mt_gene = lambda adata: adata.var_names.isin(MT_IDS)\n"
)

nbs = [
    "Human Kidney Cancer Interface 2 - HyperSpot Pipeline.ipynb",
    "Human Kidney Cancer Core 2 - HyperSpot Pipeline.ipynb",
    "Human Kidney Cancer Multi-Sample 2 - HyperSpot Pipeline.ipynb",
]
base = "/workspace/HyperSpot/demo"
for name in nbs:
    p = os.path.join(base, name)
    nb = nbformat.read(p, as_version=4)
    patched = 0
    for c in nb.cells:
        if c.cell_type != "code" or "common.filter(" not in c.source:
            continue
        src = c.source
        if "is_mt_gene" in src:
            continue
        # add the mitochondrial predicate arg to each common.filter call
        src = src.replace("min_locs=20)", "min_locs=20, is_mt_gene=is_mt_gene)")
        # insert MT_DEF just before the first common.filter line, preserving indentation
        lines = src.split("\n")
        out, inserted = [], False
        for ln in lines:
            if (not inserted) and "common.filter(" in ln:
                indent = ln[:len(ln) - len(ln.lstrip())]
                out += [indent + d for d in MT_DEF.rstrip("\n").split("\n")]
                inserted = True
            out.append(ln)
        c.source = "\n".join(out)
        patched += 1
    nbformat.write(nb, p)
    print(f"{name}: patched {patched} cell(s)")
