import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"

# palette
BG="#f5f7f6"; PANEL="#ffffff"; PANEL2="#eef2f1"; INK="#14201e"; SOFT="#45544f"
FAINT="#7a8884"; RULE="#d3ddda"; ACC="#a86a10"; ACCT="#f2e7d2"; SHARED="#48605a"
CH=["#2c8a7e","#33739f","#6a6ab0","#b0577f"]

fig=plt.figure(figsize=(9.2,15.6),dpi=200)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,176); ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

def card(y_top,h,key=False):
    x0,x1=8,96
    ax.add_patch(FancyBboxPatch((x0,y_top-h),x1-x0,h,
        boxstyle="round,pad=0.6,rounding_size=2.2",
        fc=PANEL,ec=ACC if key else RULE,lw=1.6 if key else 1.1,zorder=2))
    return x0,x1

def badge(y,txt):
    ax.add_patch(Circle((5.2,y),2.4,fc=PANEL,ec=FAINT,lw=1.4,zorder=3))
    ax.text(5.2,y,txt,ha="center",va="center",fontsize=9,color=SOFT,zorder=4,fontweight="bold")

def title(x,y,txt,tag=None,tagc=ACC):
    ax.text(x,y,txt,ha="left",va="center",fontsize=13.5,color=INK,fontweight="bold",zorder=5)
    if tag:
        tx=x+len(txt)*0.72+3
        ax.text(96-2,y,tag,ha="right",va="center",fontsize=8.2,color=tagc,zorder=5,
                fontweight="bold",bbox=dict(boxstyle="round,pad=0.3",fc=ACCT if tagc==ACC else "#e4eae8",ec="none"))

def eq(x,y,s,size=15):
    ax.text(x,y,s,ha="left",va="center",fontsize=size,color=INK,zorder=5)

def rail(y0,y1):
    ax.plot([5.2,5.2],[y0,y1],color="#c7d3cf",lw=2,zorder=1)

# ── header
ax.text(8,172,"HGCTSVG · model/hypergraph/cthgcaev4.py",fontsize=9.5,color=FAINT,
        fontweight="bold",family="monospace")
ax.text(8,167.5,"CTHGCAEv4 — the type-aware hyperedge encoder",fontsize=19,color=INK,fontweight="bold")
ax.text(8,163.6,"Each hyperedge is resolved into K per-type channels (each ÷ its own count),",fontsize=10.5,color=SOFT)
ax.text(8,161.2,"then recombined through a learned K×K cross-type gate.",fontsize=10.5,color=SOFT)

rail(12,156)

y=156
# INPUT
x0,x1=card(y,14); badge(y-7,"in")
title(11,y-3,"Inputs & degree normalization")
eq(13,y-9.2,r"$d_{v_i}=\sum_{e} H_{ie}\qquad$ (node degree — hyperedges cell $i$ joins)",13)
y-=18

# STAGE 1
x0,x1=card(y,14); badge(y-7,"1")
title(11,y-3,"Type-specific input message",tag="PER CELL TYPE · W")
eq(13,y-9.2,r"$\mathbf{m}_i = d_{v_i}^{-1/2}\,\mathbf{W}_{c_i}\,\mathbf{x}_i \;\in\; \mathbb{R}^{D}$")
y-=18

# STAGE 2 (key)
x0,x1=card(y,26,key=True); badge(y-7,"2")
title(11,y-3,"Type-resolved hyperedge pooling",tag="THE CORE IDEA",tagc=SHARED)
eq(13,y-9.5,r"$\mathbf{E}_{e,k} = \frac{1}{de_{e,k}}\sum_{i:\,c_i=k} H_{ie}\,\mathbf{m}_i + \mathbf{b}\;\in\;\mathbb{R}^{D}$",14.5)
eq(13,y-16.5,r"$de_{e,k}=\sum_{i:\,c_i=k} H_{ie}\qquad$ (count of type-$k$ cells in hyperedge $e$)",11.5)
# channel chips
cw=(x1-13-3*1.2)/4
for k in range(4):
    cx=13+k*(cw+1.2)
    ax.add_patch(FancyBboxPatch((cx,y-24),cw,4,boxstyle="round,pad=0.15,rounding_size=1",
        fc=PANEL2,ec=RULE,lw=0.8,zorder=3))
    ax.add_patch(Rectangle((cx,y-24),0.5,4,fc=CH[k],ec="none",zorder=4))
    ax.text(cx+cw/2,y-22,f"channel k={k}",ha="center",va="center",fontsize=7.6,color=CH[k],fontweight="bold",zorder=5)
y-=30

# STAGE 3
x0,x1=card(y,14); badge(y-7,"3")
title(11,y-3,"Shared hyperedge transform",tag="SHARED · W_node",tagc=SHARED)
eq(13,y-9.2,r"$\mathbf{U}_{e,k} = \mathbf{W}_{node}\,\phi(\mathbf{E}_{e,k}),\quad \phi=\mathrm{dropout}\circ\mathrm{PReLU}$",13)
y-=18

# STAGE 4
x0,x1=card(y,14); badge(y-7,"4")
title(11,y-3,"Scatter channels back to nodes")
eq(13,y-9.2,r"$\mathbf{c}_{i,k} = d_{v_i}^{-1/2}\sum_{e} H_{ie}\,\mathbf{U}_{e,k}\;\in\;\mathbb{R}^{D}\quad(\mathbf{c}\in\mathbb{R}^{N\times K\times D})$",13)
y-=18

# STAGE 5 (key)
x0,x1=card(y,30,key=True); badge(y-7,"5")
title(11,y-3,"K×K gated combine",tag="LEARNED · gate + per-CT bias")
eq(13,y-10,r"$\mathbf{z}_i = \sum_{k=1}^{K} g_{c_i,k}\,\mathbf{c}_{i,k} + \mathbf{b}_{c_i}\;\in\;\mathbb{R}^{D}$",15)
# gate matrix (identity)
mx0,my0=15,y-27; cs=2.6
for r in range(4):
    for c in range(4):
        diag=(r==c)
        ax.add_patch(Rectangle((mx0+c*cs,my0+(3-r)*cs),cs*0.86,cs*0.86,
            fc=ACCT if diag else PANEL2,ec=ACC if diag else RULE,lw=1.1 if diag else 0.7,zorder=4))
        ax.text(mx0+c*cs+cs*0.43,my0+(3-r)*cs+cs*0.43,"1" if diag else "0",
                ha="center",va="center",fontsize=8,color=ACC if diag else FAINT,
                fontweight="bold" if diag else "normal",zorder=5)
ax.text(mx0+4*cs+2,my0+2*cs+cs*0.4,r"$g_{a,k}$: reader $a$ weights source $k$",fontsize=10,color=SOFT,va="center")
ax.text(mx0+4*cs+2,my0+cs+cs*0.4,r"init $G\!\leftarrow\!I$ ; L1 on off-diagonal:",fontsize=10,color=SOFT,va="center")
ax.text(mx0+4*cs+2,my0+cs*0.4,r"$\mathcal{L}_{gate}=\sum_{a\neq k}|g_{a,k}|$",fontsize=12,color=ACC,va="center")
y-=34

# OUTPUT
x0,x1=card(y,17); badge(y-7,"out")
title(11,y-3,"Outputs")
eq(13,y-9,r"$\mathbf{z}_i\in\mathbb{R}^{N\times D}$ (node)$\qquad \mathbf{h}_e=\frac{1}{K}\sum_{k=1}^{K}\mathbf{E}_{e,k}\in\mathbb{R}^{D}$ (hyperedge)",12.5)
ax.text(13,y-14,"wrapper discards $h_e$ and re-pools node embeddings for the hyperedge task",
        fontsize=9,color=FAINT,style="italic")

out="/tmp/claude-0/-workspace-ctSVG-benchmark-sim/3934a20e-d765-4f4c-b4f2-e41c22b07e3b/scratchpad/encoder_v4.png"
fig.savefig(out,facecolor=BG,bbox_inches="tight",pad_inches=0.25)
print("saved",out)
