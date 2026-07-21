import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Ellipse, FancyArrowPatch
from matplotlib import rcParams
rcParams["font.family"]="DejaVu Sans"; rcParams["mathtext.fontset"]="dejavusans"

BG="#f5f7f6"; PANEL="#ffffff"; PANEL2="#eef2f1"; INK="#14201e"; SOFT="#46554f"
FAINT="#8a9793"; RULE="#cdd8d4"; ACC="#a86a10"; ACCT="#f2e3c6"
tA="#2c8a7e"; tB="#33739f"; tC="#b0577f"          # 3 cell types
T=[tA,tB,tC]; TN=["A","B","C"]

fig=plt.figure(figsize=(16,9.3),dpi=200)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,160); ax.set_ylim(0,93); ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

def arrow(p0,p1,c,lw=2.2,rad=0.0,style="-|>",alpha=1,ms=16):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle=style,mutation_scale=ms,color=c,lw=lw,
        connectionstyle=f"arc3,rad={rad}",alpha=alpha,zorder=7,
        shrinkA=2,shrinkB=2))
def plabel(x,txt,c=SOFT):
    ax.text(x,80,txt,ha="center",va="center",fontsize=11.5,color=c,fontweight="bold")

# header
ax.text(4,89.4,"HGCTSVG · CTHGCAEv4",fontsize=10,color=FAINT,fontweight="bold",family="monospace")
ax.text(4,85.3,"Type-aware hyperedge encoder — how one cell's neighborhood becomes its embedding",
        fontsize=16.5,color=INK,fontweight="bold")

# ────────────────────────────────────────────────────────── PANEL 1 : hypergraph
plabel(23,"1 · Hypergraph")
# hyperedge blob e
ax.add_patch(Ellipse((22,52),34,44,fc=tA,ec="none",alpha=0.06,zorder=1))
ax.add_patch(Ellipse((22,52),34,44,fc="none",ec=FAINT,lw=1.4,ls=(0,(4,3)),zorder=2))
ax.text(22,72.5,"hyperedge $e$  (spot neighborhood)",ha="center",fontsize=9.5,color=SOFT,style="italic")
# faint neighbor hyperedge
ax.add_patch(Ellipse((30,36),22,26,fc="none",ec=RULE,lw=1.1,ls=(0,(3,3)),zorder=1))
# cells: 4 A, 2 B, 1 C  (abundance imbalance)
cells=[(14,60,0),(18,66,0),(27,62,0),(15,50,0),   # A x4
       (13,44,1),(29,52,1),                        # B x2
       (21,40,2)]                                   # C x1
cellpos={0:[],1:[],2:[]}
for (x,y,t) in cells:
    ax.add_patch(Circle((x,y),2.3,fc=T[t],ec="white",lw=1.2,zorder=5))
    cellpos[t].append((x,y))
# count badges
ax.text(46,64,"type A ×4",fontsize=9,color=tA,fontweight="bold",va="center")
ax.text(46,52,"type B ×2",fontsize=9,color=tB,fontweight="bold",va="center")
ax.text(46,40,"type C ×1",fontsize=9,color=tC,fontweight="bold",va="center")
ax.text(46,34,"(rare)",fontsize=8,color=FAINT,style="italic",va="center")

# ────────────────────────────────────────────────── node→edge messages label
ax.text(59,74,r"node$\to$edge",ha="center",fontsize=9,color=FAINT,fontweight="bold")
ax.text(59,71,r"$d_v^{-1/2}\,W_{ct}$",ha="center",fontsize=10,color=FAINT)

# ────────────────────────────────────────────────────────── PANEL 2 : per-type channels
plabel(78,"2 · Type-resolved pooling")
plane_x0,plane_w=68,26
py=[62,50,38]; div=["÷ 4","÷ 2","÷ 1"]
for j,t in enumerate([0,1,2]):
    ax.add_patch(FancyBboxPatch((plane_x0,py[j]-4),plane_w,8,
        boxstyle="round,pad=0.2,rounding_size=1.4",fc=PANEL,ec=T[t],lw=2,zorder=4))
    ax.add_patch(Rectangle((plane_x0,py[j]-4),0.9,8,fc=T[t],ec="none",zorder=5))
    ax.text(plane_x0+3,py[j],f"channel {TN[t]}",fontsize=10,color=T[t],fontweight="bold",va="center")
    ax.text(plane_x0+plane_w-2,py[j],div[j],fontsize=10.5,color=ACC,fontweight="bold",va="center",ha="right")
    # arrows from that type's cells to the channel plane
    xs=[p[0] for p in cellpos[t]]; ys=[p[1] for p in cellpos[t]]
    cx,cy=np.mean(xs),np.mean(ys)
    arrow((34,cy),(plane_x0-0.5,py[j]),T[t],lw=2.6,rad=0.12 if j!=1 else 0.0)
ax.text(81,30,"each type pooled\nseparately → rare C survives",fontsize=8.6,color=SOFT,ha="center",style="italic")
ax.text(78,68.5,r"$E[e]\in\mathbb{R}^{K\times D}$",ha="center",fontsize=10.5,color=INK)

# ─────────────────────────── shared transform
sx=98
ax.add_patch(FancyBboxPatch((sx,34),8,32,boxstyle="round,pad=0.3,rounding_size=1.6",
    fc=PANEL2,ec=RULE,lw=1.2,zorder=3))
ax.text(sx+4,50,"$W_{node}$\n$\\circ\\,\\phi$",fontsize=10,color=SOFT,ha="center",va="center",fontweight="bold")
ax.text(sx+4,68.5,"shared",fontsize=8.5,color=FAINT,ha="center",fontweight="bold")
for j,t in enumerate([0,1,2]):
    arrow((plane_x0+plane_w,py[j]),(sx,50+(1-j)*0.0 if False else py[j]),T[t],lw=2,rad=0.0,ms=12)
# simplify: straight arrows into box at their y then converge; use direct
# (redo cleanly)
# edge→node label
ax.text(112,74,r"edge$\to$node",ha="center",fontsize=9,color=FAINT,fontweight="bold")
ax.text(112,71,r"$d_v^{-1/2}$",ha="center",fontsize=10,color=FAINT)

# ────────────────────────────────────────────────────────── PANEL 3 : reader cell + gate
plabel(132,"3 · Gate → embedding")
# reader cell (type A) receiving 3 channels back
rcx,rcy=112,50
ax.add_patch(Circle((rcx,rcy),3.0,fc=tA,ec="white",lw=1.4,zorder=6))
ax.text(rcx,rcy+5.4,"reader cell\n(type A)",fontsize=8.4,color=SOFT,ha="center",va="bottom")
for j,t in enumerate([0,1,2]):
    arrow((sx+8,py[j]),(rcx-2.4,rcy+ (1-j)*2.2),T[t],lw=2,rad=0.0,ms=12)

# 3x3 gate matrix
gx0,gy0,cs=120,42,5.0
ax.text(gx0+1.5*cs,gy0+3*cs+2.2,"K×K gate  $g_{a,k}$",fontsize=9.5,color=INK,ha="center",fontweight="bold")
for r in range(3):
    for c in range(3):
        d=(r==c)
        ax.add_patch(Rectangle((gx0+c*cs,gy0+(2-r)*cs),cs*0.9,cs*0.9,
            fc=ACCT if d else PANEL,ec=ACC if d else RULE,lw=1.4 if d else 0.9,zorder=5))
        ax.text(gx0+c*cs+cs*0.45,gy0+(2-r)*cs+cs*0.45,"1" if d else "0",
            ha="center",va="center",fontsize=8.5,color=ACC if d else FAINT,
            fontweight="bold" if d else "normal",zorder=6)
# row color ticks (reader a) and col ticks (source k)
for i,t in enumerate([0,1,2]):
    ax.add_patch(Rectangle((gx0-1.4,gy0+(2-i)*cs+cs*0.15),1.0,cs*0.6,fc=T[t],ec="none",zorder=6))
    ax.add_patch(Rectangle((gx0+i*cs+cs*0.15,gy0+3*cs+0.2,),cs*0.6,1.0,fc=T[t],ec="none",zorder=6))
# arrows: reader channels → matrix
arrow((rcx+3,rcy),(gx0-1.6,gy0+1.5*cs),SOFT,lw=1.8,ms=13)

# output embedding z_i
zx0=140
grad=plt.cm.get_cmap("cividis")(np.linspace(0.15,0.9,10))
for i in range(10):
    ax.add_patch(Rectangle((zx0, gy0+i*1.4), 6, 1.25, fc=grad[i], ec="none", zorder=5))
ax.add_patch(Rectangle((zx0,gy0),6,10*1.4,fc="none",ec=INK,lw=1.2,zorder=6))
ax.text(zx0+3,gy0+10*1.4+2.2,r"$z_i\in\mathbb{R}^{D}$",fontsize=10.5,color=INK,ha="center",fontweight="bold")
arrow((gx0+3*cs,gy0+1.5*cs),(zx0-0.6,gy0+7),ACC,lw=2.4,ms=15)

ax.text(132,30,"diagonal = own type (identity init)\noff-diagonal = cross-type, learned under L1",
        fontsize=8.6,color=SOFT,ha="center",style="italic")

# big stage separators (soft chevrons)
for xc in [56,94,116]:
    ax.text(xc,50,"›",fontsize=26,color=RULE,ha="center",va="center",zorder=0)

ax.text(4,5.5,"Cells in a neighborhood are pooled per type (each ÷ its own count) so abundant types can't swamp rare ones;  "
        "a learned K×K gate then lets each reader type weight every source type before forming its embedding.",
        fontsize=9.3,color=SOFT)

out="/tmp/claude-0/-workspace-ctSVG-benchmark-sim/3934a20e-d765-4f4c-b4f2-e41c22b07e3b/scratchpad/encoder_v4_schematic.png"
fig.savefig(out,facecolor=BG,bbox_inches="tight",pad_inches=0.2)
print("saved",out)
