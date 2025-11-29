import json, glob, os, random, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from itertools import combinations
sns.set_style("whitegrid"); plt.rcParams['figure.figsize']=(16,10); plt.rcParams['font.size']=10

FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0
PALETTE={0:'#E74C3C',1:'#3498DB',4:'#9B59B6',9:'#27AE60'}

def parse_hof_file(fn):
    with open(fn) as fh:
        lines=[ln.rstrip("\n") for ln in fh.readlines()]
    geno=None
    for i,ln in enumerate(lines):
        if ln.strip().startswith("genotype:"):
            if ln.strip().endswith("~"):
                start=i+1; end=None
                for j in range(start,len(lines)):
                    if lines[j].strip()=="~":
                        end=j; break
                if end is None: end=len(lines)
                block="\n".join(lines[start:end]).strip()
                if block: geno=block; break
            else:
                parts=ln.split(":",1)
                if len(parts)==2:
                    g=parts[1].strip()
                    if g: geno=g; break
    return geno

def load_parents(fmt):
    parents=[]
    pattern=f"genotypes_landscape/HoF-land-f{fmt}-run*.gen"
    for fn in sorted(glob.glob(pattern)):
        g=parse_hof_file(fn)
        if g: parents.append(g)
    return parents

def make_lib(path, sim):
    from FramsticksLib import FramsticksLib
    return FramsticksLib(path, None, sim)

def eval_fit(frams_lib, genos, opt="vertpos"):
    data=frams_lib.evaluate(genos)
    out=[]
    for d in data:
        try:
            val=d["evaluations"][""][opt]
        except Exception:
            val=FITNESS_VALUE_INFEASIBLE_SOLUTION
        out.append(val)
    return out

def crossover_one(frams_lib, g1, g2):
    # FramsticksLib.crossOver przyjmuje dwa genotypy i zwraca potomka
    try:
        child = frams_lib.crossOver(g1, g2)
        return child
    except Exception:
        return None

def sample_pairs(genos, max_pairs=300):
    # losuj bez powtórzeń, kolejność rodziców nie ma znaczenia
    all_pairs=list(combinations(range(len(genos)),2))
    random.shuffle(all_pairs)
    return [(genos[i], genos[j]) for i,j in all_pairs[:max_pairs]]

def plot_scatter(fmt, parent_mean, child_fit, outdir="crossover_plots"):
    os.makedirs(outdir, exist_ok=True)
    color=PALETTE.get(fmt,'#555')
    plt.figure(figsize=(12,9))
    plt.scatter(parent_mean, child_fit, s=20, alpha=0.5, color=color)
    lim=max(max(parent_mean+[0]), max(child_fit+[0], default=0))
    lim=max(lim, 0.1)
    plt.plot([0,lim],[0,lim],'k--',alpha=0.6,label='y=x')
    plt.xlabel("Mean(parent fitness)"); plt.ylabel("Child fitness")
    plt.title(f"Crossover scatter f{fmt}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"crossover_scatter_f{fmt}.png"), dpi=300)
    plt.close()

def plot_heatmap(fmt, parent_mean, child_fit, outdir="crossover_plots", bins_x=20, bins_y=20):
    os.makedirs(outdir, exist_ok=True)
    # delta względem średniej rodziców
    delta=np.array(child_fit) - np.array(parent_mean)
    x=np.array(parent_mean)
    # zakresy
    x_min, x_max = float(np.min(x)), float(np.max(x))
    d_min, d_max = float(np.min(delta)), float(np.max(delta))
    # siatka
    x_bins=np.linspace(x_min, x_max, bins_x+1)
    d_bins=np.linspace(d_min, d_max, bins_y+1)
    # macierz częstości ulepszeń
    grid=np.zeros((bins_y, bins_x), dtype=float)
    counts=np.zeros((bins_y, bins_x), dtype=int)
    for xi, di in zip(x, delta):
        bx = np.searchsorted(x_bins, xi, side='right')-1
        by = np.searchsorted(d_bins, di, side='right')-1
        if 0<=bx<bins_x and 0<=by<bins_y:
            counts[by,bx]+=1
            if di>0: grid[by,bx]+=1
    # współczynniki ulepszeń
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(counts>0, grid/counts, np.nan)
    plt.figure(figsize=(12,9))
    sns.heatmap(ratio, cmap="viridis", vmin=0, vmax=1, cbar_kws={'label':'P(child > mean(parents))'})
    plt.title(f"Crossover improvement probability heatmap f{fmt}")
    plt.xlabel("Mean(parent fitness) bins"); plt.ylabel("Delta (child - mean) bins")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"crossover_heatmap_f{fmt}.png"), dpi=300)
    plt.close()

def crossover_gain_index(parent1_fit, parent2_fit, child_fit):
    # CGI = E[max(0, child - max(parent1, parent2))]
    parents_max=np.maximum(parent1_fit, parent2_fit)
    gains=np.maximum(0.0, child_fit - parents_max)
    return float(np.mean(gains)) if len(gains)>0 else 0.0

def analyze_fmt(fmt, frams_lib, parents):
    # oceń rodziców
    parent_fit = eval_fit(frams_lib, parents, opt="vertpos")
    # sparuj
    pairs = sample_pairs(parents, max_pairs=400)
    parent1=[]; parent2=[]; child=[]
    for g1,g2 in pairs:
        c = crossover_one(frams_lib, g1, g2)
        if not c: continue
        cf = eval_fit(frams_lib, [c], opt="vertpos")[0]
        if cf == FITNESS_VALUE_INFEASIBLE_SOLUTION: continue
        # fitnessy rodziców
        f1 = parent_fit[parents.index(g1)]
        f2 = parent_fit[parents.index(g2)]
        if f1 == FITNESS_VALUE_INFEASIBLE_SOLUTION or f2 == FITNESS_VALUE_INFEASIBLE_SOLUTION:
            continue
        parent1.append(f1); parent2.append(f2); child.append(cf)
    if not child:
        print(f"f{fmt}: brak poprawnych potomków"); return
    # wizualizacje
    parent_mean = list((np.array(parent1)+np.array(parent2))/2.0)
    plot_scatter(fmt, parent_mean, child)
    plot_heatmap(fmt, parent_mean, child)
    # statystyki
    child_arr=np.array(child); p1=np.array(parent1); p2=np.array(parent2)
    cgi = crossover_gain_index(p1, p2, child_arr)
    better_than_best = float(np.mean(child_arr > np.maximum(p1,p2))) * 100.0
    better_than_mean = float(np.mean(child_arr > (p1+p2)/2.0)) * 100.0
    print(f"f{fmt}: pairs={len(child_arr)}, CGI={cgi:.4f}, P(child>best)={better_than_best:.1f}%, P(child>mean)={better_than_mean:.1f}%")

def main():
    path=os.environ.get("DIR_WITH_FRAMS_LIBRARY")
    assert path, "Ustaw DIR_WITH_FRAMS_LIBRARY"
    sim="eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim"
    frams_lib = make_lib(path, sim)
    fmts=[0,1,4,9]
    for fmt in fmts:
        parents = load_parents(fmt)
        if len(parents) < 2:
            print(f"f{fmt}: za mało rodziców ({len(parents)})"); continue
        analyze_fmt(fmt, frams_lib, parents)
    print("✓ Crossover analysis done.")

if __name__=="__main__":
    main()