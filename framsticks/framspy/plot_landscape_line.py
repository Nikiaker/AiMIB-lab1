import os, glob, numpy as np, matplotlib.pyplot as plt, seaborn as sns
sns.set_style("whitegrid"); plt.rcParams['figure.figsize']=(14,9); plt.rcParams['font.size']=10

FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0
PALETTE_GROUPS=["#2ECC71","#3498DB","#E67E22","#E74C3C"]  # top, upper-mid, lower-mid, worst

def parse_hof_file(fn):
    with open(fn) as fh:
        lines=[ln.rstrip("\n") for ln in fh.readlines()]
    geno=None
    for i,ln in enumerate(lines):
        if ln.strip().startswith("genotype:"):
            if ln.strip().endswith("~"):  # f0 blok
                start=i+1; end=None
                for j in range(start,len(lines)):
                    if lines[j].strip()=="~":
                        end=j; break
                if end is None: end=len(lines)
                block="\n".join(lines[start:end]).strip()
                geno=block if block else None
            else:
                parts=ln.split(":",1)
                geno=parts[1].strip() if len(parts)==2 else None
            break
    return geno

def load_parents(fmt):
    parents=[]
    for fn in sorted(glob.glob(f"genotypes_landscape/HoF-land-f{fmt}-run*.gen")):
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

def mutate_until_feasible(frams_lib, geno, max_tries=100):
    tries=0
    while True:
        tries+=1
        child=frams_lib.mutate([geno])[0]
        fit=eval_fit(frams_lib,[child])[0]
        if fit!=FITNESS_VALUE_INFEASIBLE_SOLUTION or tries>=max_tries:
            return child, fit, tries

def pick_20_balanced(genos, fits):
    # sortuj po fitness rosnąco i wybierz 4×5: worst, lower-mid, upper-mid, best
    idx=np.argsort(fits)
    if len(idx)<20:
        idx=list(idx)
    else:
        q=len(idx)//4
        worst=idx[:5]
        lowmid=idx[q:q+5]
        upmid=idx[2*q:2*q+5]
        best=idx[-5:]
        idx=list(worst)+list(lowmid)+list(upmid)+list(best)
    return idx

def random_walk_for_fmt(fmt, steps=30):
    path=os.environ.get("DIR_WITH_FRAMS_LIBRARY"); assert path, "Ustaw DIR_WITH_FRAMS_LIBRARY"
    sim="eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim"
    lib=make_lib(path, sim)

    parents=load_parents(fmt)
    if len(parents)<5:
        print(f"f{fmt}: za mało rodziców"); return
    fits=eval_fit(lib, parents)

    idx=pick_20_balanced(parents, fits)
    seqs=[]; groups=[]
    improve_steps=0; total_steps=0; tries_all=[]

    for k,i in enumerate(idx):
        geno=parents[i]; f0=fits[i]
        seq=[f0]; attempts=[]
        curr_geno=geno; curr_fit=f0
        for s in range(steps):
            child, fch, tries = mutate_until_feasible(lib, curr_geno)
            attempts.append(tries)
            if fch!=FITNESS_VALUE_INFEASIBLE_SOLUTION:
                if fch>curr_fit: improve_steps+=1
                total_steps+=1
                curr_geno, curr_fit = child, fch
                seq.append(curr_fit)
        seqs.append(seq)
        tries_all.extend(attempts)
        # grupy: 4 kolory po 5 linii
        groups.append(min(k//5,3))

    # wykres
    plt.figure(figsize=(16,10))
    for seq,grp in zip(seqs,groups):
        plt.plot(range(len(seq)), seq, color=PALETTE_GROUPS[3-grp], alpha=0.8, linewidth=1.5)
    plt.xlabel("Mutation step"); plt.ylabel("vertpos"); plt.title(f"Random walk (30 mutations) f{fmt}")
    from matplotlib.lines import Line2D
    legend_elems=[
        Line2D([0],[0],color=PALETTE_GROUPS[0],label="Top 5"),
        Line2D([0],[0],color=PALETTE_GROUPS[1],label="Upper-mid 5"),
        Line2D([0],[0],color=PALETTE_GROUPS[2],label="Lower-mid 5"),
        Line2D([0],[0],color=PALETTE_GROUPS[3],label="Worst 5"),
    ]
    plt.legend(handles=legend_elems, loc="best")
    os.makedirs("randomwalk_plots", exist_ok=True)
    plt.tight_layout(); plt.savefig(f"randomwalk_plots/randomwalk_f{fmt}.png", dpi=300); plt.close()

    imp_ratio = (improve_steps/total_steps*100.0) if total_steps>0 else 0.0
    print(f"f{fmt}: steps={total_steps}, improved_steps={imp_ratio:.1f}% , mean tries/step={np.mean(tries_all):.2f}")

def main():
    for fmt in [0,1,4,9]:
        random_walk_for_fmt(fmt, steps=30)
    print("✓ Random walks done.")

if __name__=="__main__":
    main()