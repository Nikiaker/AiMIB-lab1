# Zbiera próbki z logów, generuje mutantów, rysuje wykresy i raportuje odsetek lepszych mutantów.
import json, glob, os, random, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
sns.set_style("whitegrid"); plt.rcParams['figure.figsize']=(16,10); plt.rcParams['font.size']=10

DATA_DIR="logs_landscape"
SAMPLES_OUT="samples_landscape/samples.json"
NEIGHBORS_PER_SAMPLE=20

# Wczytanie przebiegów
def load_runs(fmt):
    pattern=f"{DATA_DIR}/log-land-f{fmt}-run*.json"
    runs=[]
    for fn in sorted(glob.glob(pattern)):
        try:
            with open(fn) as f: runs.append(json.load(f))
        except: pass
    return runs

# Ekstrakcja próbek (historia elity bez duplikatów fitness)
def collect_samples(fmt):
    runs=load_runs(fmt)
    samples=[]
    for r in runs:
        gens=r.get("generations",[])
        prev=None
        for g in gens:
            bf=g.get("best_fitness")
            if bf is None or bf==float("-inf"): continue
            # zapisuj gdy istotna zmiana fitness
            if prev is None or abs(bf - prev) > 1e-6:
                samples.append({"fmt":fmt, "fitness":float(bf)})
                prev=bf
    return samples

# Mutacja i ocena przez FramsticksLib
def eval_genotypes(frams_lib, genos, opt="vertpos"):
    data=frams_lib.evaluate(genos)
    out=[]
    for d in data:
        try:
            val=d["evaluations"][""][opt]
        except Exception:
            val=float("-999999.0")
        out.append(val)
    return out

# Losowa mutacja pojedynczego genotypu (używa FramsticksLib.mutate)
def mutate_many(frams_lib, geno, n):
    muts=[]
    for _ in range(n):
        muts.append(frams_lib.mutate([geno])[0])
    return muts

def make_toolbox(path, sim):
    from FramsticksLib import FramsticksLib
    return FramsticksLib(path, None, sim)

def plot_aggregated(fmt_list):
    palette={0:'#E74C3C',1:'#3498DB',4:'#9B59B6',9:'#27AE60'}
    for fmt in fmt_list:
        runs=load_runs(fmt)
        mats=[]; maxlen=0
        for r in runs:
            v=[g.get("best_fitness") for g in r.get("generations",[]) if g.get("best_fitness") is not None]
            if not v: continue
            maxlen=max(maxlen,len(v)); mats.append(v)
        mats=[ v + [v[-1]]*(maxlen-len(v)) for v in mats if v ]
        if not mats: continue
        arr=np.array(mats,dtype=float)
        mean=np.nanmean(arr,axis=0); std=np.nanstd(arr,axis=0); gens=np.arange(maxlen)
        plt.plot(gens,mean,color=palette[fmt],label=f"f{fmt}")
        plt.fill_between(gens, mean-0.5*std, mean+0.5*std, color=palette[fmt], alpha=0.2)
    plt.xlabel("Generation"); plt.ylabel("Best vertpos"); plt.title("Mean ± 0.5·StdDev vertpos (only-body)")
    plt.legend(); plt.tight_layout(); plt.savefig("land_vertpos_mean.png",dpi=300); plt.close()

def plot_box(fmt_list):
    palette={0:'#E74C3C',1:'#3498DB',4:'#9B59B6',9:'#27AE60'}
    vals=[]; labels=[]; colors=[]
    for fmt in fmt_list:
        runs=load_runs(fmt); final=[]
        for r in runs:
            bi=r.get("best_individual",{})
            if "fitness" in bi and bi["fitness"] not in (None, float("-inf")):
                final.append(bi["fitness"])
        if final:
            vals.append(final); labels.append(f"f{fmt}"); colors.append(palette[fmt])
    bp=plt.boxplot(vals,labels=labels,patch_artist=True, widths=0.6)
    for b,c in zip(bp['boxes'],colors): b.set_facecolor(c); b.set_alpha(0.65)
    plt.ylabel("Final best vertpos"); plt.title("Final vertpos distribution"); plt.grid(axis='y',alpha=0.3)
    plt.tight_layout(); plt.savefig("land_vertpos_box.png",dpi=300); plt.close()

def scatter_neighbors(fmt_list, path, sim):
    from FramsticksLib import FramsticksLib
    frams_lib=FramsticksLib(path, None, sim)

    def parse_hof_file(fn):
        """
        Parsuje plik HoF-*.gen i zwraca surowy genotyp jako string.
        Obsługa:
        - f0: blok między 'genotype:' i kończącym '~' (wielolinijkowy)
        - f1/f4/f9: linia z 'genotype:' po dwukropku
        """
        with open(fn) as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]
        geno = None

        # Spróbuj trybu blokowego (f0)
        for i, ln in enumerate(lines):
            if ln.strip().startswith("genotype:"):
                # jeżeli następna linia to '~', to cały blok zaczyna się tu
                # w f0 jest zwykle "genotype:~" i potem treść aż do '~' w osobnej linii
                # wykryj czy po 'genotype:' jest już '~'
                if ln.strip().endswith("~"):
                    start = i + 1
                    # znajdź kończące '~'
                    end = None
                    for j in range(start, len(lines)):
                        if lines[j].strip() == "~":
                            end = j
                            break
                    if end is None:
                        # brak końca bloku – awaryjnie weź resztę
                        end = len(lines)
                    block = "\n".join(lines[start:end]).strip()
                    if block:
                        geno = block
                        break
                else:
                    # Jednolinijkowe genotypy (inne formaty)
                    parts = ln.split(":", 1)
                    if len(parts) == 2:
                        g = parts[1].strip()
                        if g:
                            geno = g
                            break
        return geno

    # Zbierz elity jako próbki genotypów
    samples=[]
    for fmt in fmt_list:
        pattern=f"genotypes_landscape/HoF-land-f{fmt}-run*.gen"
        for fn in sorted(glob.glob(pattern)):
            geno = parse_hof_file(fn)
            if geno:
                samples.append({"fmt":fmt,"genotype":geno})

    if not samples:
        print("Brak genotypów w genotypes_landscape/. Upewnij się, że HoF zapisuje genotyp."); return

    # Oceń rodziców
    parents=[s["genotype"] for s in samples]
    parents_fit=eval_genotypes(frams_lib, parents, opt="vertpos")

    # Mutanci
    result_by_fmt={fmt:[] for fmt in fmt_list}
    better_pct={fmt:0 for fmt in fmt_list}; count_by_fmt={fmt:0 for fmt in fmt_list}
    for s,(pfit) in zip(samples,parents_fit):
        fmt=s["fmt"]
        muts=mutate_many(frams_lib, s["genotype"], NEIGHBORS_PER_SAMPLE)
        mfit=eval_genotypes(frams_lib, muts, opt="vertpos")
        good=[mf for mf in mfit if mf != -999999.0]
        for mf in good:
            result_by_fmt[fmt].append((pfit,mf))
            count_by_fmt[fmt]+=1
            if mf>pfit: better_pct[fmt]+=1

    # Wykresy
    for fmt in fmt_list:
        pts=result_by_fmt[fmt]
        if not pts: continue
        x=[a for a,b in pts]; y=[b for a,b in pts]
        plt.figure(figsize=(10,8))
        plt.scatter(x,y,s=18,alpha=0.5)
        mx=max(x+y) if x and y else 1.0
        lim=max(mx, 0.05)
        plt.plot([0,lim],[0,lim],'k--',alpha=0.6,label='y=x')
        pct=(better_pct[fmt]/count_by_fmt[fmt]*100.0) if count_by_fmt[fmt]>0 else 0.0
        plt.title(f"Neighbors scatter f{fmt} (better: {pct:.1f}%)")
        plt.xlabel("Parent vertpos"); plt.ylabel("Neighbor vertpos")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"land_neighbors_f{fmt}.png",dpi=300); plt.close()

    print("Better neighbors (%):", {fmt: (better_pct[fmt]/count_by_fmt[fmt]*100.0 if count_by_fmt[fmt]>0 else 0.0) for fmt in fmt_list})


def main():
    fmts=[0,1,4,9]
    # klasyczne wykresy z bazowych przebiegów
    plot_aggregated(fmts)
    plot_box(fmts)
    # scatter sąsiedzi
    path=os.environ.get("DIR_WITH_FRAMS_LIBRARY")
    assert path, "Ustaw DIR_WITH_FRAMS_LIBRARY"
    sim="eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim"
    scatter_neighbors(fmts, path, sim)
    print("✓ Landscape analysis done.")

if __name__ == "__main__":
    main()