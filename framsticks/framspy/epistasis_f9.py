import os, glob, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from FramsticksLib import FramsticksLib

FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0
SIM="eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim"
PREFIX="/*9*/"  # f9

def normalize_f9(geno: str) -> str:
    """Upewnia się, że genotyp ma poprawny prefix '/*9*/'."""
    g = geno.strip()
    if g.startswith("/*"):
        # popraw niepoprawne '/**/' lub inne
        if g.startswith("/**/"):
            g = "/*9*/" + g[len("/**/"):]
        elif not g.startswith("/*9*/"):
            # zamień dowolny inny prefix na f9
            end = g.find("*/")
            g = "/*9*/" + g[end+2:] if end != -1 else "/*9*/" + g
        return g
    else:
        return "/*9*/" + g

def parse_hof_file(fn):
    # proste wyciąganie genotypu z linii "genotype:"
    with open(fn) as fh:
        for ln in fh:
            ln=ln.strip()
            if ln.startswith("genotype:"):
                raw = ln.split(":",1)[1].strip()
                return normalize_f9(raw)
    return None

def load_candidates():
    gens=[]
    for fn in sorted(glob.glob("genotypes_landscape/HoF-land-f9-run*.gen")):
        g=parse_hof_file(fn)
        if g: gens.append(g)
    # deduplikacja wg długości/treści
    uniq=[]
    seen=set()
    for g in gens:
        k=(len(g), g[:20])
        if k not in seen:
            uniq.append(g); seen.add(k)
    return uniq

def make_lib():
    path=os.environ.get("DIR_WITH_FRAMS_LIBRARY")
    assert path, "Ustaw DIR_WITH_FRAMS_LIBRARY"
    return FramsticksLib(path, None, SIM)

def eval_fit(frams, geno):
    g = normalize_f9(geno)
    try:
        data=frams.evaluate([g])[0]
        return data['evaluations']['']['vertpos']
    except Exception:
        return FITNESS_VALUE_INFEASIBLE_SOLUTION

def disable_chars(s, idxs):
    # usuń znaki na pozycjach idxs (string bez tych znaków)
    arr=list(s)
    for i in idxs:
        arr[i]=''
    return ''.join(arr)

def epistasis_matrix(frams, g):
    g = normalize_f9(g)
    n=len(g)
    fit0=eval_fit(frams, g)
    if fit0==FITNESS_VALUE_INFEASIBLE_SOLUTION:
        return None, None, None
    # sprawdź pojedyncze wyłączenia
    single_ok=np.ones(n, dtype=bool)
    single_effect=np.full(n, np.nan)
    for i in range(n):
        g1=disable_chars(g, [i])
        f1=eval_fit(frams, g1)
        if f1==FITNESS_VALUE_INFEASIBLE_SOLUTION:
            single_ok[i]=False
        else:
            single_effect[i]=f1 - fit0
    valid_idx=np.where(single_ok)[0]
    if len(valid_idx)==0:
        return None, None, None
    m=len(valid_idx)
    epi=np.full((m,m), np.nan)
    # epistaza e_ij = f(g - {i,j}) - f(g) - (Δi + Δj)
    for a,ii in enumerate(valid_idx):
        for b,jj in enumerate(valid_idx):
            g2=disable_chars(g, [ii, jj])
            f2=eval_fit(frams, g2)
            if f2!=FITNESS_VALUE_INFEASIBLE_SOLUTION:
                epi[a,b]=f2 - fit0 - (single_effect[ii] + single_effect[jj])
    return epi, valid_idx, fit0

def plot_heatmap(epi, title, outpath):
    plt.figure(figsize=(6,5))
    im=plt.imshow(epi, cmap="RdYlGn", norm=CenteredNorm())
    plt.colorbar(im,label="Epistasis")
    plt.title(title); plt.xlabel("gene j (valid idx)"); plt.ylabel("gene i (valid idx)")
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def summarize(epi):
    # prosty wskaźnik: odsetek synergii (epi>0) oraz średnia |epi|
    mask=~np.isnan(epi)
    if not np.any(mask): return 0.0, 0.0, 0.0
    vals=epi[mask]
    synergy=np.mean(vals>0)*100.0
    antagonism=np.mean(vals<0)*100.0
    mean_abs=np.mean(np.abs(vals))
    return synergy, antagonism, mean_abs

def main():
    frams=make_lib()
    gens=load_candidates()
    # wybierz >=3 różne genotypy: top, mid, lower (wg fitness)
    fits=[eval_fit(frams,g) for g in gens]
    print(fits)
    ok=[(g,f) for g,f in zip(gens,fits) if f!=FITNESS_VALUE_INFEASIBLE_SOLUTION]
    if len(ok)<3:
        print("Za mało poprawnych f9 genotypów."); return
    ok.sort(key=lambda x:x[1])
    picks=[ok[0][0], ok[len(ok)//2][0], ok[-1][0]]
    os.makedirs("epistasis_plots", exist_ok=True)
    for idx,g in enumerate(picks):
        epi, valid_idx, f0 = epistasis_matrix(frams, g)
        if epi is None:
            print(f"g{idx}: brak macierzy (zbyt wiele niepoprawnych wyłączeń)"); continue
        synergy, antagonism, mean_abs = summarize(epi)
        title=f"f9 epistasis g{idx} (fit0={f0:.3f})\nsynergy={synergy:.1f}%, antagonism={antagonism:.1f}%, mean|epi|={mean_abs:.3f}"
        plot_heatmap(epi, title, f"epistasis_plots/epistasis_f9_g{idx}.png")
        print(title)
    print("✓ Epistasis analysis done.")

if __name__=="__main__":
    main()