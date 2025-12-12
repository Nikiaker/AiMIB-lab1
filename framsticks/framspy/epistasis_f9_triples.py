import os, glob, itertools, numpy as np
from FramsticksLib import FramsticksLib
import plotly.graph_objects as go

FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0
SIM="eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim"
PREFIX="/*9*/"

def normalize_f9(geno: str) -> str:
    g = geno.strip()
    if g.startswith("/*"):
        if g.startswith("/**/"):
            g = "/*9*/" + g[len("/**/"):]
        elif not g.startswith("/*9*/"):
            end = g.find("*/")
            g = "/*9*/" + g[end+2:] if end != -1 else "/*9*/" + g
        return g
    else:
        return "/*9*/" + g

def parse_hof_file(fn):
    with open(fn) as fh:
        for ln in fh:
            ln=ln.strip()
            if ln.startswith("genotype:"):
                raw = ln.split(":",1)[1].strip()
                return normalize_f9(raw)
    return None

def load_candidates():
    import glob
    gens=[]
    for fn in sorted(glob.glob("genotypes_landscape/HoF-land-f9-run*.gen")):
        g=parse_hof_file(fn)
        if g: gens.append(g)
    # deduplikacja
    uniq=[]; seen=set()
    for g in gens:
        k=(len(g), g[:20])
        if k not in seen:
            uniq.append(g); seen.add(k)
    return uniq

def make_lib():
    import os
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
    arr=list(s)
    for i in idxs:
        arr[i]=''
    return ''.join(arr)

def prepare_effects(frams, g):
    """Zwraca C, valid_idx, E1, E2 zgodnie z dekompozycją z załącznika."""
    g = normalize_f9(g)
    n=len(g)
    C=eval_fit(frams, g)
    if C==FITNESS_VALUE_INFEASIBLE_SOLUTION:
        return None
    # E1
    import numpy as np, itertools
    single_ok=np.ones(n, dtype=bool)
    E1=np.full(n, np.nan)
    for i in range(n):
        gi=disable_chars(g,[i])
        fi=eval_fit(frams,gi)
        if fi==FITNESS_VALUE_INFEASIBLE_SOLUTION:
            single_ok[i]=False
        else:
            E1[i]=fi - C
    valid_idx=np.where(single_ok)[0]
    if len(valid_idx)<3:
        return None
    # E2 tylko dla valid_idx
    E2={}
    for i,j in itertools.combinations(valid_idx,2):
        gij=disable_chars(g,[i,j])
        fij=eval_fit(frams,gij)
        if fij!=FITNESS_VALUE_INFEASIBLE_SOLUTION and not np.isnan(E1[i]) and not np.isnan(E1[j]):
            E2[(i,j)] = fij - C - (E1[i] + E1[j])
    return C, valid_idx, E1, E2

def compute_E3(frams, g):
    """Czysta trójkowa epistaza E3 zgodnie z wzorem (odejmujemy E1 i E2)."""
    import numpy as np, itertools
    prep = prepare_effects(frams,g)
    if not prep: return None
    C, valid_idx, E1, E2 = prep
    triples=[]; E3=[]
    for i,j,k in itertools.combinations(valid_idx,3):
        gijk=disable_chars(g,[i,j,k])
        fijk=eval_fit(frams,gijk)
        if fijk==FITNESS_VALUE_INFEASIBLE_SOLUTION:
            continue
        # pobierz E2 dla wszystkich par
        e2_ij = E2.get((min(i,j),max(i,j)))
        e2_ik = E2.get((min(i,k),max(i,k)))
        e2_jk = E2.get((min(j,k),max(j,k)))
        if any(v is None for v in (e2_ij,e2_ik,e2_jk)):
            continue
        # E3 = f(g-{i,j,k}) - C - sum(E1) - sum(E2)
        e3 = (fijk - C) - (E1[i] + E1[j] + E1[k]) - (e2_ij + e2_ik + e2_jk)
        triples.append((i,j,k))
        E3.append(e3)
    return valid_idx, triples, np.array(E3), C

def summarize_values(vals):
    import numpy as np
    v = vals[~np.isnan(vals)]
    if len(v)==0:
        return dict(count=0, synergy=0.0, antagonism=0.0, mean_abs=0.0, sum=float(0.0), sum_abs=float(0.0))
    return dict(
        count=len(v),
        synergy=float(np.mean(v>0)*100.0),
        antagonism=float(np.mean(v<0)*100.0),
        mean_abs=float(np.mean(np.abs(v))),
        sum=float(np.sum(v)),
        sum_abs=float(np.sum(np.abs(v))),
    )

def plot_3d_heatmap(triples, values, title, out_html):
    """3D 'heatmap' jako chmura kulek na siatce indeksów genów."""
    if len(triples)==0: return
    xs=[t[0] for t in triples]
    ys=[t[1] for t in triples]
    zs=[t[2] for t in triples]
    vals=np.array(values)
    # skala kolorów: RdYlGn w Plotly
    fig=go.Figure(data=go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=5,
            color=vals,
            colorscale='RdYlGn',
            cmin=min(np.nanmin(vals), 0.0),
            cmax=max(np.nanmax(vals), 0.0),
            colorbar=dict(title="Epistasis")
        )
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="gene i",
            yaxis_title="gene j",
            zaxis_title="gene k"
        ),
        margin=dict(l=0,r=0,b=0,t=50)
    )
    fig.write_html(out_html, include_plotlyjs='cdn')

def main():
    frams=make_lib()
    gens=load_candidates()
    fits=[eval_fit(frams,g) for g in gens]
    ok=[(g,f) for g,f in zip(gens,fits) if f!=FITNESS_VALUE_INFEASIBLE_SOLUTION]
    if len(ok)<3:
        print("Za mało poprawnych f9 genotypów."); return
    ok.sort(key=lambda x:x[1])
    picks=[ok[0][0], ok[len(ok)//2][0], ok[-1][0]]
    os.makedirs("epistasis3_plots", exist_ok=True)
    for idx,g in enumerate(picks):
        res = compute_E3(frams, g)
        if not res:
            print(f"g{idx}: brak danych do trójek"); continue
        valid_idx, triples, e3, C = res
        s=summarize_values(e3)
        title=f"f9 triple epistasis (E3) g{idx} C={C:.3f}\ncount={s['count']}, synergy={s['synergy']:.1f}%, antagonism={s['antagonism']:.1f}%, mean|E3|={s['mean_abs']:.3f}"
        plot_3d_heatmap(triples, e3, title, f"epistasis3_plots/epistasis3_E3_g{idx}.html")
        print(title)
    print("✓ Triple epistasis analysis done.")

if __name__=="__main__":
    main()