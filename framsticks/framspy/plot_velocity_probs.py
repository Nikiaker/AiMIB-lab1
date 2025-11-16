import json, glob, numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16,10)
plt.rcParams['font.size'] = 10

def load_logs(probs):
    data = {}
    for p in probs:
        pattern = f"logs_velocity/log-vel-f-prawd{p}-run*.json"
        runs = []
        for fname in sorted(glob.glob(pattern)):
            try:
                with open(fname,'r') as f:
                    runs.append(json.load(f))
            except Exception as e:
                print("Failed", fname, e)
        data[p] = runs
    return data

def extract_series(run):
    gens = []
    vals = []
    for g in run.get("generations", []):
        gens.append(g.get("generation"))
        # velocity jako best_fitness (bo -opt velocity)
        v = g.get("best_fitness")
        if v is None or v == float('-inf') or np.isnan(v):
            v = np.nan
        vals.append(v)
    return gens, vals

def plot_individual(data, probs, out='vel_plot_individual.png'):
    cols = len(probs)
    fig, axes = plt.subplots(1, cols, figsize=(6*cols,5), sharey=True)
    if cols==1: axes=[axes]
    for ax, p in zip(axes, probs):
        runs = data[p]
        for i, r in enumerate(runs):
            g,v = extract_series(r)
            if not g: continue
            ax.plot(g, v, alpha=0.25, color='#2c3e50')
        ax.set_title(f'prawd={p}')
        ax.set_xlabel('Generation'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('Best velocity')
    fig.suptitle('Velocity evolution (all runs)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print("Saved", out); plt.close()

def plot_aggregated(data, probs, out='vel_plot_mean.png'):
    fig, ax = plt.subplots(figsize=(12,6))
    palette = {0:'#E74C3C', 1:'#3498DB', 2: "#27AE60", 3: "#8E44AD"}
    for p in probs:
        runs = data[p]
        matrices=[]
        max_len=0
        for r in runs:
            g,v = extract_series(r)
            if not g: continue
            max_len = max(max_len, len(v))
        for r in runs:
            _,v = extract_series(r)
            if not v: continue
            if len(v)<max_len:
                v = v + [v[-1]]*(max_len-len(v))
            matrices.append(v)
        if not matrices: continue
        arr=np.array(matrices,dtype=float)
        mean=np.nanmean(arr,axis=0)
        std=np.nanstd(arr,axis=0)
        gens=np.arange(max_len)
        ax.plot(gens, mean, label=f'prawd={p}', color=palette[p], linewidth=2.2)
        ax.fill_between(gens, mean-0.5*std, mean+0.5*std, color=palette[p], alpha=0.2)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best velocity')
    ax.set_title('Mean ± 0.5·StdDev velocity'); ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=300); print("Saved", out); plt.close()

def plot_box(data, probs, out='vel_plot_box.png'):
    final_vals=[]
    labels=[]
    colors=[]
    palette = {0:'#E74C3C', 1:'#3498DB', 2: "#27AE60", 3: "#8E44AD"}
    for p in probs:
        vals=[]
        for r in data[p]:
            bi = r.get('best_individual')
            if bi and bi.get('fitness') not in (None, float('-inf')):
                vals.append(bi['fitness'])
        if vals:
            final_vals.append(vals); labels.append(f'prawd={p}'); colors.append(palette[p])
    if not final_vals:
        print("No data for boxplot"); return
    fig, ax = plt.subplots(figsize=(8,6))
    bp = ax.boxplot(final_vals, labels=labels, patch_artist=True, widths=0.6)
    for patch,c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_ylabel('Final best velocity')
    ax.set_title('Final velocity distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=300); print("Saved", out); plt.close()
    # Statystyka
    print("\nSUMMARY:")
    for lab,vals in zip(labels, final_vals):
        print(f"{lab}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} median={np.median(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f}")

def main():
    probs=[0,1,2,3]
    data=load_logs(probs)
    total=sum(len(data[p]) for p in probs)
    if total==0:
        print("Brak danych w logs_velocity/"); return
    print("Runs:", {p:len(data[p]) for p in probs})
    plot_individual(data, probs)
    plot_aggregated(data, probs)
    plot_box(data, probs)
    print("✓ Generated velocity comparison plots (probabilities 0 vs 1 vs 2 vs 3).")

if __name__ == "__main__":
    main()