import glob
import json
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10

# Wzorce glob dla różnych wariantów (dostosuj do swoich nazw logów)
VARIANTS = {
    # "A: f0, eps=0.02, center": "logs_jump/log-jump-f0-eps002-center-run*.json",
    # "B: f1, eps=0.02, center": "logs_jump/log-jump-f1-eps002-center-run*.json",
    # "C: f0, eps=0.01, center": "logs_jump/log-jump-f0-eps001-center-run*.json",
    # "D: f0, eps=0.02, random": "logs_jump/log-jump-f0-eps002-random-run*.json",
    "A: f1, eps=0.02, center": "logs_jump/log-jump-f1-eps002-center-run*.json",
    "B: f1, eps=0.02, center": "logs_jump/log-jump-f1-eps002-center-run*.json",
    "C: f1, eps=0.01, center": "logs_jump/log-jump-f1-eps001-center-run*.json",
    "D: f1, eps=0.02, random": "logs_jump/log-jump-f1-eps002-random-run*.json",
}

FITNESS_INFEASIBLE = -999999.0


def load_jsons(pattern: str) -> List[dict]:
    out = []
    for fn in sorted(glob.glob(pattern)):
        try:
            with open(fn) as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out


def extract_best_series(run: dict) -> Optional[List[float]]:
    """Wyciąga serię best_fitness z kolejnych generacji."""
    gens = run.get("generations", [])
    vals = []
    for g in gens:
        v = g.get("best_fitness", None)
        if v is None:
            v = g.get("max", None)
        if v is None:
            return None
        try:
            fv = float(v)
            # ignoruj infeasible
            if fv <= FITNESS_INFEASIBLE + 1:
                fv = np.nan
            vals.append(fv)
        except Exception:
            return None
    return vals if len(vals) > 0 else None


def extract_final_best(run: dict) -> Optional[float]:
    """Wyciąga końcowy najlepszy wynik z runa."""
    bi = run.get("best_individual", None)
    if isinstance(bi, dict) and "fitness" in bi:
        try:
            fv = float(bi["fitness"])
            return fv if fv > FITNESS_INFEASIBLE + 1 else None
        except Exception:
            pass
    s = extract_best_series(run)
    if s:
        # ostatnia wartość nie-NaN
        for v in reversed(s):
            if not np.isnan(v):
                return v
    return None


def stack_series(runs: List[dict]) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Zwraca:
      - macierz [run x generation] z best_fitness (uzupełniona ostatnią wartością)
      - wektor z finalnymi wynikami każdego runa
    """
    series = []
    finals = []
    maxlen = 0
    for r in runs:
        s = extract_best_series(r)
        if s is not None:
            maxlen = max(maxlen, len(s))
            series.append(s)
        f = extract_final_best(r)
        if f is not None:
            finals.append(f)

    if not series:
        return None, np.array(finals, dtype=float)

    # uzupełnij krótsze serie ostatnią wartością
    padded = []
    for s in series:
        if len(s) < maxlen:
            last_valid = s[-1]
            for i in range(len(s) - 1, -1, -1):
                if not np.isnan(s[i]):
                    last_valid = s[i]
                    break
            s = s + [last_valid] * (maxlen - len(s))
        padded.append(s)

    return np.array(padded, dtype=float), np.array(finals, dtype=float)


def plot_individual_runs(variants_data: dict, outfile="jump_individual_runs.png"):
    """Wykres 1: wszystkie przebiegi (osobne linie) dla każdego wariantu."""
    plt.figure(figsize=(16, 10))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    color_map = {}
    for i, k in enumerate(sorted(variants_data.keys())):
        color_map[k] = colors[i % len(colors)]

    for variant_name in sorted(variants_data.keys()):
        runs = variants_data[variant_name]
        shown_label = False
        for r in runs:
            s = extract_best_series(r)
            if not s:
                continue
            x = np.arange(len(s))
            y = np.array(s, dtype=float)
            plt.plot(
                x, y,
                color=color_map[variant_name],
                alpha=0.2,
                linewidth=1.0,
                label=variant_name if not shown_label else None
            )
            shown_label = True

    plt.xlabel("Generation")
    plt.ylabel("Best jump distance")
    plt.title("Evolution Progress: Individual runs (jump distance)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_mean_std(variants_data: dict, outfile="jump_mean_std.png"):
    """Wykres 2: Mean ± 0.5·StdDev dla każdego wariantu."""
    plt.figure(figsize=(16, 10))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    color_map = {}
    for i, k in enumerate(sorted(variants_data.keys())):
        color_map[k] = colors[i % len(colors)]

    for variant_name in sorted(variants_data.keys()):
        runs = variants_data[variant_name]
        mat, _ = stack_series(runs)
        if mat is None or mat.shape[0] == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=variant_name, color=color_map[variant_name], linewidth=2)
        plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std,
                         color=color_map[variant_name], alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Best jump distance")
    plt.title("Mean ± 0.5·StdDev jump distance")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_boxplot(variants_data: dict, outfile="jump_boxplot.png"):
    """Wykres 3: Boxplot końcowych wyników dla każdego wariantu."""
    plt.figure(figsize=(12, 8))
    data = []
    labels = []
    colors_list = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    for variant_name in sorted(variants_data.keys()):
        runs = variants_data[variant_name]
        _, finals = stack_series(runs)
        if finals.size > 0:
            data.append(finals)
            labels.append(variant_name)

    if len(data) == 0:
        print("No data for boxplot.")
        return

    bp = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors_list[i % len(colors_list)])
        box.set_alpha(0.65)

    plt.ylabel("Final best jump distance")
    plt.title("Final jump distance distribution (all variants)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def main():
    # Wczytaj logi dla każdego wariantu
    variants_data = {}
    for vname, pattern in VARIANTS.items():
        runs = load_jsons(pattern)
        if len(runs) > 0:
            variants_data[vname] = runs
            print(f"Loaded {len(runs)} runs for variant '{vname}'")
        else:
            print(f"No logs found for variant '{vname}' (pattern: {pattern})")

    if not variants_data:
        print("No data loaded. Check log file paths and patterns in VARIANTS.")
        return

    # Generuj 3 wykresy
    plot_individual_runs(variants_data, outfile="jump_individual_runs.png")
    plot_mean_std(variants_data, outfile="jump_mean_std.png")
    plot_boxplot(variants_data, outfile="jump_boxplot.png")

    print("✓ Plots saved:")
    print("  - jump_individual_runs.png")
    print("  - jump_mean_std.png")
    print("  - jump_boxplot.png")


if __name__ == "__main__":
    main()