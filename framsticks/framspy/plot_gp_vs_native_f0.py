import glob, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

NATIVE_DIR = "logs_landscape"
GP_DIR = "logs_gp"


def load_runs(pattern):
    runs = []
    for fn in sorted(glob.glob(pattern)):
        try:
            with open(fn) as f:
                runs.append(json.load(f))
        except Exception:
            pass
    return runs


def extract_best_series(run):
    gens = run.get("generations", [])
    if not gens:
        return None
    vals = []
    for g in gens:
        v = g.get("best_fitness", None)
        if v is None:
            v = g.get("max", None)
        if v is None:
            return None
        try:
            vals.append(float(v))
        except Exception:
            return None
    return vals if vals else None


def series_from_runs(runs):
    mats = []
    maxlen = 0
    finals = []

    for r in runs:
        vals = extract_best_series(r)
        if vals:
            maxlen = max(maxlen, len(vals))
            mats.append(vals)

        bi = r.get("best_individual", {})
        if isinstance(bi, dict) and "fitness" in bi and bi["fitness"] not in (None, float("-inf")):
            try:
                finals.append(float(bi["fitness"]))
            except Exception:
                pass

    if not mats:
        return None, np.array(finals, dtype=float)

    mats = [v + [v[-1]] * (maxlen - len(v)) for v in mats]
    return np.array(mats, dtype=float), np.array(finals, dtype=float)


def plot_individual_runs(native_runs, gp_runs, outfile="gp_vs_native_f0_individual.png"):
    plt.figure(figsize=(14, 8))

    native_color = "#E74C3C"
    gp_color = "#2E86C1"

    # NATIVE
    shown_label = False
    for r in native_runs:
        s = extract_best_series(r)
        if not s:
            continue
        x = np.arange(len(s))
        plt.plot(
            x, s,
            color=native_color,
            alpha=0.18,
            linewidth=1.0,
            label="native f0" if not shown_label else None
        )
        shown_label = True

    # GP
    shown_label = False
    for r in gp_runs:
        s = extract_best_series(r)
        if not s:
            continue
        x = np.arange(len(s))
        plt.plot(
            x, s,
            color=gp_color,
            alpha=0.18,
            linewidth=1.0,
            label="GP→f0" if not shown_label else None
        )
        shown_label = True

    plt.xlabel("Generation")
    plt.ylabel("Best vertpos")
    plt.title("Evolution Progress: Individual runs (only-body): native f0 vs GP→f0")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def main():
    native_runs = load_runs(f"{NATIVE_DIR}/log-land-f0-run*.json")
    gp_runs = load_runs(f"{GP_DIR}/log-gp-f0-run*.json")

    native_mat, native_final = series_from_runs(native_runs)
    gp_mat, gp_final = series_from_runs(gp_runs)

    # 1) Wykres: pojedyncze przebiegi
    plot_individual_runs(native_runs, gp_runs)

    # 2) krzywe Mean ± 0.5*Std
    plt.figure()
    if native_mat is not None:
        mean = np.nanmean(native_mat, axis=0)
        std = np.nanstd(native_mat, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label="native f0", color="#E74C3C")
        plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, color="#E74C3C", alpha=0.2)
    if gp_mat is not None:
        mean = np.nanmean(gp_mat, axis=0)
        std = np.nanstd(gp_mat, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label="GP→f0", color="#2E86C1")
        plt.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, color="#2E86C1", alpha=0.2)
    plt.xlabel("Generation")
    plt.ylabel("Best vertpos")
    plt.title("Mean ± 0.5·StdDev vertpos (only-body): native f0 vs GP→f0")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gp_vs_native_f0_mean.png", dpi=300)
    plt.close()

    # 3) boxplot wyników końcowych
    plt.figure()
    data = []
    labels = []
    if native_final is not None and len(native_final) > 0:
        data.append(native_final)
        labels.append("native f0")
    if gp_final is not None and len(gp_final) > 0:
        data.append(gp_final)
        labels.append("GP→f0")

    bp = plt.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    colors = ["#E74C3C", "#2E86C1"]
    for b, c in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        b.set_facecolor(c)
        b.set_alpha(0.65)

    plt.ylabel("Final best vertpos")
    plt.title("Final vertpos distribution: native f0 vs GP→f0")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("gp_vs_native_f0_box.png", dpi=300)
    plt.close()

    print("✓ Plots saved:")
    print("  - gp_vs_native_f0_individual.png")
    print("  - gp_vs_native_f0_mean.png")
    print("  - gp_vs_native_f0_box.png")


if __name__ == "__main__":
    main()