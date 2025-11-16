import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ustawienia wykresów
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_combined_logs(genetic_formats):
    """Wczytuje wszystkie logi z eksperymentów dla różnych formatów i wariantów"""
    data = {}
    
    for fmt in genetic_formats:
        data[fmt] = {
            'original': [],
            'plateau': []
        }
        
        # Original version
        pattern_orig = f"logs_combined/log-f{fmt}-original-run*.json"
        files_orig = glob.glob(pattern_orig)
        
        for file in sorted(files_orig):
            try:
                with open(file, 'r') as f:
                    log = json.load(f)
                    data[fmt]['original'].append(log)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Plateau escape version
        pattern_plat = f"logs_combined/log-f{fmt}-plateau-run*.json"
        files_plat = glob.glob(pattern_plat)
        
        for file in sorted(files_plat):
            try:
                with open(file, 'r') as f:
                    log = json.load(f)
                    data[fmt]['plateau'].append(log)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return data

def get_format_name(fmt):
    """Zwraca pełną nazwę formatu genetycznego"""
    names = {
        0: "f0",
        1: "f1",
        4: "f4",
        9: "f9"
    }
    return names.get(fmt, f"f{fmt}")

def get_category_label(fmt, variant):
    """Zwraca pełną nazwę kategorii"""
    format_name = get_format_name(fmt)
    variant_suffix = " (plateau escape)" if variant == 'plateau' else " (original)"
    return format_name + variant_suffix

def plot_individual_runs_combined(data, genetic_formats, output_file='plot1_combined_individual_runs.png'):
    """
    Wykres 1: Każdy przebieg ewolucji jako osobna linia dla każdego formatu i wariantu
    8 kategorii: f0-orig, f0-plat, f1-orig, f1-plat, f4-orig, f4-plat, f9-orig, f9-plat
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Evolution Progress: All Individual Runs (Original vs Plateau Escape)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Kolory
    color_original = '#E74C3C'  # Czerwony
    color_plateau = '#3498DB'   # Niebieski
    
    for idx, fmt in enumerate(genetic_formats):
        row = idx // 2
        col = (idx % 2) * 2
        
        # Original (lewa kolumna)
        ax_orig = axes[row, col]
        runs_orig = data[fmt]['original']
        
        for run_idx, run in enumerate(runs_orig):
            if 'generations' not in run or not run['generations']:
                continue
                
            generations = [g['generation'] for g in run['generations']]
            best_fitness = [g['best_fitness'] for g in run['generations']]
            
            valid_indices = [i for i, f in enumerate(best_fitness) 
                           if f is not None and f != float('-inf') and not np.isnan(f)]
            if valid_indices:
                generations = [generations[i] for i in valid_indices]
                best_fitness = [best_fitness[i] for i in valid_indices]
            else:
                continue
            
            alpha = 0.7 if run_idx == 0 else 0.3
            linewidth = 2 if run_idx == 0 else 1
            
            ax_orig.plot(generations, best_fitness, color=color_original, 
                        alpha=alpha, linewidth=linewidth)
        
        ax_orig.set_title(f'{get_format_name(fmt)} - Original', fontsize=12, fontweight='bold')
        ax_orig.set_xlabel('Generation')
        ax_orig.set_ylabel('Best Fitness (vertpos)')
        ax_orig.grid(True, alpha=0.3)
        
        # Plateau escape (prawa kolumna)
        ax_plat = axes[row, col + 1]
        runs_plat = data[fmt]['plateau']
        
        for run_idx, run in enumerate(runs_plat):
            if 'generations' not in run or not run['generations']:
                continue
                
            generations = [g['generation'] for g in run['generations']]
            best_fitness = [g['best_fitness'] for g in run['generations']]
            
            valid_indices = [i for i, f in enumerate(best_fitness) 
                           if f is not None and f != float('-inf') and not np.isnan(f)]
            if valid_indices:
                generations = [generations[i] for i in valid_indices]
                best_fitness = [best_fitness[i] for i in valid_indices]
            else:
                continue
            
            alpha = 0.7 if run_idx == 0 else 0.3
            linewidth = 2 if run_idx == 0 else 1
            
            ax_plat.plot(generations, best_fitness, color=color_plateau, 
                        alpha=alpha, linewidth=linewidth)
        
        ax_plat.set_title(f'{get_format_name(fmt)} - Plateau Escape', fontsize=12, fontweight='bold')
        ax_plat.set_xlabel('Generation')
        ax_plat.set_ylabel('Best Fitness (vertpos)')
        ax_plat.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_aggregated_combined(data, genetic_formats, output_file='plot2_combined_aggregated.png'):
    """
    Wykres 2: Średnia z odchyleniem - porównanie original vs plateau dla każdego formatu
    4 subploty (po jednym na format), każdy zawiera 2 linie (original i plateau)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Evolution Progress: Mean ± 0.5×StdDev (Original vs Plateau Escape)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    color_original = '#E74C3C'
    color_plateau = '#3498DB'
    
    for idx, fmt in enumerate(genetic_formats):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Original
        runs_orig = data[fmt]['original']
        valid_runs_orig = [run for run in runs_orig if 'generations' in run and run['generations']]
        
        if valid_runs_orig:
            max_gens_orig = max(len(run['generations']) for run in valid_runs_orig)
            fitness_matrix_orig = []
            
            for run in valid_runs_orig:
                fitness = [g['best_fitness'] if g['best_fitness'] is not None else np.nan 
                          for g in run['generations']]
                if len(fitness) < max_gens_orig:
                    last_valid = fitness[-1] if fitness else np.nan
                    fitness.extend([last_valid] * (max_gens_orig - len(fitness)))
                fitness_matrix_orig.append(fitness)
            
            if fitness_matrix_orig:
                fitness_matrix_orig = np.array(fitness_matrix_orig)
                generations_orig = np.arange(max_gens_orig)
                mean_fitness_orig = np.nanmean(fitness_matrix_orig, axis=0)
                std_fitness_orig = np.nanstd(fitness_matrix_orig, axis=0)
                
                ax.plot(generations_orig, mean_fitness_orig, color=color_original, 
                       linewidth=2.5, label='Original', linestyle='-')
                ax.fill_between(generations_orig, 
                               mean_fitness_orig - std_fitness_orig/2,
                               mean_fitness_orig + std_fitness_orig/2,
                               color=color_original, alpha=0.2)
        
        # Plateau escape
        runs_plat = data[fmt]['plateau']
        valid_runs_plat = [run for run in runs_plat if 'generations' in run and run['generations']]
        
        if valid_runs_plat:
            max_gens_plat = max(len(run['generations']) for run in valid_runs_plat)
            fitness_matrix_plat = []
            
            for run in valid_runs_plat:
                fitness = [g['best_fitness'] if g['best_fitness'] is not None else np.nan 
                          for g in run['generations']]
                if len(fitness) < max_gens_plat:
                    last_valid = fitness[-1] if fitness else np.nan
                    fitness.extend([last_valid] * (max_gens_plat - len(fitness)))
                fitness_matrix_plat.append(fitness)
            
            if fitness_matrix_plat:
                fitness_matrix_plat = np.array(fitness_matrix_plat)
                generations_plat = np.arange(max_gens_plat)
                mean_fitness_plat = np.nanmean(fitness_matrix_plat, axis=0)
                std_fitness_plat = np.nanstd(fitness_matrix_plat, axis=0)
                
                ax.plot(generations_plat, mean_fitness_plat, color=color_plateau, 
                       linewidth=2.5, label='Plateau Escape', linestyle='-')
                ax.fill_between(generations_plat, 
                               mean_fitness_plat - std_fitness_plat/2,
                               mean_fitness_plat + std_fitness_plat/2,
                               color=color_plateau, alpha=0.2)
        
        ax.set_title(f'{get_format_name(fmt)}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Best Fitness (vertpos)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_boxplots_combined(data, genetic_formats, output_file='plot3_combined_boxplots.png'):
    """
    Wykres 3: Boxploty dla 8 kategorii (fitness i czas)
    """
    # Zbierz dane
    fitness_data = []
    time_data = []
    labels = []
    colors_list = []
    
    color_original = '#E74C3C'
    color_plateau = '#3498DB'
    
    for fmt in genetic_formats:
        # Original
        runs_orig = data[fmt]['original']
        final_fitness_orig = []
        elapsed_times_orig = []
        
        for run in runs_orig:
            if ('best_individual' in run and 
                run['best_individual'] is not None and
                'fitness' in run['best_individual'] and
                run['best_individual']['fitness'] is not None):
                fitness_val = run['best_individual']['fitness']
                if fitness_val != float('-inf') and not np.isnan(fitness_val):
                    final_fitness_orig.append(fitness_val)
            
            if 'elapsed_time' in run and run['elapsed_time'] is not None:
                elapsed_times_orig.append(run['elapsed_time'])
        
        if final_fitness_orig:
            fitness_data.append(final_fitness_orig)
            time_data.append(elapsed_times_orig)
            labels.append(f'{get_format_name(fmt)}\nOriginal')
            colors_list.append(color_original)
        
        # Plateau
        runs_plat = data[fmt]['plateau']
        final_fitness_plat = []
        elapsed_times_plat = []
        
        for run in runs_plat:
            if ('best_individual' in run and 
                run['best_individual'] is not None and
                'fitness' in run['best_individual'] and
                run['best_individual']['fitness'] is not None):
                fitness_val = run['best_individual']['fitness']
                if fitness_val != float('-inf') and not np.isnan(fitness_val):
                    final_fitness_plat.append(fitness_val)
            
            if 'elapsed_time' in run and run['elapsed_time'] is not None:
                elapsed_times_plat.append(run['elapsed_time'])
        
        if final_fitness_plat:
            fitness_data.append(final_fitness_plat)
            time_data.append(elapsed_times_plat)
            labels.append(f'{get_format_name(fmt)}\nPlateau')
            colors_list.append(color_plateau)
    
    if not fitness_data:
        print("No valid data for boxplots")
        return
    
    # Boxploty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Final Results Comparison: Original vs Plateau Escape', 
                 fontsize=16, fontweight='bold')
    
    # Fitness
    bp1 = ax1.boxplot(fitness_data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Final Best Fitness (vertpos)', fontsize=12)
    ax1.set_title('Final Fitness Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Time
    if any(time_data):
        bp2 = ax2.boxplot(time_data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp2['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Elapsed Time (seconds)', fontsize=12)
        ax2.set_title('Computation Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No timing data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()
    
    # Statystyki
    print("\n" + "="*100)
    print("DETAILED STATISTICS: ORIGINAL vs PLATEAU ESCAPE")
    print("="*100)
    
    for i, label in enumerate(labels):
        if i < len(fitness_data) and fitness_data[i]:
            print(f"\n{label}:")
            print(f"  Fitness: mean={np.mean(fitness_data[i]):.4f}, "
                  f"std={np.std(fitness_data[i]):.4f}, "
                  f"median={np.median(fitness_data[i]):.4f}, "
                  f"min={np.min(fitness_data[i]):.4f}, "
                  f"max={np.max(fitness_data[i]):.4f}")
            if i < len(time_data) and time_data[i]:
                print(f"  Time: mean={np.mean(time_data[i]):.2f}s, "
                      f"std={np.std(time_data[i]):.2f}s, "
                      f"median={np.median(time_data[i]):.2f}s")

def generate_comparison_table(data, genetic_formats, output_file='summary_table_combined.txt'):
    """Generuje tabelę porównawczą"""
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write(" "*40 + "PLATEAU ESCAPE EXPERIMENT RESULTS\n")
        f.write("="*120 + "\n\n")
        
        f.write(f"{'Format & Variant':<30} {'Runs':<6} {'Mean Fit':<12} {'Std Fit':<12} "
                f"{'Median':<12} {'Min Fit':<12} {'Max Fit':<12} {'Mean Time':<12}\n")
        f.write("-"*120 + "\n")
        
        results_for_improvement = []
        
        for fmt in genetic_formats:
            # Original
            runs_orig = data[fmt]['original']
            final_fitness_orig = []
            elapsed_times_orig = []
            
            for run in runs_orig:
                if ('best_individual' in run and 
                    run['best_individual'] is not None and
                    'fitness' in run['best_individual'] and
                    run['best_individual']['fitness'] is not None):
                    fitness_val = run['best_individual']['fitness']
                    if fitness_val != float('-inf') and not np.isnan(fitness_val):
                        final_fitness_orig.append(fitness_val)
                
                if 'elapsed_time' in run and run['elapsed_time'] is not None:
                    elapsed_times_orig.append(run['elapsed_time'])
            
            if final_fitness_orig:
                mean_orig = np.mean(final_fitness_orig)
                f.write(f"{get_format_name(fmt) + ' (Original)':<30} {len(runs_orig):<6} "
                       f"{mean_orig:<12.4f} "
                       f"{np.std(final_fitness_orig):<12.4f} "
                       f"{np.median(final_fitness_orig):<12.4f} "
                       f"{np.min(final_fitness_orig):<12.4f} "
                       f"{np.max(final_fitness_orig):<12.4f} "
                       f"{np.mean(elapsed_times_orig) if elapsed_times_orig else 0:<12.2f}\n")
            else:
                mean_orig = None
            
            # Plateau
            runs_plat = data[fmt]['plateau']
            final_fitness_plat = []
            elapsed_times_plat = []
            
            for run in runs_plat:
                if ('best_individual' in run and 
                    run['best_individual'] is not None and
                    'fitness' in run['best_individual'] and
                    run['best_individual']['fitness'] is not None):
                    fitness_val = run['best_individual']['fitness']
                    if fitness_val != float('-inf') and not np.isnan(fitness_val):
                        final_fitness_plat.append(fitness_val)
                
                if 'elapsed_time' in run and run['elapsed_time'] is not None:
                    elapsed_times_plat.append(run['elapsed_time'])
            
            if final_fitness_plat:
                mean_plat = np.mean(final_fitness_plat)
                f.write(f"{get_format_name(fmt) + ' (Plateau Escape)':<30} {len(runs_plat):<6} "
                       f"{mean_plat:<12.4f} "
                       f"{np.std(final_fitness_plat):<12.4f} "
                       f"{np.median(final_fitness_plat):<12.4f} "
                       f"{np.min(final_fitness_plat):<12.4f} "
                       f"{np.max(final_fitness_plat):<12.4f} "
                       f"{np.mean(elapsed_times_plat) if elapsed_times_plat else 0:<12.2f}\n")
            else:
                mean_plat = None
            
            f.write("-"*120 + "\n")
            
            # Calculate improvement
            if mean_orig is not None and mean_plat is not None:
                improvement = ((mean_plat - mean_orig) / max(abs(mean_orig), 0.0001)) * 100
                results_for_improvement.append((fmt, mean_orig, mean_plat, improvement))
        
        f.write("="*120 + "\n\n")
        
        # Improvement analysis
        f.write("IMPROVEMENT ANALYSIS (Plateau Escape vs Original):\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Format':<15} {'Original Mean':<20} {'Plateau Mean':<20} {'Improvement':<20}\n")
        f.write("-"*120 + "\n")
        
        for fmt, orig, plat, improvement in results_for_improvement:
            direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            f.write(f"{get_format_name(fmt):<15} {orig:<20.4f} {plat:<20.4f} "
                   f"{direction} {abs(improvement):<18.2f}%\n")
        
        f.write("="*120 + "\n")
    
    print(f"\nSaved summary table to {output_file}")

def main():
    genetic_formats = [0, 1, 4, 9]
    
    print("Loading combined experiment logs (original + plateau escape)...")
    data = load_combined_logs(genetic_formats)
    
    # Check data availability
    total_runs = 0
    print("\nData availability:")
    for fmt in genetic_formats:
        count_orig = len(data[fmt]['original'])
        count_plat = len(data[fmt]['plateau'])
        total_runs += count_orig + count_plat
        status_orig = "✓" if count_orig > 0 else "✗"
        status_plat = "✓" if count_plat > 0 else "✗"
        print(f"  {get_format_name(fmt)}:")
        print(f"    {status_orig} Original: {count_orig} runs")
        print(f"    {status_plat} Plateau Escape: {count_plat} runs")
    
    if total_runs == 0:
        print("\nNo data found! Make sure experiments have been run.")
        print("Looking for files matching: logs_combined/log-*.json")
        return
    
    print(f"\nLoaded {total_runs} total runs")
    print("\nGenerating plots...")
    
    # Generate all plots
    plot_individual_runs_combined(data, genetic_formats)
    plot_aggregated_combined(data, genetic_formats)
    plot_boxplots_combined(data, genetic_formats)
    generate_comparison_table(data, genetic_formats)
    
    print("\n✓ All combined plots generated successfully!")
    print("\nGenerated files:")
    print("  - plot1_combined_individual_runs.png (8 subplots: 4 formats × 2 variants)")
    print("  - plot2_combined_aggregated.png (4 subplots: each shows original vs plateau)")
    print("  - plot3_combined_boxplots.png (8 categories side by side)")
    print("  - summary_table_combined.txt (detailed statistics and improvement analysis)")

if __name__ == "__main__":
    main()