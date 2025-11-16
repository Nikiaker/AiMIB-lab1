import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ustawienia wykresów
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_logs_formats(genetic_formats):
    """Wczytuje wszystkie logi z eksperymentów dla różnych formatów genetycznych"""
    data = {}
    
    for fmt in genetic_formats:
        pattern = f"logs_formats/log-f{fmt}-run*.json"
        files = glob.glob(pattern)
        
        data[fmt] = []
        
        for file in sorted(files):
            try:
                with open(file, 'r') as f:
                    log = json.load(f)
                    data[fmt].append(log)
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

def plot_individual_runs_formats(data, genetic_formats, output_file='plot1_formats_individual_runs.png'):
    """
    Wykres 1: Każdy przebieg ewolucji jako osobna linia dla każdego formatu
    """
    plt.figure(figsize=(16, 9))
    
    # Filtruj formaty, które mają dane
    available_formats = [fmt for fmt in genetic_formats if data.get(fmt) and len(data[fmt]) > 0]
    
    if not available_formats:
        print("No data available for plot_individual_runs_formats")
        plt.close()
        return
    
    # Definicja kolorów dla każdego formatu
    colors_map = {
        0: plt.cm.Reds(np.linspace(0.3, 0.9, 1))[0],      # Czerwony dla f0
        1: plt.cm.Blues(np.linspace(0.3, 0.9, 1))[0],     # Niebieski dla f1
        4: plt.cm.Greens(np.linspace(0.3, 0.9, 1))[0],    # Zielony dla f4
        9: plt.cm.Purples(np.linspace(0.3, 0.9, 1))[0]    # Fioletowy dla f9
    }
    
    for fmt in available_formats:
        runs = data[fmt]
        color = colors_map.get(fmt, 'gray')
        
        for run_idx, run in enumerate(runs):
            if 'generations' not in run or not run['generations']:
                continue
                
            generations = [g['generation'] for g in run['generations']]
            best_fitness = [g['best_fitness'] for g in run['generations']]
            
            # Filtruj nieskończone wartości
            valid_indices = [i for i, f in enumerate(best_fitness) 
                           if f is not None and f != float('-inf') and not np.isnan(f)]
            if valid_indices:
                generations = [generations[i] for i in valid_indices]
                best_fitness = [best_fitness[i] for i in valid_indices]
            else:
                continue
            
            label = get_format_name(fmt) if run_idx == 0 else None
            alpha = 0.8 if run_idx == 0 else 0.3
            linewidth = 2.5 if run_idx == 0 else 1
            
            plt.plot(generations, best_fitness, color=color, 
                    alpha=alpha, linewidth=linewidth, label=label)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (vertpos)', fontsize=12)
    plt.title('Evolution Progress: All Individual Runs by Genetic Format', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_aggregated_with_std_formats(data, genetic_formats, output_file='plot2_formats_aggregated_std.png'):
    """
    Wykres 2: Średnia z odchyleniem standardowym dla każdego formatu
    """
    plt.figure(figsize=(16, 9))
    
    # Filtruj formaty, które mają dane
    available_formats = [fmt for fmt in genetic_formats if data.get(fmt) and len(data[fmt]) > 0]
    
    if not available_formats:
        print("No data available for plot_aggregated_with_std_formats")
        plt.close()
        return
    
    colors_map = {
        0: plt.cm.Reds(np.linspace(0.3, 0.9, 1))[0],
        1: plt.cm.Blues(np.linspace(0.3, 0.9, 1))[0],
        4: plt.cm.Greens(np.linspace(0.3, 0.9, 1))[0],
        9: plt.cm.Purples(np.linspace(0.3, 0.9, 1))[0]
    }
    
    for fmt in available_formats:
        runs = data[fmt]
        
        # Sprawdź, czy runs nie jest puste
        if not runs:
            continue
        
        # Znajdź maksymalną liczbę pokoleń
        valid_runs = [run for run in runs if 'generations' in run and run['generations']]
        if not valid_runs:
            continue
            
        max_gens = max(len(run['generations']) for run in valid_runs)
        
        # Macierz fitness dla wszystkich przebiegów
        fitness_matrix = []
        for run in valid_runs:
            fitness = [g['best_fitness'] if g['best_fitness'] is not None else np.nan 
                      for g in run['generations']]
            # Uzupełnij brakujące wartości ostatnią znaną wartością
            if len(fitness) < max_gens:
                last_valid = fitness[-1] if fitness else np.nan
                fitness.extend([last_valid] * (max_gens - len(fitness)))
            fitness_matrix.append(fitness)
        
        if not fitness_matrix:
            continue
            
        fitness_matrix = np.array(fitness_matrix)
        generations = np.arange(max_gens)
        
        # Oblicz średnią i std (ignorując NaN)
        mean_fitness = np.nanmean(fitness_matrix, axis=0)
        std_fitness = np.nanstd(fitness_matrix, axis=0)
        
        color = colors_map.get(fmt, 'gray')
        
        # Rysuj średnią
        plt.plot(generations, mean_fitness, color=color, linewidth=2.5, 
                label=get_format_name(fmt))
        
        # Rysuj obszar std (dzielony przez 2 dla czytelności)
        plt.fill_between(generations, 
                        mean_fitness - std_fitness/2,
                        mean_fitness + std_fitness/2,
                        color=color, alpha=0.2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (vertpos)', fontsize=12)
    plt.title('Evolution Progress: Mean ± 0.5×StdDev by Genetic Format', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_boxplots_formats(data, genetic_formats, output_file_prefix='plot3_formats_boxplot'):
    """
    Wykres 3: Boxploty dla fitness i czasu dla różnych formatów
    """
    # Zbierz dane do boxplotów
    fitness_data = []
    time_data = []
    labels = []
    colors_list = []
    
    colors_map = {
        0: plt.cm.Reds(np.linspace(0.5, 0.8, 1))[0],
        1: plt.cm.Blues(np.linspace(0.5, 0.8, 1))[0],
        4: plt.cm.Greens(np.linspace(0.5, 0.8, 1))[0],
        9: plt.cm.Purples(np.linspace(0.5, 0.8, 1))[0]
    }
    
    for fmt in genetic_formats:
        runs = data.get(fmt, [])
        
        if not runs:
            continue
        
        # Fitness najlepszego osobnika z każdego przebiegu
        final_fitness = []
        for run in runs:
            if ('best_individual' in run and 
                run['best_individual'] is not None and
                'fitness' in run['best_individual'] and
                run['best_individual']['fitness'] is not None):
                fitness_val = run['best_individual']['fitness']
                if fitness_val != float('-inf') and not np.isnan(fitness_val):
                    final_fitness.append(fitness_val)
        
        # Czas wykonania każdego przebiegu
        elapsed_times = [run['elapsed_time'] for run in runs 
                        if 'elapsed_time' in run and run['elapsed_time'] is not None]
        
        if final_fitness:
            fitness_data.append(final_fitness)
            time_data.append(elapsed_times)
            labels.append(get_format_name(fmt))
            colors_list.append(colors_map.get(fmt, 'gray'))
    
    if not fitness_data:
        print("No valid data for boxplots")
        return
    
    # Boxplot dla fitness i czasu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Fitness boxplot
    bp1 = ax1.boxplot(fitness_data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Genetic Format', fontsize=12)
    ax1.set_ylabel('Final Best Fitness (vertpos)', fontsize=12)
    ax1.set_title('Final Fitness Distribution by Genetic Format', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Time boxplot
    if any(time_data):
        bp2 = ax2.boxplot(time_data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp2['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Genetic Format', fontsize=12)
        ax2.set_ylabel('Elapsed Time (seconds)', fontsize=12)
        ax2.set_title('Computation Time by Genetic Format', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No timing data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_file_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"Saved {output_file_prefix}.png")
    plt.close()
    
    # Statystyki
    print("\n=== Statistics by Genetic Format ===")
    for i, fmt in enumerate(genetic_formats):
        if i < len(fitness_data) and fitness_data[i]:
            print(f"\n{get_format_name(fmt)}:")
            print(f"  Fitness: mean={np.mean(fitness_data[i]):.4f}, "
                  f"std={np.std(fitness_data[i]):.4f}, "
                  f"min={np.min(fitness_data[i]):.4f}, "
                  f"max={np.max(fitness_data[i]):.4f}")
            if i < len(time_data) and time_data[i]:
                print(f"  Time: mean={np.mean(time_data[i]):.2f}s, "
                      f"std={np.std(time_data[i]):.2f}s, "
                      f"min={np.min(time_data[i]):.2f}s, "
                      f"max={np.max(time_data[i]):.2f}s")

def generate_summary_table_formats(data, genetic_formats, output_file='summary_table_formats.txt'):
    """Generuje tabelę podsumowującą wyniki dla różnych formatów"""
    with open(output_file, 'w') as f:
        f.write("=" * 95 + "\n")
        f.write(" " * 30 + "GENETIC FORMATS COMPARISON\n")
        f.write("=" * 95 + "\n\n")
        
        f.write(f"{'Format':<25} {'Runs':<6} {'Mean Fit':<12} {'Std Fit':<12} "
                f"{'Min Fit':<12} {'Max Fit':<12} {'Mean Time':<12}\n")
        f.write("-" * 95 + "\n")
        
        for fmt in genetic_formats:
            runs = data.get(fmt, [])
            
            if not runs:
                continue
            
            final_fitness = []
            for run in runs:
                if ('best_individual' in run and 
                    run['best_individual'] is not None and
                    'fitness' in run['best_individual'] and
                    run['best_individual']['fitness'] is not None):
                    fitness_val = run['best_individual']['fitness']
                    if fitness_val != float('-inf') and not np.isnan(fitness_val):
                        final_fitness.append(fitness_val)
            
            elapsed_times = [run['elapsed_time'] for run in runs 
                            if 'elapsed_time' in run and run['elapsed_time'] is not None]
            
            if final_fitness:
                f.write(f"{get_format_name(fmt):<25} {len(runs):<6} "
                       f"{np.mean(final_fitness):<12.4f} "
                       f"{np.std(final_fitness):<12.4f} "
                       f"{np.min(final_fitness):<12.4f} "
                       f"{np.max(final_fitness):<12.4f} "
                       f"{np.mean(elapsed_times) if elapsed_times else 0:<12.2f}\n")
        
        f.write("=" * 95 + "\n")
        
        # Dodatkowe analizy
        f.write("\n\nADDITIONAL ANALYSIS:\n")
        f.write("-" * 95 + "\n\n")
        
        # Ranking formatów według średniego fitness
        format_means = []
        for fmt in genetic_formats:
            runs = data.get(fmt, [])
            if runs:
                final_fitness = []
                for run in runs:
                    if ('best_individual' in run and 
                        run['best_individual'] is not None and
                        'fitness' in run['best_individual'] and
                        run['best_individual']['fitness'] is not None):
                        fitness_val = run['best_individual']['fitness']
                        if fitness_val != float('-inf') and not np.isnan(fitness_val):
                            final_fitness.append(fitness_val)
                if final_fitness:
                    format_means.append((fmt, np.mean(final_fitness)))
        
        format_means.sort(key=lambda x: x[1], reverse=True)
        
        f.write("Ranking by Mean Fitness:\n")
        for rank, (fmt, mean_fit) in enumerate(format_means, 1):
            f.write(f"  {rank}. {get_format_name(fmt)}: {mean_fit:.4f}\n")
    
    print(f"Saved summary table to {output_file}")

def main():
    genetic_formats = [0, 1, 4, 9]
    
    print("Loading experiment logs for genetic formats...")
    data = load_logs_formats(genetic_formats)
    
    # Sprawdź, czy mamy dane
    total_runs = sum(len(runs) for runs in data.values())
    print(f"Loaded {total_runs} experiment runs across {len(genetic_formats)} genetic formats")
    
    if total_runs == 0:
        print("No data found! Make sure experiments have been run.")
        print("Looking for files matching: logs_formats/log-f*.json")
        return
    
    # Pokaż, które formaty mają dane
    print("\nData availability:")
    for fmt in genetic_formats:
        count = len(data.get(fmt, []))
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {get_format_name(fmt)}: {count} runs")
    
    print("\nGenerating plots...")
    
    # Wykres 1: Wszystkie przebiegi
    plot_individual_runs_formats(data, genetic_formats)
    
    # Wykres 2: Średnie z odchyleniem
    plot_aggregated_with_std_formats(data, genetic_formats)
    
    # Wykres 3: Boxploty
    plot_boxplots_formats(data, genetic_formats)
    
    # Tabela podsumowująca
    generate_summary_table_formats(data, genetic_formats)
    
    print("\n✓ All plots for genetic formats generated successfully!")

if __name__ == "__main__":
    main()