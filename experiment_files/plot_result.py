import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ustawienia wykresów
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_logs(mutation_rates):
    """Wczytuje wszystkie logi z eksperymentów"""
    data = {}
    
    for mut_rate in mutation_rates:
        mut_str = str(mut_rate).zfill(3)
        pattern = f"logs/log-mut{mut_str}-run*.json"
        files = glob.glob(pattern)
        
        data[mut_rate] = []
        
        for file in sorted(files):
            try:
                with open(file, 'r') as f:
                    log = json.load(f)
                    data[mut_rate].append(log)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return data

def plot_individual_runs(data, mutation_rates, output_file='plot1_individual_runs.png'):
    """
    Wykres 1: Każdy przebieg ewolucji jako osobna linia
    """
    plt.figure(figsize=(14, 8))
    
    # Filtruj siły mutacji, które mają dane
    available_rates = [rate for rate in mutation_rates if data.get(rate) and len(data[rate]) > 0]
    
    if not available_rates:
        print("No data available for plot_individual_runs")
        plt.close()
        return
    
    # Definicja kolorów dla każdej siły mutacji
    colors = plt.cm.viridis(np.linspace(0, 1, len(available_rates)))
    color_map = dict(zip(available_rates, colors))
    
    for mut_rate in available_rates:
        runs = data[mut_rate]
        color = color_map[mut_rate]
        
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
            
            label = f"Mutation {mut_rate}%" if run_idx == 0 else None
            alpha = 0.7 if run_idx == 0 else 0.4
            linewidth = 2 if run_idx == 0 else 1
            
            plt.plot(generations, best_fitness, color=color, 
                    alpha=alpha, linewidth=linewidth, label=label)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (vertpos)')
    plt.title('Evolution Progress: All Individual Runs')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_aggregated_with_std(data, mutation_rates, output_file='plot2_aggregated_std.png'):
    """
    Wykres 2: Średnia z odchyleniem standardowym
    """
    plt.figure(figsize=(14, 8))
    
    # Filtruj siły mutacji, które mają dane
    available_rates = [rate for rate in mutation_rates if data.get(rate) and len(data[rate]) > 0]
    
    if not available_rates:
        print("No data available for plot_aggregated_with_std")
        plt.close()
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(available_rates)))
    color_map = dict(zip(available_rates, colors))
    
    for mut_rate in available_rates:
        runs = data[mut_rate]
        
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
        
        color = color_map[mut_rate]
        
        # Rysuj średnią
        plt.plot(generations, mean_fitness, color=color, linewidth=2, 
                label=f'Mutation {mut_rate}%')
        
        # Rysuj obszar std
        plt.fill_between(generations, 
                        mean_fitness - std_fitness/2,  # Dzielimy przez 2 dla czytelności
                        mean_fitness + std_fitness/2,
                        color=color, alpha=0.2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (vertpos)')
    plt.title('Evolution Progress: Mean ± 0.5×StdDev')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close()

def plot_boxplots(data, mutation_rates, output_file_prefix='plot3_boxplot'):
    """
    Wykres 3: Boxploty dla fitness i czasu
    """
    # Zbierz dane do boxplotów
    fitness_data = []
    time_data = []
    labels = []
    
    for mut_rate in mutation_rates:
        runs = data.get(mut_rate, [])
        
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
            labels.append(f"{mut_rate}%")
    
    if not fitness_data:
        print("No valid data for boxplots")
        return
    
    # Boxplot dla fitness
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bp1 = ax1.boxplot(fitness_data, labels=labels, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Mutation Rate')
    ax1.set_ylabel('Final Best Fitness (vertpos)')
    ax1.set_title('Final Fitness Distribution by Mutation Rate')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot dla czasu
    if any(time_data):  # Sprawdź czy są jakieś dane czasu
        bp2 = ax2.boxplot(time_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Mutation Rate')
        ax2.set_ylabel('Elapsed Time (seconds)')
        ax2.set_title('Computation Time by Mutation Rate')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No timing data available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_file_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"Saved {output_file_prefix}.png")
    plt.close()
    
    # Dodatkowo: statystyki
    print("\n=== Statistics ===")
    for i, mut_rate in enumerate([mr for mr in mutation_rates if data.get(mr)]):
        if i < len(fitness_data) and fitness_data[i]:
            print(f"\nMutation {mut_rate}%:")
            print(f"  Fitness: mean={np.mean(fitness_data[i]):.4f}, "
                  f"std={np.std(fitness_data[i]):.4f}, "
                  f"min={np.min(fitness_data[i]):.4f}, "
                  f"max={np.max(fitness_data[i]):.4f}")
            if i < len(time_data) and time_data[i]:
                print(f"  Time: mean={np.mean(time_data[i]):.2f}s, "
                      f"std={np.std(time_data[i]):.2f}s")

def generate_summary_table(data, mutation_rates, output_file='summary_table.txt'):
    """Generuje tabelę podsumowującą wyniki"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Mutation':<10} {'Runs':<6} {'Mean Fit':<12} {'Std Fit':<12} "
                f"{'Min Fit':<12} {'Max Fit':<12} {'Mean Time':<12}\n")
        f.write("-" * 80 + "\n")
        
        for mut_rate in mutation_rates:
            runs = data.get(mut_rate, [])
            
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
                f.write(f"{mut_rate}%{'':<7} {len(runs):<6} "
                       f"{np.mean(final_fitness):<12.4f} "
                       f"{np.std(final_fitness):<12.4f} "
                       f"{np.min(final_fitness):<12.4f} "
                       f"{np.max(final_fitness):<12.4f} "
                       f"{np.mean(elapsed_times) if elapsed_times else 0:<12.2f}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Saved summary table to {output_file}")

def main():
    mutation_rates = [0, 5, 10, 20, 30, 40, 50]
    
    print("Loading experiment logs...")
    data = load_logs(mutation_rates)
    
    # Sprawdź, czy mamy dane
    total_runs = sum(len(runs) for runs in data.values())
    print(f"Loaded {total_runs} experiment runs across {len(mutation_rates)} mutation rates")
    
    if total_runs == 0:
        print("No data found! Make sure experiments have been run.")
        print("Looking for files matching: logs/log-mut*.json")
        return
    
    # Pokaż, które siły mutacji mają dane
    print("\nData availability:")
    for rate in mutation_rates:
        count = len(data.get(rate, []))
        status = "✓" if count > 0 else "✗"
        print(f"  {status} Mutation {rate}%: {count} runs")
    
    print("\nGenerating plots...")
    
    # Wykres 1: Wszystkie przebiegi
    plot_individual_runs(data, mutation_rates)
    
    # Wykres 2: Średnie z odchyleniem
    plot_aggregated_with_std(data, mutation_rates)
    
    # Wykres 3: Boxploty
    plot_boxplots(data, mutation_rates)
    
    # Tabela podsumowująca
    generate_summary_table(data, mutation_rates)
    
    print("\n✓ All plots generated successfully!")

if __name__ == "__main__":
    main()