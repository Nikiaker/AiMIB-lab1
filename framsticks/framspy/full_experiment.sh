echo "========================================="
echo "  FULL MUTATION EXPERIMENT PIPELINE"
echo "========================================="

# 1. Uruchom eksperymenty
echo ""
echo "Step 1: Running evolution experiments..."
bash run_mutation_experiments.sh

# 2. Generuj wykresy
echo ""
echo "Step 2: Generating plots and analysis..."
python plot_results.py

# 3. Wyświetl podsumowanie
echo ""
echo "Step 3: Experiment summary"
echo "========================================="
cat summary_table.txt

echo ""
echo "✓ Full experiment completed!"
echo ""
echo "Generated files:"
echo "  - plot1_individual_runs.png"
echo "  - plot2_aggregated_std.png"
echo "  - plot3_boxplot.png"
echo "  - summary_table.txt"
echo "  - genotypes/HoF-*.gen (Hall of Fame genotypes)"
echo "  - logs/log-*.json (detailed logs)"