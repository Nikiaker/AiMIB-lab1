
echo "========================================="
echo "  GENETIC FORMATS EXPERIMENT PIPELINE"
echo "========================================="

# 1. Uruchom eksperymenty
echo ""
echo "Step 1: Running evolution experiments for different genetic formats..."
echo "This will test: f0, f1, f4, f9"
bash run_experiment2.sh

# 2. Generuj wykresy
echo ""
echo "Step 2: Generating plots and analysis..."
python plot_result2.py

# 3. Wyświetl podsumowanie
echo ""
echo "Step 3: Experiment summary"
echo "========================================="
cat summary_table_formats.txt

echo ""
echo "✓ Full genetic formats experiment completed!"
echo ""
echo "Generated files:"
echo "  - plot1_formats_individual_runs.png"
echo "  - plot2_formats_aggregated_std.png"
echo "  - plot3_formats_boxplot.png"
echo "  - summary_table_formats.txt"
echo "  - genotypes_formats/HoF-*.gen (Hall of Fame genotypes)"
echo "  - logs_formats/log-*.json (detailed logs)"