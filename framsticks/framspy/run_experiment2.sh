GENETIC_FORMATS=(0 1 4 9)
REPEATS=10

# Tworzenie struktury katalogów
echo "Creating directory structure..."
mkdir -p data results_formats logs_formats genotypes_formats

# Funkcja uruchamiająca pojedynczy eksperyment
run_format_experiment() {
    FORMAT=$1
    N=$2
    
    echo "[$(date '+%H:%M:%S')] Starting: format=f$FORMAT, run=$N/$REPEATS"
    
    python FramstickEvolutionModified.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim;uneven-ground.sim" \
        -opt vertpos \
        -max_numparts 30 \
        -genformat "$FORMAT" \
        -popsize 50 \
        -generations 200 \
        -tournament 7 \
        -hof_size 1 \
        -hof_savefile "genotypes_formats/HoF-f$FORMAT-run$N.gen" \
        -log_file "logs_formats/log-f$FORMAT-run$N.json" \
        -stagnation_generations 50 \
        > "logs_formats/console-f$FORMAT-run$N.txt" 2>&1
    
    echo "[$(date '+%H:%M:%S')] Finished: format=f$FORMAT, run=$N/$REPEATS"
}

# Eksportujemy funkcję i zmienne
export -f run_format_experiment
export REPEATS

# Uruchomienie eksperymentów równolegle
echo "================================"
echo "Starting genetic format experiments"
echo "Formats: ${GENETIC_FORMATS[@]}"
echo "Repeats per format: $REPEATS"
echo "================================"

# Wszystkie kombinacje równolegle z paskiem postępu
parallel --bar --jobs 0 \
    run_format_experiment {1} {2} \
    ::: "${GENETIC_FORMATS[@]}" ::: $(seq 1 $REPEATS)

echo ""
echo "All experiments completed!"
echo "Results saved in:"
echo "  - Genotypes: genotypes_formats/HoF-*.gen"
echo "  - Logs: logs_formats/log-*.json"
echo "  - Console outputs: logs_formats/console-*.txt"