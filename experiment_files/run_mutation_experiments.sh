MUTATION_RATES=(0 005 010 020 030 040 050)
REPEATS=10

# Funkcja uruchamiająca pojedynczy eksperyment
run_mutation_experiment() {
    M=$1
    N=$2

    echo "[$(date '+%H:%M:%S')] Starting: mutation=$M%, run=$N/$REPEATS"
    
    python FramstickEvolutionModified.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;f9-mut-$M.sim" \
        -opt vertpos \
        -max_numparts 30 \
        -max_numgenochars 50 \
        -initialgenotype "/*9*/BLU" \
        -popsize 50 \
        -generations 200 \
        -tournament 7 \
        -hof_size 1 \
        -hof_savefile "genotypes/HoF-f9-mut$M-run$N.gen" \
        -log_file "logs/log-mut$M-run$N.json" \
        -stagnation_generations 50 \
        > "logs/console-mut$M-run$N.txt" 2>&1
    
    echo "[$(date '+%H:%M:%S')] Finished: mutation=$M%, run=$N/$REPEATS"
}

# Eksportujemy funkcję i zmienne
export -f run_mutation_experiment
export REPEATS

# Uruchomienie eksperymentów równolegle
echo "================================"
echo "Starting mutation experiments"
echo "Mutation rates: ${MUTATION_RATES[@]}"
echo "Repeats per rate: $REPEATS"
echo "================================"

# Wszystkie kombinacje równolegle z paskiem postępu
parallel --bar --jobs 0 \
    run_mutation_experiment {1} {2} \
    ::: "${MUTATION_RATES[@]}" ::: $(seq 1 $REPEATS)

echo ""
echo "All experiments completed!"
echo "Results saved in:"
echo "  - Genotypes: genotypes/HoF-*.gen"
echo "  - Logs: logs/log-*.json"
echo "  - Console outputs: logs/console-*.txt"