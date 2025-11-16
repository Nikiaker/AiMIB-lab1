GENETIC_FORMATS=(0 1 4 9)
REPEATS=10

# Tworzenie katalogów
mkdir -p logs_combined genotypes_combined

# Funkcja uruchamiająca eksperyment
run_combined_experiment() {
    FORMAT=$1
    N=$2
    PLATEAU_ESCAPE=$3
    
    if [ "$PLATEAU_ESCAPE" = "true" ]; then
        SUFFIX="plateau"
        EXTRA_ARG="-plateau_escape"
    else
        SUFFIX="original"
        EXTRA_ARG=""
    fi
    
    echo "[$(date '+%H:%M:%S')] f$FORMAT-$SUFFIX-run$N"
    
    python FramstickEvolutionModified2.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim;uneven-ground.sim" \
        -opt vertpos \
        -max_numparts 30 \
        -genformat "$FORMAT" \
        -popsize 50 \
        -generations 200 \
        -tournament 7 \
        -hof_size 1 \
        -hof_savefile "genotypes_combined/HoF-f$FORMAT-$SUFFIX-run$N.gen" \
        -log_file "logs_combined/log-f$FORMAT-$SUFFIX-run$N.json" \
        -stagnation_generations 50 \
        $EXTRA_ARG \
        > "logs_combined/console-f$FORMAT-$SUFFIX-run$N.txt" 2>&1
}

export -f run_combined_experiment

echo "Starting combined experiments (original + plateau escape)..."

# Uruchom oba warianty dla każdego formatu
parallel --bar --jobs 0 \
    run_combined_experiment {1} {2} {3} \
    ::: "${GENETIC_FORMATS[@]}" ::: $(seq 1 $REPEATS) ::: "false" "true"

echo "All experiments completed!"