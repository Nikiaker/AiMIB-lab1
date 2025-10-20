# Funkcja uruchamiająca pojedynczy eksperyment
run_experiment() {
    N=$1
    
    echo "Running experiment $N/10..."
    python FramsticksEvolution.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim" \
        -opt vertpos \
        -max_numparts 30 \
        -max_numgenochars 50 \
        -initialgenotype "/*9*/BLU" \
        -popsize 50 \
        -generations 20 \
        -hof_size 1 \
        -hof_savefile "experiment/HoF-f9-$N.gen"
}

# Eksportujemy funkcję, żeby parallel mogła z niej korzystać
export -f run_experiment

# Uruchomienie równolegle (domyślnie wykorzysta wszystkie dostępne rdzenie)
seq 1 10 | parallel -j 0 run_experiment {}

# Jeśli chcesz ograniczyć do np. 4 wątków, użyj: parallel -j 4

echo "All experiments completed!"