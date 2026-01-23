set -euo pipefail

REPEATS=10
POP_SIZE=60
GENERATIONS=300
TOURNAMENT=7
MAX_PARTS=20
MAX_JOINTS=40

mkdir -p logs_jump genotypes_jump

# === Wariant A: f0, only-body, eps=0.02 (domyślny), centralne ===
run_variant_A() {
    RUN=$1
    TAG="jump-f0-eps002-center-run${RUN}"
    echo "[$(date '+%H:%M:%S')] Variant A: $TAG"
    python FramstickEvolutionFinal.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria-mini.sim;deterministic.sim;recording-body-coords.sim;only-body.sim" \
        -opt jumpdistance \
        -genformat 0 \
        -pxov 0 \
        -popsize 60 \
        -generations 300 \
        -tournament 7 \
        -max_numparts 20 \
        -max_numjoints 40 \
        -max_numneurons 0 \
        -max_numconnections 0 \
        -hof_size 1 \
        -eps 0.02 \
        -log_file "logs_jump/log-jump-f0-eps002-center-run${RUN}.json" \
        -hof_savefile "genotypes_jump/HoF-jump-f0-eps002-center-run${RUN}.gen" \
        > "logs_jump/console-${TAG}.txt" 2>&1
}

# === Wariant B: f1 z neuronami, eps=0.02, centralne ===
run_variant_B() {
    RUN=$1
    TAG="jump-f1-eps002-center-run${RUN}"
    echo "[$(date '+%H:%M:%S')] Variant B: $TAG"
    python FramstickEvolutionFinal.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria-mini.sim;deterministic.sim;recording-body-coords.sim" \
        -opt jumpdistance \
        -genformat 1 \
        -pxov 0 \
        -popsize 60 \
        -generations 300 \
        -tournament 7 \
        -max_numparts 20 \
        -max_numjoints 40 \
        -max_numneurons 10 \
        -max_numconnections 20 \
        -hof_size 1 \
        -eps 0.02 \
        -log_file "logs_jump/log-jump-f1-eps002-center-run${RUN}.json" \
        -hof_savefile "genotypes_jump/HoF-jump-f1-eps002-center-run${RUN}.gen" \
        > "logs_jump/console-${TAG}.txt" 2>&1
}

# === Wariant C: f0, eps=0.01 (niższy próg), centralne ===
run_variant_C() {
    RUN=$1
    TAG="jump-f0-eps001-center-run${RUN}"
    echo "[$(date '+%H:%M:%S')] Variant C: $TAG"
    python FramstickEvolutionFinal.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria-mini.sim;deterministic.sim;recording-body-coords.sim;only-body.sim" \
        -opt jumpdistance \
        -genformat 0 \
        -pxov 0 \
        -popsize 60 \
        -generations 300 \
        -tournament 7 \
        -max_numparts 20 \
        -max_numjoints 40 \
        -max_numneurons 0 \
        -max_numconnections 0 \
        -hof_size 1 \
        -eps 0.01 \
        -log_file "logs_jump/log-jump-f0-eps001-center-run${RUN}.json" \
        -hof_savefile "genotypes_jump/HoF-jump-f0-eps001-center-run${RUN}.gen" \
        > "logs_jump/console-${TAG}.txt" 2>&1
}

# === Wariant D: f0, eps=0.02, LOSOWE umieszczenie ===
run_variant_D() {
    RUN=$1
    TAG="jump-f0-eps002-random-run${RUN}"
    echo "[$(date '+%H:%M:%S')] Variant D: $TAG"
    python FramstickEvolutionFinal.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria-mini.sim;deterministic.sim;recording-body-coords.sim;only-body.sim;placement_random.sim" \
        -opt jumpdistance \
        -genformat 0 \
        -pxov 0 \
        -popsize 60 \
        -generations 300 \
        -tournament 7 \
        -max_numparts 20 \
        -max_numjoints 40 \
        -max_numneurons 0 \
        -max_numconnections 0 \
        -hof_size 1 \
        -eps 0.02 \
        -placement random \
        -log_file "logs_jump/log-jump-f0-eps002-random-run${RUN}.json" \
        -hof_savefile "genotypes_jump/HoF-jump-f0-eps002-random-run${RUN}.gen" \
        > "logs_jump/console-${TAG}.txt" 2>&1
}

export -f run_variant_A
export -f run_variant_B
export -f run_variant_C
export -f run_variant_D
export DIR_WITH_FRAMS_LIBRARY

echo "=========================================="
echo "Starting all 4 variants × 10 repeats (40 jobs total)"
echo "=========================================="
START_TIME=$(date +%s)

# Uruchom wszystkie warianty równolegle z `parallel`
# Każdy wariant ma osobny job pool, więc mogą działać jednocześnie
(
    parallel --bar --jobs 0 run_variant_A ::: $(seq 1 "$REPEATS")
) &
PID_A=$!

(
    parallel --bar --jobs 0 run_variant_B ::: $(seq 1 "$REPEATS")
) &
PID_B=$!

(
    parallel --bar --jobs 0 run_variant_C ::: $(seq 1 "$REPEATS")
) &
PID_C=$!

(
    parallel --bar --jobs 0 run_variant_D ::: $(seq 1 "$REPEATS")
) &
PID_D=$!

# Czekaj na wszystkie procesy
wait $PID_A $PID_B $PID_C $PID_D

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "All jump experiments completed!"
echo "Total time: ${DURATION}s"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Logs:      logs_jump/log-jump-*.json"
echo "  Genotypes: genotypes_jump/HoF-jump-*.gen"
echo ""
echo "Run plotting:"
echo "  python plot_jump.py"