set -euo pipefail

# Parametry bazowe (baseline)
GENFORMAT=1              # f1
REPEATS=10               # liczba przebiegów
POP_SIZE=60              # ≥50
GENERATIONS=300          # "odpowiednia dlugosc" – można zmienić
TOURNAMENT_SIZE=7
PXOV=0                   # -pxov 0 (brak crossoveru)
PROBS=(2)                # nadrzędna pętla (na razie tylko 0)

mkdir -p logs_velocity genotypes_velocity

run_velocity_experiment() {
    PROB=$1   # odpowiada %%P (tu zawsze 0, ale zostawione na przyszłość)
    RUN=$2    # odpowiada %%N

    TAG="vel-f${GENFORMAT}-prawd${PROB}-run${RUN}"
    echo "[$(date '+%H:%M:%S')] $TAG"

    python FramstickEvolutionModified2.py \
        -path "$DIR_WITH_FRAMS_LIBRARY" \
        -sim "eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim;wlasne-prawd-${PROB}.sim" \
        -opt velocity \
        -genformat 1 \
        -pxov 0 \
        -popsize 60 \
        -generations 300 \
        -tournament 7 \
        -max_numparts 15 \
        -max_numjoints 30 \
        -max_numneurons 20 \
        -max_numconnections 30 \
        -hof_size 1 \
        -hof_savefile "genotypes_velocity/HoF-${TAG}.gen" \
        -log_file "logs_velocity/log-${TAG}.json" \
        > "logs_velocity/console-${TAG}.txt" 2>&1
}
export -f run_velocity_experiment

echo "Starting velocity baseline experiments (f1)..."
parallel --bar --jobs 0 run_velocity_experiment {1} {2} ::: "${PROBS[@]}" ::: $(seq 1 "$REPEATS")
echo "All velocity baseline experiments completed."