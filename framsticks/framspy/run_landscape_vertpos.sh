set -euo pipefail

GENFORMATS=(0 1 4 9)
REPEATS=50
POP_SIZE=50
GENERATIONS=120
TOURNAMENT=7
PXOV=0
PMUT=0.9

mkdir -p logs_landscape genotypes_landscape samples_landscape

run_one() {
  FMT=$1
  RUN=$2
  TAG="land-f${FMT}-run${RUN}"
  echo "[$(date '+%H:%M:%S')] $TAG"
  python FramstickEvolutionModified2.py \
    -path "$DIR_WITH_FRAMS_LIBRARY" \
    -sim "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim" \
    -opt vertpos \
    -genformat $FMT \
    -pxov 0 \
    -pmut 0.9 \
    -popsize 50 \
    -generations 120 \
    -tournament 7 \
    -max_numparts 30 \
    -max_numjoints 60 \
    -max_numneurons 0 \
    -max_numconnections 0 \
    -hof_size 1 \
    -hof_savefile "genotypes_landscape/HoF-${TAG}.gen" \
    -log_file "logs_landscape/log-${TAG}.json" \
    > "logs_landscape/console-${TAG}.txt" 2>&1
}
export -f run_one

echo "Running vertpos landscape runs (only-body)..."
parallel --bar --jobs 0 run_one {1} {2} ::: "${GENFORMATS[@]}" ::: $(seq 1 "$REPEATS")
echo "Done base runs."