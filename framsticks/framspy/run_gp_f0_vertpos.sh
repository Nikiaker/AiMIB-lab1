set -euo pipefail

REPEATS=50
POP_SIZE=50
GENERATIONS=120
TOURNAMENT=7
CX=0.2
MUT=0.9

mkdir -p logs_gp genotypes_gp

run_one() {
  RUN=$1
  TAG="gp-f0-run${RUN}"
  echo "[$(date '+%H:%M:%S')] $TAG"
  python FramstickGPEvolution_f0.py \
    -path "$DIR_WITH_FRAMS_LIBRARY" \
    -sim "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim" \
    -popsize 50 \
    -generations 120 \
    -tournament 7 \
    -cxpb 0.2 \
    -mutpb 0.9 \
    -max_numparts 30 \
    -max_numjoints 60 \
    -max_numneurons 0 \
    -max_numconnections 0 \
    -hof_savefile "genotypes_gp/HoF-${TAG}.gen" \
    -log_file "logs_gp/log-${TAG}.json" \
    -seed $((1000+RUN)) \
    > "logs_gp/console-${TAG}.txt" 2>&1
}
export -f run_one

echo "Running GP->f0 vertpos runs (only-body)..."
parallel --bar --jobs 0 run_one ::: $(seq 1 "$REPEATS")
echo "Done GP runs."