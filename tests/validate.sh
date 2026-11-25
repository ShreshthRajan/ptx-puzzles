# Maintainers' check: puzzles fail as given, solutions pass, tests/answers/wrong fails, tests/answers/right passes,
# and puzzle 7 reports its race every time. Run with: python remote/run.py --gpu H100 --script tests/validate.sh
set -o pipefail
export DEBIAN_FRONTEND=noninteractive PATH=/opt/conda/bin:$PATH
SYNCSAN_REF=${SYNCSAN_REF:-c8e9ce444d386061e9b68287d6d217786d08e76b}   # syncsan 0.10.5
echo "=== setup [$(date +%H:%M:%S)]"
(apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null) 2>&1 | tail -1
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
nvcc --version | tail -1
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
pip install -q "cmake>=3.29" "syncsan @ git+https://github.com/ShreshthRajan/syncsan@$SYNCSAN_REF" 2>&1 | grep -v WARNING
syncsan build --arch "$ARCH" > /tmp/syncsan_build.log 2>&1 || { tail -30 /tmp/syncsan_build.log; exit 1; }
echo "syncsan built for sm_$ARCH: $(syncsan where)"
echo "=== grade [$(date +%H:%M:%S)]"
python grade.py --validate
echo "=== wrong answers (each must FAIL)"
for d in tests/answers/wrong/*; do python grade.py --dir $d; done 2>&1 | grep -E "^Puzzle|^  (races|adversarial|native|PASS|FAIL)"
echo "=== other correct answers (each must PASS)"
for d in tests/answers/right/*; do python grade.py --dir $d; done 2>&1 | grep -E "^Puzzle|^  (races|adversarial|native|cost|PASS|FAIL)"
echo "=== puzzle 7 as given, 10 runs"
nvcc -O2 -lineinfo -arch=sm_$ARCH -I puzzles -o /tmp/p7 puzzles/07_grid_barrier.cu
for i in $(seq 10); do LD_PRELOAD=$(syncsan where) /tmp/p7 2>&1 | grep -c "syncsan: RACE"; done | sort | uniq -c
