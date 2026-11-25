# Experiment: how often syncsan reports puzzle 7's reset race, by barrier crossings and block count (10 runs each).
export PATH=/opt/conda/bin:$PATH DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null) 2>&1 | tail -1
pip install -q "cmake>=3.29" "syncsan @ git+https://github.com/ShreshthRajan/syncsan@11d7393cfd456d663cc358d2004bdb2463af8d97" 2>&1 | grep -v WARN
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
syncsan build --arch $ARCH > /tmp/b.log 2>&1 || tail /tmp/b.log
SO=$(syncsan where)
P=puzzles/07_grid_barrier.cu
v() { name=$1; shift; cp $P /tmp/$name.cu; for e in "$@"; do sed -i "$e" /tmp/$name.cu; done
  nvcc -O2 -lineinfo -arch=sm_$ARCH -I puzzles -o /tmp/$name /tmp/$name.cu || return
  out=$(LD_PRELOAD=$SO /tmp/$name 2>&1); echo "$name: $(grep -c 'syncsan: RACE' <<< "$out") races; $(grep -E '^a:' <<< "$out")"; grep -A3 'syncsan: RACE' <<< "$out" | head -8; grep -E '^syncsan' <<< "$out" | grep -v RACE | tail -4; }
rate() { name=$1; cp $P /tmp/$name.cu; sed -i "s/STEPS = 1024;/STEPS = $2;/; s/constexpr int B = 8,/constexpr int B = $3,/" /tmp/$name.cu
  nvcc -O2 -lineinfo -arch=sm_$ARCH -I puzzles -o /tmp/$name /tmp/$name.cu || return
  hit=0; t0=$(date +%s); for i in $(seq 10); do n=$(LD_PRELOAD=$SO /tmp/$name 2>&1 | grep -c 'syncsan: RACE'); [ $n -gt 0 ] && hit=$((hit+1)); done
  echo "STEPS=$2 B=$3: detected in $hit of 10 runs, $(( ($(date +%s)-t0)/10 )) s per run"; }
rate s4 4 8
rate s16 16 8
rate s64 64 8
rate s256 256 8
rate s64b32 64 32
rate s256b32 256 32
