# Runs in the sandbox: install syncsan for this GPU, then grade. Arguments go to grade.py.
set -o pipefail
export DEBIAN_FRONTEND=noninteractive PATH=/opt/conda/bin:$PATH
SYNCSAN_REF=${SYNCSAN_REF:-11d7393cfd456d663cc358d2004bdb2463af8d97}   # syncsan 0.10.2
echo "=== setup [$(date +%H:%M:%S)]"
(apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null) 2>&1 | tail -1
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
nvcc --version | tail -1
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
pip install -q "cmake>=3.29" "syncsan @ git+https://github.com/ShreshthRajan/syncsan@$SYNCSAN_REF" 2>&1 | grep -v WARNING
syncsan build --arch "$ARCH" > /tmp/syncsan_build.log 2>&1 || { tail -30 /tmp/syncsan_build.log; exit 1; }
echo "syncsan built for sm_$ARCH: $(syncsan where)"
echo "=== grade [$(date +%H:%M:%S)]"
python grade.py "$@"
