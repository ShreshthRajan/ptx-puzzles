# Runs on a CPU sandbox (--gpu none): compile every puzzle and solution for sm_80, sm_90 and sm_120.
export PATH=/opt/conda/bin:$PATH
nvcc --version | tail -1
fail=0
for arch in 80 90 120; do
  for f in puzzles/[0-9]*.cu solutions/[0-9]*.cu; do
    nvcc -O2 -lineinfo -arch=sm_$arch -I puzzles -o /tmp/p "$f" 2> /tmp/err || { echo "FAIL sm_$arch $f"; cat /tmp/err; fail=1; continue; }
    [ -s /tmp/err ] && { echo "WARN sm_$arch $f"; cat /tmp/err; }
  done
  echo "sm_$arch done"
done
echo "compile check: $([ $fail = 0 ] && echo all compiled || echo failures)"
exit $fail
