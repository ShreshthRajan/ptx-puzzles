# Execute puzzles.ipynb end to end (as given, and with puzzle 1 solved) with nbconvert.
# Run with: python remote/run.py --gpu H100 --script tests/notebook_check.sh   (copies the uploaded tree in place of the git clone, since the repo may be private)
set -o pipefail
export DEBIAN_FRONTEND=noninteractive PATH=/opt/conda/bin:$PATH
echo "=== setup [$(date +%H:%M:%S)]"
(apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null) 2>&1 | tail -1
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
pip install -q nbconvert nbclient ipykernel 2>&1 | grep -v WARNING
python - <<'PY'
import json, pathlib, copy
src = pathlib.Path("/work/ptx-puzzles/puzzles.ipynb")
nb = json.loads(src.read_text())
setup = "".join(nb["cells"][1]["source"])
old = "[ -d ptx-puzzles ] || git clone -q https://github.com/ShreshthRajan/ptx-puzzles"
assert old in setup, setup
nb["cells"][1]["source"] = setup.replace(old, "[ -d ptx-puzzles ] || cp -r /work/ptx-puzzles .")
for d in ("given", "sol1"):
    pathlib.Path(f"/work/nb_{d}").mkdir()
pathlib.Path("/work/nb_given/puzzles.ipynb").write_text(json.dumps(nb))
nb2 = copy.deepcopy(nb)
i = next(i for i, c in enumerate(nb2["cells"]) if "".join(c["source"]).startswith("%%writefile ptx-puzzles/puzzles/01_message_passing.cu"))
nb2["cells"][i]["source"] = "%%writefile ptx-puzzles/puzzles/01_message_passing.cu\n" + pathlib.Path("/work/ptx-puzzles/solutions/01_message_passing.cu").read_text()
pathlib.Path("/work/nb_sol1/puzzles.ipynb").write_text(json.dumps(nb2))
print("notebooks prepared; puzzle 1 cell is", i)
PY
for d in given sol1; do
  echo "=== execute nb_$d [$(date +%H:%M:%S)]"
  (cd /work/nb_$d && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 --output executed.ipynb puzzles.ipynb 2>&1 | tail -3)
  python - /work/nb_$d/executed.ipynb <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    print(f"--- cell {i}: {''.join(c['source']).splitlines()[0]}")
    for o in c.get("outputs", []):
        if o["output_type"] == "stream": print("".join(o["text"]), end="")
        elif o["output_type"] == "error": print("ERROR", o["ename"], o["evalue"])
        elif "data" in o: print("".join(o["data"].get("text/plain", "")))
PY
  # the file the puzzle-1 cell wrote, compared byte for byte with its source
  if [ $d = sol1 ]; then cmp /work/nb_sol1/ptx-puzzles/puzzles/01_message_passing.cu /work/ptx-puzzles/solutions/01_message_passing.cu && echo "puzzle 1 file == solution, byte for byte"; fi
  if [ $d = given ]; then for f in /work/ptx-puzzles/puzzles/*.cu; do cmp -s $f /work/nb_given/ptx-puzzles/puzzles/$(basename $f) || echo "differs: $(basename $f)"; done; echo "given files compared"; fi
done
