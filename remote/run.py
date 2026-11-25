"""Grade on a rented GPU: one command, no local GPU needed.

    pip install "daytona[v0-sdk]"
    export DAYTONA_API_KEY=...
    python remote/run.py                       # grade every puzzle on an RTX 5090
    python remote/run.py --gpu H100 -- 3 7     # grade puzzles 3 and 7 on an H100 (arguments after -- go to grade.py)
    python remote/run.py -- --validate         # the maintainers' check: puzzles fail, solutions pass

It uploads this folder with your edits, installs syncsan and NVBit in the sandbox, runs grade.py, prints its output,
saves it under results/, and deletes the sandbox. A first run spends about four minutes on setup.
"""
import argparse
import io
import pathlib
import sys
import tarfile
import time

from daytona import Daytona
from daytona.v0 import CreateSandboxFromImageParams, GpuType, Resources
from daytona.v0.common.process import SessionExecuteRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="RTX-5090", help="RTX-5090 (default), RTX-4090, H100, ... or none for a CPU sandbox")
ap.add_argument("--script", default=str(ROOT / "remote" / "job.sh"))
ap.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel", help="any image with CUDA 12 nvcc")
ap.add_argument("--minutes", type=float, default=60, help="give up after this long")
ap.add_argument("args", nargs=argparse.REMAINDER)
a = ap.parse_args()
args = a.args[1:] if a.args[:1] == ["--"] else a.args

out = ROOT / "results" / time.strftime(f"{pathlib.Path(a.script).stem}_{a.gpu}_%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
def say(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()

buf = io.BytesIO()
skip = ("/build", "/results", "/.git", "/.venv", "__pycache__")
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(ROOT, arcname="ptx-puzzles", filter=lambda i: None if any(s in i.name for s in skip) else i)

d = Daytona().v0
t0 = time.time()
res = Resources(cpu=4, memory=8, disk=10) if a.gpu == "none" else Resources(cpu=8, memory=32, disk=100, gpu=1, gpu_type=[GpuType(a.gpu)])
while True:   # Daytona may allow one GPU sandbox at a time: wait for a free slot
    try:
        sb = d.create(CreateSandboxFromImageParams(image=a.image, auto_delete_interval=0, auto_stop_interval=0, resources=res), timeout=0)
        break
    except Exception as e:
        if time.time() - t0 > a.minutes * 60 or not any(w in str(e).lower() for w in ("limit", "quota", "exceed", "available")):
            raise
        say(f"waiting for a sandbox: {str(e).splitlines()[0][:160]}")
        time.sleep(60)
say(f"sandbox {sb.id} ({a.gpu}) up after {time.time() - t0:.0f} s")
try:
    sb.fs.upload_file(buf.getvalue(), "/tmp/tree.tgz")
    sb.fs.upload_file(pathlib.Path(a.script).read_bytes(), "/tmp/job.sh")
    sb.process.exec("mkdir -p /work && tar xzf /tmp/tree.tgz -C /work", timeout=600)
    sess = "job%d" % int(time.time())
    sb.process.create_session(sess)
    cmd = ("cd /work/ptx-puzzles && bash /tmp/job.sh " + " ".join(args) +
           " > /work/job.log 2>&1; echo EXIT $? >> /work/job.log")
    sb.process.execute_session_command(sess, SessionExecuteRequest(command=cmd, run_async=True))
    seen = 0
    while time.time() - t0 < a.minutes * 60:
        time.sleep(15)
        txt = sb.process.exec("cat /work/job.log 2>/dev/null", timeout=60).result or ""
        if len(txt) > seen:
            new = txt[seen:]; seen = len(txt)
            print(new, end="", flush=True); log.write(new); log.flush()
        if "\nEXIT " in txt or txt.startswith("EXIT "):
            break
    else:
        say("TIMED OUT")
finally:
    sb.delete()
    say(f"sandbox deleted; {time.time() - t0:.0f} s in all; output in {out.relative_to(ROOT)}")
