# ── EsaSim block ──────────────────────────────────────────────────────
# GPU PyTorch check (grasping) (ws-gpu-pytorch-check-grasping) · version 3 · worker block, Python: def run(ctx)
# Checks which CUDA build of PyTorch runs on this worker's NVIDIA GPU (for the conveyor grasping block).
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-19
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   (none)
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   spec        text      Torch spec = "torch==2.14.1"
#   cu          text      CUDA index = "cu126"
#   install     select    Install = "No" of No | Yes | Remove
#   hold        number    Then wait (minutes) = 0
# Runs on  NVIDIA GPU. Return a dict to fill the "result" output.
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

import os
import shutil
import subprocess
import sys
import time


def run(ctx):
    here = os.path.abspath(os.getcwd())
    jobs = os.path.dirname(os.path.dirname(here))
    base = os.path.dirname(jobs) if os.path.basename(jobs).lower() == "jobs" else os.path.dirname(here)
    target = os.path.join(base, "cache", "torch-cuda")
    mode = str(ctx.params.get("install")).lower()
    if mode.startswith("r"):
        shutil.rmtree(target, ignore_errors=True)
        ctx.log("Removed %s: %s" % (target, not os.path.exists(target)))
    else:
        ctx.log(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:1500])
        if mode.startswith("y"):
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "--target", target, ctx.params.get("spec") or "torch",
                                "--index-url", "https://download.pytorch.org/whl/%s" % (ctx.params.get("cu") or "cu126")], capture_output=True, text=True)
            ctx.log("pip: %s %s" % (r.returncode, (r.stdout + r.stderr)[-800:]))
        probe = ("import torch; print(torch.__version__, torch.__file__); print(torch.cuda.is_available(), torch.version.cuda); "
                 "print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); "
                 "x = torch.ones(2, device='cuda') * 2; print(float(x.sum()))")
        env = dict(os.environ)
        env["PYTHONPATH"] = target + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
        ctx.log("probe rc %s\n%s\n%s" % (r.returncode, r.stdout[-1500:], r.stderr[-2500:]))
    hold = float(ctx.params.get("hold") or 0)
    t0 = time.time()
    while time.time() - t0 < hold * 60:
        if any(isinstance(m, dict) for m in ctx.messages("control")):
            break
        time.sleep(5)
    return {"ok": True}
