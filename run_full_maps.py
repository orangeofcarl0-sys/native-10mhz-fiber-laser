"""Resume-safe reproduction of 640 cases on each of two grids (600 RT ceiling).

Set LASER_OUTPUT_DIR to a fresh run directory and MAP_GPU=1 before execution.
Numerical boundary stops remain unresolved; they are not silently called stable.
"""

from datetime import datetime, timezone
from pathlib import Path
import gc, hashlib, json, os, subprocess, time
from scipy.io import whosmat
from config import ROOT, PROJECT
from scan import scan


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8"
    )
    temporary.replace(path)


def run():
    files = [
        "scan.py",
        "gpu_engine.py",
        "spectral_engine.py",
        "gpu_kernels.py",
        "config.py",
        "parameters/effective_parameters.json",
    ]
    hashes = {
        name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        for name in files
    }
    path = ROOT / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf8"))
        if manifest["code_sha256"] != hashes:
            raise RuntimeError(
                "Code changed since this run started; use a new output directory"
            )
        if manifest["status"] == "complete":
            if len(manifest["completed_groups"]) != 20:
                raise RuntimeError("Invalid completed manifest")
            print(
                "Run already complete; original completion timestamp retained",
                flush=True,
            )
            return
    else:
        manifest = dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            code_sha256=hashes,
            backend=(
                "GPU spectral FP64" if os.environ.get("MAP_GPU") == "1" else "CPU FP64"
            ),
            case_count_per_grid=640,
            grids=[
                dict(tag="", dt_ps=0.5, n=2048),
                dict(tag="fine_", dt_ps=0.125, n=8192),
            ],
            round_limit=600,
            completed_groups=[],
            status="running",
            scope="Same paired initialization and fixed-grid stop criteria as previous map; not full-cavity certification",
        )
        save(path, manifest)
    started = time.perf_counter()
    for tag, dt, n in [("", 0.5, 2048), ("fine_", 0.125, 8192)]:
        for group in range(10):
            key = f"{tag}group_{group:02d}"
            output = ROOT / f"{key}.json"
            if key in manifest["completed_groups"]:
                rows = json.loads(output.read_text(encoding="utf8"))
                assert len(rows) == 64 and (ROOT / f"{key}.mat").is_file()
                continue
            manifest.update(status="running", current_group=key)
            save(path, manifest)
            scan(group, rounds=600, dt=dt, n=n, tag=tag)
            rows = json.loads(output.read_text(encoding="utf8"))
            assert len(rows) == 64 and all(0 < r["rounds"] <= 600 for r in rows)
            variables = dict(
                (name, shape) for name, shape, _ in whosmat(ROOT / f"{key}.mat")
            )
            assert variables["a"] == (64, 2, n) and variables["trace"] == (600, 64, 12)
            manifest["completed_groups"].append(key)
            manifest["last_update_utc"] = datetime.now(timezone.utc).isoformat()
            save(path, manifest)
            gc.collect()
            if os.environ.get("MAP_GPU") == "1":
                from gpu_setup import cp

                cp.get_default_memory_pool().free_all_blocks()
    manifest.update(
        status="complete",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        last_invocation_seconds=time.perf_counter() - started,
    )
    save(path, manifest)
    print("FULL RUN COMPLETE: 20 groups / 1280 case-grid trajectories", flush=True)


if __name__ == "__main__":
    run()
