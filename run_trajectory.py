"""Simple sequential reference runner; scheduling optimizations are the exercise.

Usage: python run_trajectory.py <competition folder>, e.g. spaceship_titanic_students
"""

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(__file__).name} <competition folder>")
    folder = Path(sys.argv[1]).resolve()
    manifest = json.loads((folder / "trajectory.json").read_text())
    maximize = manifest["evaluation"]["maximize"]
    results = []
    started_wall = perf_counter()
    for step in manifest["steps"]:
        print(f"Step {step['step']}: {step['name']}", flush=True)
        for candidate in step["pipelines"]:
            pipeline_id = Path(candidate["file"]).stem
            started = perf_counter()
            score = None
            # Pipelines read ./input/ relative to their competition folder.
            with subprocess.Popen(
                [sys.executable, candidate["file"]],
                cwd=folder, stdout=subprocess.PIPE, text=True,
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    if line.startswith("Final Validation Score: "):
                        score = float(line.split(":", 1)[1])
                returncode = process.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, process.args)
            if score is None:
                raise RuntimeError(f"{pipeline_id} did not report a final validation score")
            results.append({"pipeline_id": pipeline_id,
                            "score": score,
                            "seconds": perf_counter() - started})
    # Stable display order for ties; equal scores still share rank.
    real_time = perf_counter() - started_wall
    print("-"*60)
    print(f"Ranking by {manifest['evaluation']['metric']} "
          f"({'higher' if maximize else 'lower'} is better)")
    results.sort(key=lambda result: -result["score"] if maximize else result["score"])
    previous_score, rank = None, 0
    for position, result in enumerate(results, start=1):
        if result["score"] != previous_score:
            rank = position
        result["rank"] = rank
        previous_score = result["score"]
        print(f"{rank:2d}  {result['pipeline_id']}  {result['score']:.6f}")
    output_dir = folder / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
