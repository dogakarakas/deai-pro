"""Simple sequential reference runner; scheduling optimizations are the exercise."""

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    root = Path(__file__).resolve().parent
    output_dir = Path("results")
    manifest = json.loads((root / "trajectory.json").read_text())
    results = []
    started = perf_counter()
    for step in manifest["steps"]:
        print(f"Step {step['step']}: {step['name']}", flush=True)
        started_wall = perf_counter()
        for candidate in step["pipelines"]:
            pipeline_id = Path(candidate["file"]).stem             #deletes .py
            started = perf_counter()
            score = None
            with subprocess.Popen(                                 #starting the pipeline as a separate program
                [sys.executable, str(root / candidate["file"])],
                stdout=subprocess.PIPE, text=True,
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    if line.startswith("Final Validation Score: "):
                        score = float(line.split(":", 1)[1])
                returncode = process.wait()                       #waits till termination of the subprocess returns the returncode
                if returncode:
                    raise subprocess.CalledProcessError(returncode, process.args)
            if score is None:
                raise RuntimeError(f"{pipeline_id} did not report a final validation score")
            results.append({"pipeline_id": pipeline_id,
                            "cv_accuracy": score,
                            "seconds": perf_counter() - started})
    real_time = perf_counter() - started_wall
    # Stable display order for ties; equal scores still share rank.
    print("-"*60)
    results.sort(key=lambda result: -result["cv_accuracy"])
    previous_score, rank = None, 0
    for position, result in enumerate(results, start=1):
        if result["cv_accuracy"] != previous_score:
            rank = position
        result["rank"] = rank
        previous_score = result["cv_accuracy"]
        print(f"{rank:2d}  {result['pipeline_id']}  {result['cv_accuracy']:.6f}")
    print(f"\n Total real time: {real_time} with sequential processing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
