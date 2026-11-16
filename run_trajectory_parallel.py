import json
import os 
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from time import perf_counter

MAX_WORKERS = 4

def run_pipeline(folder, candidate):
    pipeline_id = Path(candidate["file"]).stem
    start = perf_counter()
    env = {**os.environ, "OMP_NUM_THREADS" : "1"}
    process = subprocess.run(
        [sys.executable,  candidate["file"]], cwd=folder,
        stdout=subprocess.PIPE, text=True, env=env
    )
    
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    
    score = None
    for line in process.stdout.splitlines():
        if line.startswith("Final Validation Score: "):
            score = float(line.split(":",1)[1])
    if score is None:
            raise RuntimeError(f"{pipeline_id} did not report a final validation score")
    return {"pipeline_id" : pipeline_id, "score" : score, "seconds" : perf_counter() - start, "output" : process.stdout}
    
    
def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(__file__).name} <competition folder>")
    folder = Path(sys.argv[1]).resolve()
    output_dir = folder / "results"
    manifest = json.loads((folder / "trajectory.json").read_text())
    results = []
    started = perf_counter()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for step in manifest["steps"]:
            print(f"Step {step["step"]} : {step["name"]}", flush=True)
            
            for result in pool.map(lambda c : run_pipeline(folder,c), step["pipelines"]):
                print(result.pop("output"), end="", flush=True)
                results.append(result)
    
    real_time = perf_counter() - started
    print("-" * 60)
    results.sort(key= lambda result: - result["score"])
    previous_score, rank = None, 0
    for position, result in enumerate(results, start=1):
        if result["score"] != previous_score:
            rank = position
        result["rank"] = rank
        previous_score = result["score"]
        print(f"{rank:2d}  {result['pipeline_id']}  {result["score"]:.6f}")
    
    print(f"\n Total real time: {real_time} with {MAX_WORKERS} workers.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/ "summary.json").write_text(json.dumps(results, indent=2))
    (output_dir / "summary.wall").write_text(f"{real_time:.3f}")
if __name__ == "__main__":
    main()        
            
                