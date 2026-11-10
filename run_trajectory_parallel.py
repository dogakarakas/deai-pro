import json
import os 
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from time import perf_counter

MAX_WORKERS = 4

def run_pipeline(root, candidate):
    pipeline_id = Path(candidate["file"]).stem
    start = perf_counter()
    env = {**os.environ, "OMP_NUM_THREADS" : "1"}
    process = subprocess.run(
        [sys.executable, str(root / candidate["file"])],
        stdout=subprocess.PIPE, text=True, env=env,
    )
    
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    
    score = None
    for line in process.stdout.splitlines():
        if line.startswith("Final Validation Score: "):
            score = float(line.split(":",1)[1])
    if score is None:
            raise RuntimeError("f{pipeline_id} did not report a final validation score")
    return {"pipeline_id" : pipeline_id, "cv_accuracy" : score, "seconds" : perf_counter() - start, "output" : process.stdout}
    
    
def main():
    root = Path(__file__).resolve().parent
    output_dir = Path("results")
    manifest = json.loads((root / "trajectory.json").read_text())
    results = []
    started = perf_counter()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for step in manifest["steps"]:
            print(f"Step {step["step"]} : {step["name"]}", flush=True)
            
            for result in pool.map(lambda c : run_pipeline(root,c), step["pipelines"]):
                print(result.pop("output"), end="", flush=True)
                results.append(result)
    
    real_time = perf_counter() - started
    print("-" * 60)
    results.sort(key= lambda result: - result["cv_accuracy"])
    previous_score, rank = None, 0
    for position, result in enumerate(results, start=1):
        if result["cv_accuracy"] != previous_score:
            rank = position
        result["rank"] = rank
        previous_score = result["cv_accuracy"]
        print(f"{rank:2d}  {result['pipeline_id']}  {result['cv_accuracy']:.6f}")
    
    print(f"\n Total real time: {real_time} with {MAX_WORKERS} workers.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/ "summary.json").write_text(json.dumps(results, indent=2))
    
if __name__ == "__main__":
    main()        
            
                