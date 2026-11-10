import argparse
import json
import shutil
import sys
from pathlib import Path

RESULTS = Path("results")
RUNS = RESULTS / "runs"
SUMMARY = RESULTS/ "summary.json"

def resolve(ref):
    path = Path(ref)
    if path.exists():
        return path
    named = RUNS / f"{ref}.json"
    if named.exists():
        return named
    sys.exit(f"Cannot find run '{ref}' looked for {path} and {named}")
    
def load(ref):
    return {r["pipeline_id"]: r for r in json.loads(resolve(ref).read_text())}

#Snapshot the current summary
def cmd_save(args):
    if not SUMMARY.exists():
        sys.exit(f"{SUMMARY} not found. Run 'run_trajectory.py' first.")
    RUNS.mkdir(parents=True, exist_ok=True)
    dest = RUNS/ f"{args.name}.json"
    if dest.exists() and not args.force:
        sys.exit(f"{dest} already exists. Use --force to overwrite")
    shutil.copy(SUMMARY, dest)
    print(f"Saved {SUMMARY} -> {dest}")

#Duration of a single run of all pipelines    
def cmd_list(_args):
    runs = sorted(RUNS.glob("*.json")) if RUNS.exists() else []
    if not runs:
        print("No saved runs.")
    
    for run in runs:
        data = json.loads(run.read_text())
        total = sum(pipeline_of_run["seconds"] for pipeline_of_run in data)
        print(f"{run.stem:20s} | Total duration: {total:7.1f}s  ")
        
def cmd_compare(args):
    prev, curr = load(args.previous), load(args.current)
    if set(prev) != set(curr):
        sys.exit(f"Runs contain different pipelines {sorted(set(prev) ^ set(curr))}")
    ids = sorted(prev, key=lambda i: [int(x) for x in i.split("_")[1:]])
    
    print(f"{'pipeline':14s} {'acc before':>11s} {'acc after':>11s} {'delta':>10s}"
          f" {'sec before':>10s} {'sec after':>10s} {'speedup':>8s}")
    print("-" * 80)
    total_prev = total_curr = 0.0
    changed = []
    
    for pid in ids:
        run_prev, run_curr = prev[pid], curr[pid]
        
        acc_diff = run_prev["cv_accuracy"] - run_curr["cv_accuracy"]
        flag = ""
        
        if abs(acc_diff) > args.tol:
            flag = " <- ACCURACY CHANGED"
            changed.append(pid)
            
        speed_up = run_prev["seconds"] / run_curr["seconds"] if run_prev["seconds"] else float("inf")
        total_curr += run_curr["seconds"]
        total_prev += run_prev["seconds"]
        
        print(f"{pid:14s} {run_prev['cv_accuracy']:11.6f} {run_curr['cv_accuracy']:11.6f} {acc_diff:+10.6f}"
              f" {run_prev['seconds']:10.2f} {run_curr['seconds']:10.2f} {speed_up:7.2f}x{flag}")
        
    print("--" * 40)
        
    print(f"{'TOTAL':14s} {'':11s} {'':11s} {'':10s} {total_curr:10.2f} {total_prev:10.2f} {total_prev / total_curr:7.2f}x")
        
    rank_moves = [(p, prev[p]["rank"], curr[p]["rank"]) for p in ids if prev[p]["rank"] != curr[p]["rank"]]
    
    if rank_moves:
        print("\nRank changes: ")
        for p, old_rank, new_rank in rank_moves:
            print(f" {p}: {old_rank} -> {new_rank}")
            
    print("--" * 40)
    
    if changed:
        print(f"ACCURACY CHANGED\n")
        print(f"Number of pipelnes changed: {len(changed)}")
    print(f"All accuracies identical withing {args.tol}")
    

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class= argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    #Subparser for saving the summar.json file for the current run
    p_save = sub.add_parser("save", help="snapshot the current results/summary.json")
    p_save.add_argument("name")
    p_save.add_argument("--force", action="store_true")
    p_save.set_defaults(function=cmd_save)
    
    #Comparing the current run's summary.json file to the baseline
    p_cmp = sub.add_parser("compare", help="compare two runs")
    p_cmp.add_argument("previous")
    p_cmp.add_argument("current", nargs="?", default=str(SUMMARY))
    p_cmp.add_argument("--tol", type=float, default=1e-9, help="accuracy tolerance")
    p_cmp.set_defaults(function=cmd_compare)
    
    p_list = sub.add_parser("list", help="list saved runs")
    p_list.set_defaults(function=cmd_list)
    
    args = parser.parse_args()
    args.function(args)
    
    
if __name__ == "__main__":
    main()