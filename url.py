import subprocess
import sys
import os
import shutil
from pathlib import Path

def run(cmd, out=None, check=False):
    if out:
        with open(out, 'w') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.DEVNULL, check=check)
    else:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    domain = sys.argv[1]
    
    os.environ["PATH"] += os.pathsep + os.path.expanduser("~/go/bin")
    
    run(f"subfinder -d {domain} -silent", "subfinder.txt")
    run(f"assetfinder --subs-only {domain}", "assetfinder.txt")
    
    with open("subfinder.txt") as f:
        s1 = set(f.read().splitlines())
    with open("assetfinder.txt") as f:
        s2 = set(f.read().splitlines())
    all_subs = sorted(s1 | s2 | {domain})
    with open("unique_subs.txt", 'w') as f:
        f.write('\n'.join(all_subs))
    
    run(f"httpx -l {'unique_subs.txt'} -mc -o {'alive_subs.txt'}")
    
    with open("alive_subs.txt") as f:
        raw = f.read().splitlines()
    clean = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        if line.startswith("http://"):
            line = line[7:]
        elif line.startswith("https://"):
            line = line[8:]
        if line.endswith("/"):
            line = line[:-1]
        clean.append(line)
    with open("clean_subs.txt", 'w') as f:
        f.write('\n'.join(clean))
    
    run(f"paramspider -l {'clean_subs.txt'}")
    
    results_dir = Path("results")
    if results_dir.exists():
        with open("all_params.txt", 'w') as out_f:
            for res_file in results_dir.glob("*.txt"):
                with open(res_file) as in_f:
                    out_f.write(in_f.read())
        if ("all_params.txt").stat().st_size > 0:
            with open("all_params.txt") as f:
                lines = sorted(set(f.read().splitlines()))
            with open("unique_params.txt", 'w') as f:
                f.write('\n'.join(lines))
            run(f"httpx -l {'unique_params.txt'} -mc -follow-redirects -o final_urls.txt")
    
    shutil.rmtree("results", ignore_errors=True)

if __name__ == "__main__":
    main()
