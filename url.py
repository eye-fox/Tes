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
    
    run(f"subfinder -d {domain} -silent -o subfinder.txt")
    run(f"assetfinder --subs-only {domain} -o assetfinder.txt")
    run("cat subfinder.txt assetfinder.txt > unique_subs.txt")
    run("sort -u unique_subs.txt")
    run("httpx -l unique_subs.txt -mc -o alive_subs.txt")
    run("sed -i 's|https\?://||g' alive_subs.txt") 
    run("paramspider -l alive_subs.txt")
    run("cat results/*.txt > urls.txt")
    run("sort urls.txt | uniq > hasil.txt")
    run("httpx -l hasil.txt -mc -follow-redirects -o final_urls.txt")

if __name__ == "__main__":
    main()
