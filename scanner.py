#!/usr/bin/env python3

import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import shutil
import re

def run_command(cmd, timeout=300, check=False):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", -1

def run_parallel(commands, max_workers=2):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_command, cmd): cmd for cmd in commands}
        for future in as_completed(futures):
            results.append(future.result())
    return results

def stage_subdomain_enum(domain):
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    cmds = [
        f"subfinder -d {domain} -silent > temp/subfinder.txt 2>/dev/null",
        f"assetfinder --subs-only {domain} > temp/assetfinder.txt 2>/dev/null"
    ]
    
    run_parallel(cmds, max_workers=2)
    
    subfinder_domains = set()
    assetfinder_domains = set()
    
    if Path("temp/subfinder.txt").exists():
        with open("temp/subfinder.txt") as f:
            subfinder_domains = set(line.strip() for line in f if line.strip())
    
    if Path("temp/assetfinder.txt").exists():
        with open("temp/assetfinder.txt") as f:
            assetfinder_domains = set(line.strip() for line in f if line.strip())
    
    all_domains = subfinder_domains | assetfinder_domains
    all_domains.add(domain)
    
    with open("temp/all_domains.txt", "w") as f:
        for d in sorted(all_domains):
            f.write(f"{d}\n")
    
    return len(all_domains)

def stage_probe_live():
    cmd = "cat temp/all_domains.txt | httpx -silent -threads 100 -status-code -content-length -title -o temp/live_subs.txt"
    run_command(cmd)
    
    if Path("temp/live_subs.txt").exists():
        with open("temp/live_subs.txt") as infile, open("temp/live_domains.txt", "w") as outfile:
            for line in infile:
                parts = line.split()
                if parts:
                    outfile.write(f"{parts[0]}\n")
    
    live_count = 0
    if Path("temp/live_domains.txt").exists():
        with open("temp/live_domains.txt") as f:
            live_count = sum(1 for _ in f)
    
    return live_count

def extract_urls_gau(domain_file):
    with open(domain_file) as f:
        domains = [line.strip() for line in f if line.strip()]
    
    output_file = "temp/gau_urls.txt"
    Path(output_file).touch()
    
    for domain in domains:
        cmd = f"gau --subs {domain} >> {output_file} 2>/dev/null"
        run_command(cmd, timeout=120)
    
    count = 0
    if Path(output_file).exists():
        with open(output_file) as f:
            count = sum(1 for _ in f)
    return count

def extract_urls_wayback(domain_file):
    with open(domain_file) as f:
        domains = [line.strip() for line in f if line.strip()]
    
    output_file = "temp/wayback_urls.txt"
    Path(output_file).touch()
    
    for domain in domains:
        cmd = f"echo {domain} | waybackurls >> {output_file} 2>/dev/null"
        run_command(cmd, timeout=120)
    
    count = 0
    if Path(output_file).exists():
        with open(output_file) as f:
            count = sum(1 for _ in f)
    return count

def stage_url_extraction():
    domain_file = "temp/live_domains.txt"
    if not Path(domain_file).exists():
        return 0
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_gau = executor.submit(extract_urls_gau, domain_file)
        future_wayback = executor.submit(extract_urls_wayback, domain_file)
        gau_count = future_gau.result()
        wayback_count = future_wayback.result()
    
    url_files = ["temp/gau_urls.txt", "temp/wayback_urls.txt"]
    all_urls = set()
    for f in url_files:
        if Path(f).exists():
            with open(f) as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        all_urls.add(line)
    
    with open("temp/urls_raw.txt", "w") as f:
        for url in sorted(all_urls):
            f.write(f"{url}\n")
    
    return len(all_urls)

def filter_urls_with_params():
    input_file = "temp/urls_raw.txt"
    output_file = "temp/param_urls.txt"
    
    if not Path(input_file).exists():
        return 0
    
    static_extensions = re.compile(r'\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|mp4|mp3|webm)(\?|$)', re.IGNORECASE)
    
    seen = {}
    filtered_urls = []
    
    with open(input_file) as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            
            if not re.search(r'\?[^=]+=', url):
                continue
            
            if static_extensions.search(url):
                continue
            
            url = re.sub(r'#.*$', '', url)
            url = re.sub(r'\/$', '', url)
            
            try:
                parts = url.split('?')
                if len(parts) != 2:
                    continue
                
                base = parts[0]
                params = parts[1]
                
                host_match = re.search(r'https?://([^/]+)', base)
                if not host_match:
                    continue
                hostname = host_match.group(1)
                
                path_parts = base.split('/')
                path_and_params = '/'.join(path_parts[3:]) if len(path_parts) > 3 else ''
                
                param_keys = sorted(re.findall(r'([^=&]+)=', params))
                param_sig = '&'.join(param_keys)
                
                key = f"{hostname}|{path_and_params}|{param_sig}"
                
                if key not in seen:
                    seen[key] = True
                    filtered_urls.append(url)
            except Exception:
                continue
    
    with open(output_file, "w") as f:
        for url in filtered_urls:
            f.write(f"{url}\n")
    
    return len(filtered_urls)

def stage_ghauri_scan():
    param_urls = "temp/param_urls.txt"
    
    if not Path(param_urls).exists():
        return False
    
    with open(param_urls) as f:
        line_count = sum(1 for _ in f)
    
    if line_count == 0:
        return False
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    cmd = f"ghauri -m {param_urls} --batch --level=3 --technique=BEUSTQ --threads=10 --delay=1 --time-sec=15 --random-agent --flush-session --output-dir=./results"
    run_command(cmd, timeout=3600)
    
    return True

def cleanup():
    for d in ["temp", "results"]:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    
    domain = sys.argv[1]
    
    cleanup()
    Path("temp").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    total_domains = stage_subdomain_enum(domain)
    live_count = stage_probe_live()
    
    if live_count == 0:
        cleanup()
        sys.exit(0)
    
    total_urls = stage_url_extraction()
    
    param_count = filter_urls_with_params()
    
    if param_count > 0:
        stage_ghauri_scan()
    
    cleanup()

if __name__ == "__main__":
    main()
