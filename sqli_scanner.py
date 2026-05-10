import subprocess
import sys
import os
import re
import time
import json
from pathlib import Path

def run_command(cmd, capture_output=False, check=False):
    print(f"[+] Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"[-] Warning: Command failed with exit code {result.returncode}")
            print(f"[-] stderr: {result.stderr[:500]}")
        return result
    else:
        result = subprocess.run(cmd, shell=True, text=True)
        if check and result.returncode != 0:
            print(f"[-] Warning: Command failed with exit code {result.returncode}")
        return result

def check_dependencies():
    print("[*] Checking dependencies...")
    go_bin = os.path.expanduser("~/go/bin")
    os.environ["PATH"] += os.pathsep + go_bin
    
    deps = {
        "paramspider": "paramspider --help",
        "httpx": "httpx -version",
        "ghauri": "ghauri --version"
    }
    
    missing = []
    for dep, cmd in deps.items():
        result = run_command(cmd, capture_output=True)
        if result.returncode != 0:
            missing.append(dep)
        else:
            print(f"[+] {dep} found")
    
    if missing:
        print(f"[-] Missing dependencies: {', '.join(missing)}")
        print("[*] Install with:")
        print("  git clone https://github.com/devanshbatham/ParamSpider")
        print("  cd ParamSpider && pip install .")
        print("  go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest")
        print("  git clone https://github.com/r0oth3x49/ghauri.git")
        print("  cd ghauri && python3 setup.py install")
        sys.exit(1)

def run_paramspider(domain):
    print(f"\n[*] Running ParamSpider on {domain}")
    cmd = f"paramspider -d {domain}"
    result = run_command(cmd, capture_output=True)
    
    results_file = f"results/{domain}.txt"
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        print(f"[+] ParamSpider found parameters: {results_file}")
        return results_file
    else:
        print(f"[-] ParamSpider returned no results")
        return None

def run_httpx(input_file, output_file):
    print(f"\n[*] Running httpx on {input_file}")
    cmd = f"httpx -l {input_file} -silent -no-color -threads 100 -rate-limit 500 -timeout 5 -retries 1 -follow-host-redirects -mc 200,201,202,203,204,301,302,307,308,401,403 -random-agent -o {output_file}"
    run_command(cmd)
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"[+] httpx found {count_lines(output_file)} live URLs")
        return True
    else:
        print(f"[-] httpx found no live URLs")
        return False

def clean_urls(input_file, output_file):
    print(f"\n[*] Cleaning URLs from {input_file}")
    cleaned = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                line = line.replace("https://", "").replace("http://", "")
                line = line.rstrip('/')
                if line:
                    cleaned.append(line)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(cleaned) + '\n')
    
    print(f"[+] Cleaned {len(cleaned)} URLs")
    return cleaned

def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

def is_url_vulnerable(output_text):
    output_lower = output_text.lower()
    indicators = [
        'vulnerable',
        'injection found',
        'parameter.*vulnerable',
        'identified',
        'payload',
        'true'
    ]
    
    for indicator in indicators:
        if re.search(indicator, output_lower):
            return True
    return False

def test_url_with_ghauri(url, index, total):
    print(f"\n[*] Testing URL {index}/{total}: {url[:80]}...")
    
    cmd = (
        f"ghauri -u 'http://{url}' "
        f"--batch "
        f"--level=3 "
        f"--technique=BEUSTQ "
        f"--threads=5 "
        f"--delay=2 "
        f"--time-sec=15 "
        f"--random-agent "
        f"--flush-session "
        f"--text-only "
        f"--banner"
    )
    
    result = run_command(cmd, capture_output=True)
    output = result.stdout + result.stderr
    
    if is_url_vulnerable(output):
        print(f"[!] VULNERABLE: {url}")
        return True, output
    else:
        print(f"[-] Not vulnerable: {url}")
        return False, output

def run_ghauri_bulk(urls_file):
    print(f"\n[*] Running Ghauri on URLs from {urls_file}")
    
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("[-] No URLs to test")
        return [], {}
    
    print(f"[+] Testing {len(urls)} URLs")
    vulnerable_urls = []
    all_outputs = {}
    
    for idx, url in enumerate(urls, 1):
        is_vuln, output = test_url_with_ghauri(url, idx, len(urls))
        if is_vuln:
            vulnerable_urls.append(url)
        all_outputs[url] = output[:1000]
        time.sleep(1)
    
    return vulnerable_urls, all_outputs

def save_results(vulnerable_urls, all_outputs, domain):
    print(f"\n[*] Saving results...")
    
    with open("vulnerable_urls.txt", 'w') as f:
        if vulnerable_urls:
            f.write('\n'.join(vulnerable_urls) + '\n')
            print(f"[+] Found {len(vulnerable_urls)} vulnerable URLs")
        else:
            print("[+] No vulnerable URLs found")
    
    with open("ghauri_full_output.txt", 'w') as f:
        f.write(f"Domain: {domain}\n")
        f.write(f"Total URLs tested: {len(all_outputs)}\n")
        f.write(f"Vulnerable URLs found: {len(vulnerable_urls)}\n\n")
        f.write("="*60 + "\n\n")
        
        for url, output in all_outputs.items():
            f.write(f"URL: {url}\n")
            f.write(f"Vulnerable: {'YES' if url in vulnerable_urls else 'NO'}\n")
            f.write(f"Output:\n{output}\n")
            f.write("-"*40 + "\n\n")
    
    summary = {
        "domain": domain,
        "total_urls_tested": len(all_outputs),
        "vulnerable_urls_found": len(vulnerable_urls),
        "vulnerable_urls_list": vulnerable_urls,
        "scan_status": "completed"
    }
    
    with open("scan_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[+] Results saved to: vulnerable_urls.txt, ghauri_full_output.txt, scan_summary.json")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 sql_injection_scanner.py <domain>")
        print("Example: python3 sql_injection_scanner.py example.com")
        sys.exit(1)
    
    domain = sys.argv[1].lower().strip()
    domain = domain.replace("https://", "").replace("http://", "").replace("/", "")
    
    print(f"\n{'='*60}")
    print(f"SQL Injection Scanner - Target: {domain}")
    print(f"{'='*60}\n")
    
    check_dependencies()
    
    paramspider_output = run_paramspider(domain)
    if not paramspider_output:
        print("[-] ParamSpider failed to find parameters. Exiting...")
        save_results([], {}, domain)
        sys.exit(1)
    
    httpx_output = "live_urls.txt"
    if not run_httpx(paramspider_output, httpx_output):
        print("[-] No live URLs found. Exiting...")
        save_results([], {}, domain)
        sys.exit(1)
    
    cleaned_urls_file = "cleaned_urls.txt"
    cleaned_urls = clean_urls(httpx_output, cleaned_urls_file)
    
    if not cleaned_urls:
        print("[-] No URLs after cleaning. Exiting...")
        save_results([], {}, domain)
        sys.exit(1)
    
    vulnerable_urls, all_outputs = run_ghauri_bulk(cleaned_urls_file)
    
    save_results(vulnerable_urls, all_outputs, domain)
    
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETED")
    print(f"Domain: {domain}")
    print(f"URLs tested: {len(all_outputs)}")
    print(f"Vulnerable URLs: {len(vulnerable_urls)}")
    if vulnerable_urls:
        print(f"\nVulnerable URLs:")
        for url in vulnerable_urls:
            print(f"  - {url}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
