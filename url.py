import subprocess
import sys
import os
import glob
from pathlib import Path

def run(cmd, out=None, check=False):
    """Menjalankan perintah shell eksternal (hanya untuk tools)."""
    print(f"[+] Running: {cmd}")
    if out:
        with open(out, 'w') as f:
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.PIPE, text=True)
    else:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if check and result.returncode != 0:
        print(f"[-] Warning: Command failed with exit code {result.returncode}")
    return result.returncode

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <domain>")
        sys.exit(1)
    domain = sys.argv[1]
    
    # Tambahkan ~/go/bin ke PATH
    go_bin = os.path.expanduser("~/go/bin")
    os.environ["PATH"] += os.pathsep + go_bin
    
    # 1. Subdomain enumeration
    run(f"subfinder -d {domain} -silent -o subfinder.txt", check=False)
    run(f"assetfinder --subs-only {domain} -o assetfinder.txt", check=False)
    
    # 2. Gabungkan dan unique kan subdomain dengan Python native
    print("[+] Merging and sorting unique subdomains...")
    subdomains = set()
    
    for file in ["subfinder.txt", "assetfinder.txt"]:
        if os.path.exists(file) and os.path.getsize(file) > 0:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        subdomains.add(line)
        else:
            print(f"[-] Warning: {file} is empty or doesn't exist")
    
    if not subdomains:
        print("[-] No subdomains found! Exiting...")
        sys.exit(1)
    
    # Write unique subdomains
    with open("unique_subs.txt", 'w') as f:
        for sub in sorted(subdomains):
            f.write(sub + '\n')
    
    print(f"[+] Found {len(subdomains)} unique subdomains")
    
    # 3. Cek subdomain hidup dengan httpx
    run("httpx -l unique_subs.txt -o alive_subs.txt", check=False)
    
    # 4. Hapus protokol dari alive_subs.txt (Python native)
    if os.path.exists("alive_subs.txt") and os.path.getsize("alive_subs.txt") > 0:
        print("[+] Removing protocol from alive_subs.txt...")
        cleaned_subs = []
        with open("alive_subs.txt", 'r') as f:
            for line in f:
                line = line.strip()
                # Hapus http://, https://, http://, https://
                line = line.replace("https://", "").replace("http://", "")
                if line:
                    cleaned_subs.append(line)
        
        with open("alive_subs.txt", 'w') as f:
            f.write('\n'.join(cleaned_subs) + '\n')
    else:
        print("[-] Warning: alive_subs.txt is empty, skipping protocol removal")
    
    # 5. Jalankan paramspider
    if os.path.exists("alive_subs.txt") and os.path.getsize("alive_subs.txt") > 0:
        run("paramspider -l alive_subs.txt", check=False)
    else:
        print("[-] Skipping paramspider: no alive subdomains found")
    
    # 6. Kumpulkan hasil paramspider dengan Python native
    print("[+] Collecting all URLs from paramspider...")
    all_urls = set()
    
    results_dir = Path("results")
    if results_dir.exists():
        for txt_file in results_dir.glob("*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip comments
                        all_urls.add(line)
    
    if not all_urls:
        print("[-] No URLs found from paramspider, using alive_subs.txt as fallback")
        if os.path.exists("alive_subs.txt"):
            with open("alive_subs.txt", 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_urls.add(f"http://{line}")
                        all_urls.add(f"https://{line}")
    
    # 7. Write hasil.txt
    with open("hasil.txt", 'w') as f:
        for url in sorted(all_urls):
            f.write(url + '\n')
    
    print(f"[+] Collected {len(all_urls)} unique URLs")
    
    # 8. Final httpx scan
    if all_urls:
        run("httpx -l hasil.txt -follow-redirects -o final_urls.txt", check=False)
        print("[+] Done! Results saved in final_urls.txt")
    else:
        print("[-] No URLs to scan, skipping final httpx")

if __name__ == "__main__":
    main()
