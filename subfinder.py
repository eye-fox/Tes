import subprocess
import sys
import os

def run(cmd, out=None, check=False):
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
    
    go_bin = os.path.expanduser("~/go/bin")
    os.environ["PATH"] += os.pathsep + go_bin
    
    run(f"subfinder -d {domain} -silent -o subfinder.txt", check=False)
    run(f"assetfinder --subs-only {domain} -o assetfinder.txt", check=False)
    
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
    
    with open("unique_subs.txt", 'w') as f:
        for sub in sorted(subdomains):
            f.write(sub + '\n')
    
    print(f"[+] Found {len(subdomains)} unique subdomains")
    
    run("httpx -l unique_subs.txt -o alive_subs.txt", check=False)
    
    if os.path.exists("alive_subs.txt") and os.path.getsize("alive_subs.txt") > 0:
        print("[+] Removing protocol from alive_subs.txt...")
        cleaned_subs = []
        with open("alive_subs.txt", 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace("https://", "").replace("http://", "")
                if line:
                    cleaned_subs.append(line)
        
        with open("alive_subs.txt", 'w') as f:
            f.write('\n'.join(cleaned_subs) + '\n')
        
        print(f"[+] Found {len(cleaned_subs)} alive subdomains")
        print("[+] Done! Results saved in alive_subs.txt")
    else:
        print("[-] No alive subdomains found")

if __name__ == "__main__":
    main()
