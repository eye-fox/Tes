import subprocess
import sys
import os

def run(cmd, capture=False):
    print(f"[+] {cmd}")
    if capture:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return subprocess.run(cmd, shell=True)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 sql_injection_scanner.py <domain>")
        sys.exit(1)
    
    domain = sys.argv[1]
    
    go_bin = os.path.expanduser("~/go/bin")
    os.environ["PATH"] += os.pathsep + go_bin
    
    run(f"paramspider -d {domain}")
    
    results_file = f"results/{domain}.txt"
    if not os.path.exists(results_file):
        print("[-] ParamSpider failed")
        sys.exit(1)
    
    run(f"httpx -l {results_file} -o urls.txt")
    
    if not os.path.exists("urls.txt") or os.path.getsize("urls.txt") == 0:
        print("[-] No live URLs found")
        with open(f"{domain}.txt", "w") as f:
            f.write("No live URLs found")
        sys.exit(1)
    
    print(f"[+] Running Ghauri on {domain}")
    result = run(f"ghauri -m urls.txt --batch --level=3 --technique=BEUSTQ --threads=5 --delay=2 --time-sec=15 --random-agent --flush-session --text-only --banner", capture=True)
    
    with open(f"{domain}.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write(f"Ghauri Scan Results for {domain}\n")
        f.write("="*60 + "\n\n")
        f.write("STDOUT:\n")
        f.write("-"*40 + "\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write("-"*40 + "\n")
        f.write(result.stderr)
    
    print(f"[+] Results saved to {domain}.txt")

if __name__ == "__main__":
    main()
