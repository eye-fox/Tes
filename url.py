import subprocess
import sys
import os
from pathlib import Path

def run(cmd, out=None, check=True):
    """Menjalankan perintah shell dengan opsi redirect output."""
    if out:
        with open(out, 'w') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.PIPE, check=check)
    else:
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <domain>")
        sys.exit(1)
    domain = sys.argv[1]

    # Tambahkan ~/go/bin ke PATH untuk menemukan binari Go
    go_bin = os.path.expanduser("~/go/bin")
    os.environ["PATH"] += os.pathsep + go_bin

    # 1. Subdomain enumeration
    print("[+] Running subfinder...")
    run(f"subfinder -d {domain} -silent -o subfinder.txt")
    
    print("[+] Running assetfinder...")
    run(f"assetfinder --subs-only {domain} -o assetfinder.txt")
    
    # 2. Gabungkan hasil
    print("[+] Merging subdomain lists...")
    run("cat subfinder.txt assetfinder.txt > unique_subs.txt")
    
    # 3. Sort dan hapus duplikat (in-place)
    print("[+] Sorting unique subdomains...")
    run("sort -u unique_subs.txt -o unique_subs.txt")
    
    # 4. Cek subdomain hidup dengan httpx (tanpa -mc karena tidak ada nilai)
    print("[+] Checking alive subdomains with httpx...")
    run("httpx -l unique_subs.txt -o alive_subs.txt")
    
    # 5. Hapus protokol http/https dari file
    print("[+] Removing protocol from alive_subs.txt...")
    run("sed -i 's|https\\?://||g' alive_subs.txt")
    
    # 6. Jalankan paramspider untuk mengumpulkan parameter URL
    print("[+] Running paramspider...")
    run("paramspider -l alive_subs.txt")
    
    # 7. Kumpulkan semua hasil paramspider
    print("[+] Collecting all URLs...")
    run("cat results/*.txt > urls.txt")
    
    # 8. Sort dan hapus duplikat URL
    print("[+] Sorting and deduplicating URLs...")
    run("sort -u urls.txt -o hasil.txt")
    
    # 9. Final probe dengan httpx (follow redirects)
    print("[+] Final httpx scan with follow-redirects...")
    run("httpx -l hasil.txt -follow-redirects -o final_urls.txt")
    
    print("[+] Done! Results saved in final_urls.txt")

if __name__ == "__main__":
    main()
