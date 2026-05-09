#!/bin/bash

DOMAIN=$1

if [ -z "$DOMAIN" ]; then
    exit 1
fi

mkdir -p temp results

subfinder -d $DOMAIN -silent > temp/subfinder.txt 2>/dev/null &
assetfinder --subs-only $DOMAIN > temp/assetfinder.txt 2>/dev/null &
wait

cat temp/subfinder.txt temp/assetfinder.txt 2>/dev/null | sort -u > temp/subs_sorted.txt
echo "$DOMAIN" > temp/all_domains.txt
cat temp/subs_sorted.txt >> temp/all_domains.txt
sort -u -o temp/all_domains.txt temp/all_domains.txt

cat temp/all_domains.txt | httpx -silent -threads 100 -status-code -content-length -title -o temp/live_subs.txt
cut -d' ' -f1 temp/live_subs.txt > temp/live_domains.txt

LIVE_COUNT=$(wc -l < temp/live_domains.txt 2>/dev/null)
if [ "$LIVE_COUNT" -eq 0 ]; then
    rm -rf temp
    exit 0
fi

extract_gau_parallel() {
    cat temp/live_domains.txt | xargs -P 20 -I {} sh -c "gau --subs {} >> temp/gau_urls.txt 2>/dev/null"
}

extract_wayback_parallel() {
    cat temp/live_domains.txt | xargs -P 20 -I {} sh -c "echo {} | waybackurls >> temp/wayback_urls.txt 2>/dev/null"
}

> temp/gau_urls.txt
> temp/wayback_urls.txt

extract_gau_parallel &
EXTRACT_PID=$!
extract_wayback_parallel &
WAYBACK_PID=$!

wait $EXTRACT_PID
wait $WAYBACK_PID

cat temp/gau_urls.txt temp/wayback_urls.txt 2>/dev/null | sort -u > temp/urls_raw.txt

grep -E '\?[^=]+=' temp/urls_raw.txt | \
grep -vE '\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|mp4|mp3|webm)$' | \
sed -E 's/#.*$//' | \
sed -E 's/\/$//' | \
awk -F'?' '{print $1"?"$2}' | \
sort -u > temp/param_urls.txt

PARAM_COUNT=$(wc -l < temp/param_urls.txt 2>/dev/null)
if [ "$PARAM_COUNT" -gt 0 ]; then
    ghauri -m temp/param_urls.txt \
        --batch \
        --level=3 \
        --technique=BEUSTQ \
        --threads=10 \
        --delay=1 \
        --time-sec=15 \
        --random-agent \
        --flush-session \
    cd ..
fi

rm -rf temp
