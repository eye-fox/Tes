#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <domain>"
    exit 1
fi

DOMAIN=$1
WORKDIR="scan_$DOMAIN"
mkdir -p $WORKDIR
cd $WORKDIR

subfinder -d $DOMAIN -silent > subfinder.txt &
assetfinder --subs-only $DOMAIN > assetfinder.txt &
wait

cat subfinder.txt assetfinder.txt | sort -u > all_subs.txt

cat all_subs.txt | httpx -silent -status-code -follow-redirects | grep -E '^\[200\|201\|202\|203\|204\|301\|302\|303\|307\|308\]' | awk '{print $2}' > active_subs_raw.txt

split -l 10 active_subs_raw.txt batch_
for f in batch_*; do mv "$f" "${f}.txt"; done

for file in batch_*.txt; do
    paramspider -l $file &
    count=$(jobs -r | wc -l)
    while [ $count -ge 5 ]; do
        sleep 1
        count=$(jobs -r | wc -l)
    done
done
wait

cat results/*.txt 2>/dev/null | sort -u > all_urls.txt

cat all_urls.txt | httpx -silent -status-code -follow-redirects | grep -E '^\[200\|201\|202\|203\|204\|301\|302\|303\|307\|308\]' | awk '{print $2}' > final_urls.txt

echo "Done. Results: $WORKDIR/final_urls.txt"
