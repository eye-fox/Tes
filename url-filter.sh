#!/bin/bash
# url-filter.sh - Filter URLs with parameters and deduplicate

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found"
    exit 1
fi

TEMP_FILE="${OUTPUT_FILE}.tmp"

grep -E '\?[^=]+=' "$INPUT_FILE" | \
grep -vE '\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|mp4|mp3|webm)$' | \
sed -E 's/#.*$//' | \
sed -E 's/\/$//' | \
awk -F'?' '{print $1"?"$2}' | \
sort -u > "$TEMP_FILE"

declare -A seen
while IFS= read -r url; do
    hostname=$(echo "$url" | awk -F/ '{print $3}')
    path_and_params=$(echo "$url" | cut -d'?' -f1 | cut -d'/' -f4-)
    params=$(echo "$url" | cut -d'?' -f2 | sed 's/[^=&]*=//g' | tr '&' '\n' | sort | tr '\n' '&')
    key="${hostname}|${path_and_params}|${params}"
    
    if [ -z "${seen[$key]}" ]; then
        seen[$key]=1
        echo "$url"
    fi
done < "$TEMP_FILE" > "$OUTPUT_FILE"

rm -f "$TEMP_FILE"

if [ ! -s "$OUTPUT_FILE" ]; then
    echo "No URLs with valid parameters found"
fi

exit 0
