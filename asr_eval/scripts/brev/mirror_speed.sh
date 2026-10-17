#!/usr/bin/env bash
# Instance side: download speed of each OpenSLR mirror (first 100 MB of musan.tar.gz, 15 s cap each).
for host in https://openslr.elda.org https://us.openslr.org https://openslr.magicdatatech.com https://www.openslr.org; do
  s=$(curl -s -r 0-100000000 -o /dev/null -m 15 -w "%{speed_download} %{http_code}" "$host/resources/17/musan.tar.gz")
  echo "$host  $(echo "$s" | awk '{printf "%.1f MB/s  http %s", $1/1e6, $2}')"
done
