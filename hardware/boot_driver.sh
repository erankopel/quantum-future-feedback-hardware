#!/bin/bash
# Collect gauge-invariant bootstrap samples: 2136 to 8, then 2122 to 8.
cd /home/claude/paper7work
count() { python3 -c "import json;print(len(json.load(open('gst_bootgi_$1.json'))['boots']))" 2>/dev/null || echo 0; }
while [ "$(count 2136)" -lt 8 ]; do python3 gst_boot_gi.py 2136 1 >> bootgi_2136.log 2>&1; done
while [ "$(count 2122)" -lt 8 ]; do python3 gst_boot_gi.py 2122 1 >> bootgi_2122.log 2>&1; done
touch BOOT_DONE
