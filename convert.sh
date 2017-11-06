#!/bin/bash
for f in ./data/Texts/*.opin.*; do
    echo $f
    iconv -f "windows-1251" -t "UTF-8" $f | sponge $f
done
