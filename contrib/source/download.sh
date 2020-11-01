#!/bin/bash
data_dir="./data/"
# Downloading RuAttitudes-1.2
curl -L -o "$data_dir/ruattitudes-v1_2.zip" https://github.com/nicolay-r/RuAttitudes/raw/v1.2/ruattitudes_v1_2.zip
# Downloading RuAttitudes-2.0-base
curl -L -o "$data_dir/ruattitudes-v2_0_base.zip" https://www.dropbox.com/s/y39vqzzjumqhce1/ruattitudes_20_base.zip?dl=1
# Downloading RuAttitudes-2.0-large
curl -L -o "$data_dir/ruattitudes-v2_0_large.zip" https://www.dropbox.com/s/43iqoxlyh38qk8u/ruattitudes_20_large.zip?dl=1