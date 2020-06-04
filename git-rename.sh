#!/bin/bash
remote=$1
old=$2
new=$3
git branch -m $old $new
git push $remote :$old $new
git push $remote -u $new
